import time
import numpy as np
from ..utils import RegressionResult


def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def gauss_newton(problem, w0, max_iter=100, tol=1e-10):
    t0 = time.perf_counter()
    w = np.asarray(w0, dtype=float).copy()
    n_grad = 0
    hist = {"loss": [], "risk": [], "reg": [], "grad": [], "iter": []}
    converged = False
    status = "Достигнут лимит итераций"
    k = 0
    for k in range(max_iter):
        g = problem.grad(w); n_grad += 1
        risk = problem.risk(w); reg = problem.reg(w)
        hist["loss"].append(risk + reg); hist["risk"].append(risk)
        hist["reg"].append(reg); hist["grad"].append(n_grad); hist["iter"].append(k)
        if np.linalg.norm(g) < tol:
            converged = True
            status = "Сходимость по градиенту"
            break
        H = problem.hessian(w)
        w = w - _solve(H, g)

    elapsed = time.perf_counter() - t0
    risk = problem.risk(w); reg = problem.reg(w)
    return RegressionResult(
        w, risk + reg, risk, reg, k + 1, 0, n_grad, elapsed,
        converged, status, hist)


def levenberg_marquardt(problem, w0, max_iter=100, tol=1e-10,
                        mu0=1e-2, nu=3.0):
    t0 = time.perf_counter()
    w = np.asarray(w0, dtype=float).copy()
    mu = mu0
    n_grad = 0
    hist = {"loss": [], "risk": [], "reg": [], "grad": [], "iter": [], "mu": []}
    converged = False
    status = "Достигнут лимит итераций"
    fcur = problem.loss(w)
    I = np.eye(len(w))
    k = 0
    for k in range(max_iter):
        g = problem.grad(w); n_grad += 1
        risk = problem.risk(w); reg = problem.reg(w)
        hist["loss"].append(risk + reg); hist["risk"].append(risk)
        hist["reg"].append(reg); hist["grad"].append(n_grad)
        hist["iter"].append(k); hist["mu"].append(mu)
        if np.linalg.norm(g) < tol:
            converged = True
            status = "Сходимость по градиенту"
            break
        H = problem.hessian(w)
        accepted = False
        for _ in range(40):
            step = _solve(H + mu * I, g)
            w_new = w - step
            f_new = problem.loss(w_new)
            if np.isfinite(f_new) and f_new < fcur:
                w = w_new
                fcur = f_new
                mu = max(mu / nu, 1e-12)
                accepted = True
                break
            mu = min(mu * nu, 1e12)
        if not accepted:
            if np.linalg.norm(problem.grad(w)) < 1e-6:
                converged = True
                status = "Сходимость (шаг не уменьшает потери)"
            else:
                status = "Шаг не принят (LM)"
            break

    elapsed = time.perf_counter() - t0
    risk = problem.risk(w); reg = problem.reg(w)
    return RegressionResult(
        w, risk + reg, risk, reg, k + 1, 0, n_grad, elapsed,
        converged, status, hist)
