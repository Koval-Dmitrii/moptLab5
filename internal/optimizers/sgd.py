import time
import numpy as np
from ..utils import RegressionResult


def _step_size(lr0, decay, t, schedule):
    if schedule == "constant":
        return lr0
    return lr0 / (1.0 + decay * t)


def minibatch_gd(problem, w0, batch_size=16, lr0=0.1, decay=0.005,
                 n_epochs=300, schedule="inv", tol=1e-10, seed=0,
                 record_every=1):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, dtype=float).copy()
    m = problem.m
    n_grad = 0
    t = 0
    hist = {"loss": [], "risk": [], "reg": [], "grad": [], "epoch": []}
    converged = False
    status = "Достигнут лимит эпох"
    prev = problem.loss(w)

    risk = problem.risk(w); reg = problem.reg(w)
    hist["loss"].append(risk + reg); hist["risk"].append(risk)
    hist["reg"].append(reg); hist["grad"].append(0); hist["epoch"].append(0)

    epoch = 0
    for epoch in range(1, n_epochs + 1):
        order = rng.permutation(m)
        for start in range(0, m, batch_size):
            idx = order[start:start + batch_size]
            g = problem.grad_batch(w, idx)
            lr = _step_size(lr0, decay, t, schedule)
            w = w - lr * g
            n_grad += 1
            t += 1
        if epoch % record_every == 0 or epoch == n_epochs:
            risk = problem.risk(w); reg = problem.reg(w)
            hist["loss"].append(risk + reg); hist["risk"].append(risk)
            hist["reg"].append(reg); hist["grad"].append(n_grad)
            hist["epoch"].append(epoch)
        cur = problem.loss(w)
        if not np.isfinite(cur):
            status = "Расходимость"
            break
        if abs(prev - cur) <= tol * (1.0 + abs(prev)):
            converged = True
            status = "Сходимость по функции потерь"
            break
        prev = cur

    elapsed = time.perf_counter() - t0
    risk = problem.risk(w); reg = problem.reg(w)
    return RegressionResult(
        w, risk + reg, risk, reg, t, epoch, n_grad, elapsed,
        converged, status, hist)


def sgd(problem, w0, lr0=0.05, decay=0.01, n_epochs=300,
        schedule="inv", tol=1e-10, seed=0, record_every=1):
    return minibatch_gd(problem, w0, batch_size=1, lr0=lr0, decay=decay,
                        n_epochs=n_epochs, schedule=schedule, tol=tol,
                        seed=seed, record_every=record_every)
