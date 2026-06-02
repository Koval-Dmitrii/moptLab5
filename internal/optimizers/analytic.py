import time
import numpy as np
from ..utils import RegressionResult


def analytic_linear_1d(problem, w0=None):
    t0 = time.perf_counter()
    z = problem.Phi[:, 1]
    y = problem.y
    zbar = float(np.mean(z))
    ybar = float(np.mean(y))
    denom = float(np.sum((z - zbar) ** 2))
    w1 = float(np.sum((z - zbar) * (y - ybar)) / denom)
    w0_ = ybar - w1 * zbar
    w = np.array([w0_, w1])
    elapsed = time.perf_counter() - t0
    risk = problem.risk(w)
    reg = problem.reg(w)
    hist = {"loss": [risk + reg], "risk": [risk], "reg": [reg],
            "grad": [0], "iter": [0]}
    return RegressionResult(
        w, risk + reg, risk, reg, 1, 0, 0, elapsed, True,
        "Аналитическое решение (оценки средних)", hist)
