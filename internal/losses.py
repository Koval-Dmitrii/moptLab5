import numpy as np


class RegressionProblem:
    def __init__(self, Phi, y, l1=0.0, l2=0.0, smooth=1e-6, reg_bias=False):
        self.Phi = np.asarray(Phi, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.m = len(self.y)
        self.p = self.Phi.shape[1]
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.smooth = float(smooth)
        self.mask = np.ones(self.p)
        if not reg_bias:
            self.mask[0] = 0.0

    def residuals(self, w):
        return self.Phi @ w - self.y

    def risk(self, w):
        r = self.residuals(w)
        return float(r @ r) / self.m

    def reg_l1(self, w):
        wm = w * self.mask
        return self.l1 * float(np.sum(np.sqrt(wm * wm + self.smooth ** 2)))

    def reg_l2(self, w):
        wm = w * self.mask
        return self.l2 * float(wm @ wm)

    def reg(self, w):
        return self.reg_l1(w) + self.reg_l2(w)

    def loss(self, w):
        return self.risk(w) + self.reg(w)

    def _reg_grad(self, w):
        wm = w * self.mask
        g1 = self.l1 * (wm / np.sqrt(wm * wm + self.smooth ** 2))
        g2 = 2.0 * self.l2 * wm
        return g1 + g2

    def grad(self, w):
        r = self.residuals(w)
        return (2.0 / self.m) * (self.Phi.T @ r) + self._reg_grad(w)

    def grad_batch(self, w, idx):
        Phi_b = self.Phi[idx]
        r = Phi_b @ w - self.y[idx]
        return (2.0 / len(idx)) * (Phi_b.T @ r) + self._reg_grad(w)

    def hessian(self, w):
        H = (2.0 / self.m) * (self.Phi.T @ self.Phi)
        wm = w * self.mask
        d1 = self.l1 * self.smooth ** 2 / np.power(wm * wm + self.smooth ** 2, 1.5)
        diag = d1 * self.mask + 2.0 * self.l2 * self.mask
        return H + np.diag(diag)

    def jacobian(self):
        return self.Phi
