import numpy as np


class PolynomialModel:
    def __init__(self, degree):
        self.degree = degree
        self.n_params = degree + 1
        self.mean = 0.0
        self.std = 1.0
        self.col_mean = np.zeros(self.n_params)
        self.col_std = np.ones(self.n_params)
        self.name = "Линейная" if degree == 1 else f"Полином степени {degree}"

    def _raw_design(self, x):
        z = (np.asarray(x, dtype=float) - self.mean) / self.std
        return np.vander(z, self.n_params, increasing=True)

    def fit(self, x):
        self.mean = float(np.mean(x))
        self.std = float(np.std(x)) or 1.0
        z = self._raw_design(x)
        cm = z.mean(axis=0)
        cs = z.std(axis=0)
        cm[0] = 0.0
        cs[0] = 1.0
        cs[cs == 0.0] = 1.0
        self.col_mean = cm
        self.col_std = cs
        return self

    def design(self, x):
        z = self._raw_design(x)
        return (z - self.col_mean) / self.col_std

    def predict(self, x, w):
        return self.design(x) @ w


class LinearModel(PolynomialModel):
    def __init__(self):
        super().__init__(1)
