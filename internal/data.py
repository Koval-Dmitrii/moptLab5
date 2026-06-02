import numpy as np


class Dataset:
    def __init__(self, name, slug, x, y, true_fn, x_range, sigma, formula):
        self.name = name
        self.slug = slug
        self.x = x
        self.y = y
        self.true_fn = true_fn
        self.x_range = x_range
        self.sigma = sigma
        self.formula = formula

    def grid(self, n=400):
        g = np.linspace(self.x_range[0], self.x_range[1], n)
        return g, self.true_fn(g)


def make_near_linear(m=160, sigma=0.35, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = 0.8, -0.5, 3.0
    x_range = (-3.0, 3.0)

    def true_fn(x):
        x = np.asarray(x, dtype=float)
        return a * x + b + 0.1 * np.sin(c * x)

    x = np.sort(rng.uniform(x_range[0], x_range[1], m))
    y = true_fn(x) + rng.normal(0.0, sigma, m)
    formula = "0.8 x - 0.5 + 0.1 sin(3 x)"
    return Dataset("Почти линейная", "linear", x, y, true_fn,
                   x_range, sigma, formula)


def make_nonlinear(m=160, sigma=0.45, seed=1):
    rng = np.random.default_rng(seed)
    x_range = (-3.0, 3.0)

    def true_fn(x):
        x = np.asarray(x, dtype=float)
        return (np.sin(2.0 * x) + 0.15 * x ** 3 - 0.5 * x
                + 1.5 * np.exp(-((x - 1.0) ** 2) / 0.1))

    x = np.sort(rng.uniform(x_range[0], x_range[1], m))
    y = true_fn(x) + rng.normal(0.0, sigma, m)
    formula = "sin(2 x) + 0.15 x^3 - 0.5 x + 1.5 exp(-(x-1)^2 / 0.1)"
    return Dataset("Сильно нелинейная", "nonlinear", x, y, true_fn,
                   x_range, sigma, formula)
