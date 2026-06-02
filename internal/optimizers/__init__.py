from .analytic import analytic_linear_1d
from .sgd import sgd, minibatch_gd
from .gauss_newton import gauss_newton, levenberg_marquardt

__all__ = [
    'analytic_linear_1d',
    'sgd',
    'minibatch_gd',
    'gauss_newton',
    'levenberg_marquardt',
]
