import numpy as np
from dataclasses import dataclass, field


@dataclass
class RegressionResult:
    w:         np.ndarray
    loss:      float
    risk:      float
    reg:       float
    n_iter:    int
    n_epoch:   int
    n_grad:    int
    time:      float
    converged: bool
    status:    str
    history:   dict = field(default_factory=dict)

    def __str__(self):
        mark = "✓" if self.converged else "✗"
        return (f"{mark} iter={self.n_iter:5d}  epoch={self.n_epoch:4d}  "
                f"grad={self.n_grad:7d}  loss={self.loss:.4e}  "
                f"risk={self.risk:.4e}  reg={self.reg:.4e}  "
                f"t={self.time:.3f}s")
