import yasfpy.log as log

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, bicgstab


class Solver:
    def __init__(
        self,
        solver_type: str = "gmres",
        tolerance: float = 1e-4,
        max_iter: int = 1e4,
        restart: int = 1e2,
    ):
        self.type = solver_type.lower()
        self.tolerance = tolerance
        self.max_iter = int(max_iter)
        self.restart = int(restart)

        self.log = log.scattering_logger(__name__)

    def run(self, a: LinearOperator, b: np.ndarray, x0: np.ndarray = None):
        if x0 is None:
            x0 = np.copy(b)

        if np.any(np.isnan(b)):
            print(b)

        if self.type == "bicgstab":
            counter = GMResCounter(callback_type="x")
            value, err_code = bicgstab(
                a,
                b,
                x0,
                tol=self.tolerance,
                atol=0,
                maxiter=self.max_iter,
                callback=counter,
            )
        elif self.type == "gmres":
            counter = GMResCounter(callback_type="pr_norm")
            value, err_code = gmres(
                a,
                b,
                x0,
                restart=self.restart,
                tol=self.tolerance,
                atol=self.tolerance**2,
                maxiter=self.max_iter,
                callback=counter,
                callback_type="pr_norm",
            )
        elif self.type == "lgmres":
            counter = GMResCounter(callback_type="x")
            value, err_code = lgmres(
                a,
                b,
                x0,
                tol=self.tolerance,
                atol=self.tolerance**2,
                maxiter=self.max_iter,
                callback=counter,
            )
        else:
            self.log.error("Please specify a valid solver type")
            exit(1)

        return value, err_code


class GMResCounter(object):
    def __init__(self, disp=False, callback_type="pr_norm"):
        self.log = log.scattering_logger(__name__)
        self._disp = disp
        self.niter = 0
        if callback_type == "pr_norm":
            self.header = "% 10s \t % 15s" % ("Iteration", "Residual")
        elif callback_type == "x":
            self.header = "% 10s \t %s" % ("Iteration", "Current Iterate")

    def __call__(self, rk=None):
        self.niter += 1
        if isinstance(rk, float):
            msg = "% 10i \t % 15.5f" % (self.niter, rk)
        elif isinstance(rk, np.ndarray):
            msg = "% 10i \t " % self.niter + np.array2string(rk)

        self.log.numerics(self.header)
        self.log.numerics(msg)
        if self._disp:
            print(self.header)
            print(msg)
