import yasfpy.log as log

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, bicgstab


import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, lgmres
from . import log


class Solver:
    """
    The Solver class provides a generic interface for solving linear systems of equations using
    different iterative solvers such as GMRES, BiCGSTAB, and LGMRES, and the GMResCounter class is used
    to count the number of iterations and display the residual or current iterate during the GMRES
    solver.
    """

    def __init__(
        self,
        solver_type: str = "gmres",
        tolerance: float = 1e-4,
        max_iter: int = 1e4,
        restart: int = 1e2,
    ):
        """Initializes a solver object with specified parameters and creates a logger object.

        Args:
            solver_type (str, optional): The type of solver to be used. Defaults to "gmres".
            tolerance (float): The desired accuracy of the solver.
            max_iter (int): The maximum number of iterations that the solver will perform.
            restart (int): The number of iterations after which the solver will restart.
        """
        self.type = solver_type.lower()
        self.tolerance = tolerance
        self.max_iter = int(max_iter)
        self.restart = int(restart)

        self.log = log.scattering_logger(__name__)

    def run(self, a: LinearOperator, b: np.ndarray, x0: np.ndarray = None):
        """
        Runs the solver on the given linear system of equations.

        Args:
            a (LinearOperator): The linear operator representing the system matrix.
            b (np.ndarray): The right-hand side vector.
            x0 (np.ndarray, optional): The initial guess for the solution. If not provided, a copy of b will be used.

        Returns:
            value (np.ndarray): The solution to the linear system of equations.
            err_code (int): The error code indicating the convergence status of the solver.

        """
        if x0 is None:
            x0 = np.copy(b)

        if np.any(np.isnan(b)):
            print(b)

        if self.type == "bicgstab":
            # Add your code here for the bicgstab solver
            pass
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


import numpy as np


class GMResCounter(object):
    """
    The GMResCounter class is a helper class that counts the number of iterations and displays the
    residual or current iterate during the GMRES solver.
    """

    def __init__(self, disp: bool = False, callback_type: str = "pr_norm"):
        """Initializes an object with optional display and callback type parameters.

        Args:
            disp (bool, optional): A boolean flag that determines whether or not to display the progress
                of the algorithm. If `disp` is set to `True`, the algorithm will display the progress.
                If `disp` is set to `False`, the algorithm will not display the progress.
            callback_type (str, optional): The type of callback to be used. It can have two possible values.

        """
        self.log = log.scattering_logger(__name__)
        self._disp = disp
        self.niter = 0
        if callback_type == "pr_norm":
            # self.header = "% 10s \t % 15s" % ("Iteration", "Residual")
            self.header = " Iteration \t        Residual"
        elif callback_type == "x":
            # self.header = "% 10s \t %s" % ("Iteration", "Current Iterate")
            self.header = " Iteration \t Current Iterate"

    def __call__(self, rk=None):
        """The function increments a counter, formats a message based on the input, logs the header and
        message, and prints the header and message if the `_disp` flag is True.

        Args:
            rk (Union[np.ndarray, float]): The parameter `rk` can be either a float or a numpy array.

        """
        self.niter += 1
        if isinstance(rk, float):
            # msg = "% 10i \t % 15.5f" % (self.niter, rk)
            msg = f"{self.niter:10} \t {rk:15.5f}"
        elif isinstance(rk, np.ndarray):
            # msg = "% 10i \t " % self.niter + np.array2string(rk)
            msg = f"{self.niter:10} \t {np.array2string(rk)}"

        self.log.numerics(self.header)
        self.log.numerics(msg)
        if self._disp:
            print(self.header)
            print(msg)
