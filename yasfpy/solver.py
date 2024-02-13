import yasfpy.log as log

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, lgmres, bicgstab


import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, lgmres
from . import log

class Solver:
    """
    A class that represents a solver for linear systems of equations.

    Parameters
    ----------
    solver_type : str, optional
        The type of solver to use. Valid options are "gmres", "bicgstab", and "lgmres".
    tolerance : float, optional
        The tolerance for convergence.
    max_iter : int, optional
        The maximum number of iterations.
    restart : int, optional
        The number of iterations before restarting the GMRES solver.

    Attributes
    ----------
    type : str
        The type of solver.
    tolerance : float
        The tolerance for convergence.
    max_iter : int
        The maximum number of iterations.
    restart : int
        The number of iterations before restarting the GMRES solver.
    log : logger
        The logger for logging solver information.

    Methods
    -------
    run(a, b, x0=None)
        Runs the solver on the given linear system of equations.

    """

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
        """
        Runs the solver on the given linear system of equations.

        Parameters
        ----------
        a : LinearOperator
            The linear operator representing the system matrix.
        b : np.ndarray
            The right-hand side vector.
        x0 : np.ndarray, optional
            The initial guess for the solution. If not provided, a copy of b will be used.

        Returns
        -------
        value : np.ndarray
            The solution to the linear system of equations.
        err_code : int
            The error code indicating the convergence status of the solver.

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
    A class that counts the number of iterations and displays the residual or current iterate during the GMRES solver.

    Parameters:
    - disp (bool): Whether to display the iteration information.
    - callback_type (str): The type of information to display. Can be "pr_norm" for residual or "x" for current iterate.
    """

    def __init__(self, disp=False, callback_type="pr_norm"):
            """
            Initialize the Solver object.

            Parameters:
            - disp (bool): Whether to display intermediate results. Default is False.
            - callback_type (str): The type of callback to use. Possible values are "pr_norm" and "x".
                                   Default is "pr_norm".
            """
            self.log = log.scattering_logger(__name__)
            self._disp = disp
            self.niter = 0
            if callback_type == "pr_norm":
                self.header = "% 10s \t % 15s" % ("Iteration", "Residual")
            elif callback_type == "x":
                self.header = "% 10s \t %s" % ("Iteration", "Current Iterate")

    def __call__(self, rk=None):
        """
        Perform the solver iteration.

        Parameters:
            rk (float or np.ndarray): The residual value or array.

        Returns:
            None
        """
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
