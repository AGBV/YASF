"""Linear-system solve for scattered field coefficients.

This module contains the implementation that used to live in
`yasfpy.simulation.Simulation.compute_scattered_field_coefficients`, extracted
to keep `simulation.py` smaller.

In the multiple-scattering formulation used by YASF, the unknown scattered-field
expansion coefficients are obtained by solving a linear system involving the
"master" operator

.. math::

    M = I - T W,

where ``T`` is the (block-diagonal) particle T-matrix and ``W`` is the coupling
operator encoding translations between particle-centered VSWF bases.

Notes
-----
This module does not implement a specific Krylov method itself; it wraps the
solver configured in ``sim.numerics.solver`` and provides a SciPy
``LinearOperator`` matvec that delegates to ``sim.master_matrix_multiply``.

References
----------
Multiple-sphere scattering via VSWF bases and coupling operators is described
in :cite:`Waterman-1971-ID50` and, in a modern implementation context,
:cite:`Egel-2017-ID1`.
"""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse.linalg._interface import _CustomLinearOperator

if TYPE_CHECKING:
    from yasfpy.simulation import Simulation


def compute_scattered_field_coefficients(
    sim: Simulation, guess: np.ndarray | None = None
) -> None:
    """Solve for scattered-field expansion coefficients.

    Parameters
    ----------
    sim:
        Simulation instance with ``right_hand_side`` already assembled and a
        configured solver in ``sim.numerics.solver``.
    guess:
        Optional initial guess for the scattered-field coefficients.

        - If provided, ``guess[:, :, w]`` is used for each wavelength ``w``.
        - If not provided, the implementation defaults to starting from the RHS.
          If enabled, subsequent wavelengths may warm-start from the previous
          wavelength's solution.

    Returns
    -------
    None

    Notes
    -----
    This routine loops over wavelengths and builds a SciPy
    :class:`scipy.sparse.linalg.LinearOperator` compatible matvec (implemented via
    SciPy's internal ``_CustomLinearOperator`` wrapper).

    For each wavelength ``w``, it solves:

    .. math::

        (I - T W) x_w = b_w,

    where ``b_w`` comes from ``sim.right_hand_side`` and the matvec routes through
    :meth:`yasfpy.simulation.Simulation.master_matrix_multiply`.

    Solver statistics (matvec call count and time) are logged. When
    ``sim._matvec_detail_enabled`` is set, a timing breakdown of selected
    sub-operations is logged at DEBUG level.
    """

    sim.log.info("compute scattered field coefficients ...")
    jmax = sim.parameters.particles.number * sim.numerics.nmax
    sim.scattered_field_coefficients = np.zeros_like(sim.initial_field_coefficients)
    sim.scattered_field_err_codes = np.zeros(sim.parameters.wavelengths_number)

    # If `guess` is not provided, we default to using the RHS as starting point.
    # Additionally, if enabled, we warm-start subsequent wavelengths using the
    # previous wavelength's solution.
    # TODO: Look into performing this loop in parallel
    prev_solution: np.ndarray | None = None
    for w in range(sim.parameters.wavelengths_number):
        matvec_calls = 0
        matvec_time = 0.0

        def _matvec(x: np.ndarray) -> np.ndarray:
            nonlocal matvec_calls, matvec_time
            matvec_calls += 1
            t0 = time()

            if sim._matvec_detail_enabled:
                sim._matvec_detail = {}
                sim._matvec_detail["calls"] = float(
                    sim._matvec_detail.get("calls", 0.0) + 1.0
                )

            out = sim.master_matrix_multiply(x, w)
            matvec_time += time() - t0

            if sim._matvec_detail_enabled:
                details = ", ".join(
                    f"{k}={v:.3f}s" for k, v in sorted(sim._matvec_detail.items())
                )
                # Only show breakdown at DEBUG to avoid log spam.
                sim.log.debug("Matvec breakdown w=%d: %s", w, details)

            return out

        A = _CustomLinearOperator((jmax, jmax), matvec=_matvec, dtype=complex)
        b = sim.right_hand_side[:, :, w].ravel()

        warm_start_enabled = bool(getattr(sim.numerics.solver, "warm_start", True))

        # Initial guess selection:
        # - If user provided a guess, always use it.
        # - Otherwise, for w>0 and warm-start enabled, reuse previous wavelength solution.
        # - Otherwise, start from RHS.
        if guess is not None:
            x0 = guess[:, :, w].ravel()
        elif warm_start_enabled and prev_solution is not None:
            x0 = prev_solution
        else:
            x0 = b

        sim.log.info("Solver run %d/%d" % (w + 1, sim.parameters.wavelengths_number))
        x, err_code = sim.numerics.solver.run(A, b, x0)
        sim.scattered_field_coefficients[:, :, w] = x.reshape(
            sim.right_hand_side.shape[:2]
        )
        sim.scattered_field_err_codes[w] = err_code
        prev_solution = x

        sim.log.info(
            "Solver stats w=%d: matvec_calls=%d matvec_time=%.3fs",
            w,
            matvec_calls,
            matvec_time,
        )
        sim.log.debug(
            "Solver detail w=%d: avg_matvec=%.3fs",
            w,
            (matvec_time / matvec_calls) if matvec_calls else 0.0,
        )
