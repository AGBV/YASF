"""Protocols for dense coupling operations.

The main simulation code routes dense coupling matvecs via a backend that
implements :meth:`yasfpy.coupling.backends.CouplingBackend.multiply`. For the
"dense" backend, we further delegate to an object implementing the
:class:`DenseCouplingOps` protocol.

This extra level of indirection allows the implementation to:

- choose a CPU or CUDA matvec implementation at runtime
- attach per-`Simulation` persistent buffers/caches for performance

"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class DenseCouplingOps(Protocol):
    """Protocol for applying ``W`` in the dense coupling backend.

    Implementations may maintain persistent device buffers, pinned host buffers,
    and/or autotuning state.
    """

    def matvec(self, x: np.ndarray, idx: int | None = None) -> np.ndarray:
        """Compute ``W @ x``.

        Parameters
        ----------
        x:
            Input vector of unknowns. The expected shape is backend-dependent
            (typically a flattened VSWF coefficient array).
        idx:
            Optional wavelength/channel index.

            - If provided, compute a single wavelength/channel (solver-style
              matvec on a 1D vector).
            - If ``None``, compute all wavelengths at once (multi-wavelength
              matvec, usually producing a 2D array).

        Returns
        -------
        numpy.ndarray
            The product ``W @ x``.
        """
