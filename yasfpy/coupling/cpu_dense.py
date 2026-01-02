"""CPU implementation of the dense coupling matvec.

This module implements :class:`yasfpy.coupling.ops.DenseCouplingOps` using the CPU
Numba kernels in :mod:`yasfpy.functions.cpu_numba`. It is used when
``numerics.gpu`` is disabled but a dense coupling backend is selected.

Notes
-----
The underlying coupling operator follows a VSWF translation formulation commonly
used in multiple-scattering approaches :cite:`Waterman-1971-ID50` and in CELES
:cite:`Egel-2017-ID1`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from yasfpy.coupling.ops import DenseCouplingOps
from yasfpy.functions.cpu_numba import particle_interaction, particle_interaction_scalar

if TYPE_CHECKING:  # pragma: no cover
    from yasfpy.simulation import Simulation


class CpuDenseCouplingOps(DenseCouplingOps):
    """CPU dense coupling linear operator.

    This class provides the dense coupling matvec ``W @ x`` using CPU Numba kernels.

    Parameters
    ----------
    sim:
        Simulation instance providing precomputed pairwise lookup tables.

    Notes
    -----
    The Numba kernels are sensitive to memory layout; contiguous lookup table
    buffers are stored for the lifetime of this instance.
    """

    def __init__(self, sim: "Simulation"):
        """Construct the operator and cache contiguous lookup buffers."""

        self._sim = sim

        lmax = sim.numerics.lmax
        particle_number = sim.parameters.particles.number
        nmax = 2 * lmax * (lmax + 2)
        self._jmax = int(particle_number * nmax)

        if sim.plm is None or sim.sph_h is None or sim.e_j_dm_phi is None:
            raise RuntimeError(
                "Dense coupling backend requires pairwise lookup tables but they are missing."
            )

        # CPU Numba kernels are sensitive to memory layout.
        # Keep contiguous buffers for the lifetime of the Simulation.
        self._translation_table = np.ascontiguousarray(sim.numerics.translation_ab5)
        self._plm = np.ascontiguousarray(sim.plm)
        self._sph_h = np.ascontiguousarray(sim.sph_h)
        self._e_phi = np.ascontiguousarray(sim.e_j_dm_phi)
        self._idx = np.ascontiguousarray(sim.idx_lookup)

    def matvec(self, x: np.ndarray, idx: int | None = None) -> np.ndarray:
        """Compute the dense coupling matvec.

        Parameters
        ----------
        x:
            Input array. For multi-wavelength usage, the trailing dimension is the
            wavelength/channel axis.
        idx:
            Optional wavelength/channel index. When provided, the computation is
            restricted to a single channel.

        Returns
        -------
        numpy.ndarray
            The product ``W @ x``. If ``idx`` is not ``None``, the result is
            squeezed to drop the singleton channel axis.
        """

        x = np.ascontiguousarray(x)

        sph_h = self._sph_h
        if idx is not None:
            # Use a 4D slice to preserve the channel axis.
            sph_h = np.ascontiguousarray(sph_h[:, :, :, idx : idx + 1])

        if sph_h.shape[-1] == 1:
            wx = particle_interaction_scalar(
                self._sim.numerics.lmax,
                self._sim.parameters.particles.number,
                self._idx,
                x,
                self._translation_table,
                self._plm,
                sph_h,
                self._e_phi,
            )
            if idx is None:
                wx = wx.reshape((self._jmax, 1))
        else:
            wx = particle_interaction(
                self._sim.numerics.lmax,
                self._sim.parameters.particles.number,
                self._idx,
                x,
                self._translation_table,
                self._plm,
                sph_h,
                self._e_phi,
            )

        if idx is not None:
            wx = np.squeeze(wx)

        return wx
