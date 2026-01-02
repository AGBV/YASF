"""Factory for per-simulation dense coupling operations.

The dense coupling backend uses a low-level "ops" object that performs matvecs
and owns any per-simulation caches/buffers.

This module provides a single entry point
:func:`get_dense_coupling_ops` that selects a CPU or CUDA implementation at
runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from yasfpy.coupling.ops import DenseCouplingOps

if TYPE_CHECKING:  # pragma: no cover
    from yasfpy.simulation import Simulation


def get_dense_coupling_ops(sim: "Simulation") -> DenseCouplingOps:
    """Return a per-simulation dense coupling ops instance.

    Parameters
    ----------
    sim:
        Simulation instance.

    Returns
    -------
    DenseCouplingOps
        A CPU or CUDA implementation of :class:`yasfpy.coupling.ops.DenseCouplingOps`.

    Notes
    -----
    The returned object is cached on the simulation instance under
    ``sim._dense_coupling_ops`` so repeated matvec calls reuse internal buffers.

    If ``sim.numerics.gpu`` is true but Numba CUDA is unavailable, the factory
    falls back to CPU ops (and emits a warning).
    """

    cached = getattr(sim, "_dense_coupling_ops", None)
    if cached is not None:
        return cast(DenseCouplingOps, cached)

    if sim.numerics.gpu:
        from numba import cuda

        cuda_ok = cuda.is_available()
        if cuda_ok:
            try:
                cuda.get_current_device()
            except Exception:
                cuda_ok = False

        if cuda_ok:
            from yasfpy.coupling.cuda_dense import CudaDenseCouplingOps

            ops: DenseCouplingOps = CudaDenseCouplingOps(sim)
        else:
            sim.log.warning(
                "Numba CUDA not available; falling back to CPU coupling ops."
            )
            from yasfpy.coupling.cpu_dense import CpuDenseCouplingOps

            ops = CpuDenseCouplingOps(sim)
    else:
        from yasfpy.coupling.cpu_dense import CpuDenseCouplingOps

        ops = CpuDenseCouplingOps(sim)

    setattr(sim, "_dense_coupling_ops", ops)
    return ops
