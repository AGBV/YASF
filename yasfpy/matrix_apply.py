"""Matrix-apply helpers for simulations.

This module contains functionality extracted from `yasfpy.simulation.Simulation`
related to applying the coupling operator ``W`` and the "master" operator

.. math::

    M = I - T W,

where ``T`` is the block-diagonal particle T-matrix (here: Mie coefficients for
spheres) and ``W`` encodes inter-particle translation/coupling.

The operator form and VSWF translation-based coupling are commonly used in
multiple-sphere scattering formulations; see, for example,
:cite:`Waterman-1971-ID50,Egel-2017-ID1`.
"""

from __future__ import annotations

from time import perf_counter, time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from yasfpy.simulation import Simulation


def coupling_matrix_multiply(sim: Simulation, x: np.ndarray, idx: int | None = None):
    """Apply the coupling operator ``W``.

    Parameters
    ----------
    sim:
        Simulation instance providing a configured ``coupling_backend``.
    x:
        Input vector (flattened VSWF coefficients). The expected shape matches
        the backend: usually ``(jmax,)`` for a single channel or ``(jmax, channels)``
        for multi-wavelength runs.
    idx:
        Optional wavelength/channel index. Some backends use it to select a
        single channel without allocating a full multi-channel output.

    Returns
    -------
    numpy.ndarray
        The product ``W @ x``.

    Notes
    -----
    If per-matvec timing is enabled (``sim._matvec_detail_enabled``), this
    function records the elapsed time under ``sim._matvec_detail['coupling']``.
    """

    if sim._matvec_detail_enabled:
        t0 = perf_counter()
        out = sim.coupling_backend.multiply(x, idx)
        sim._matvec_detail["coupling"] = sim._matvec_detail.get("coupling", 0.0) + (
            perf_counter() - t0
        )
        return out

    return sim.coupling_backend.multiply(x, idx)


def master_matrix_multiply(sim: Simulation, value: np.ndarray, idx: int):
    """Apply the master operator ``M = I - T W``.

    Parameters
    ----------
    sim:
        Simulation instance providing ``mie_coefficients`` and
        ``coupling_backend``.
    value:
        Input vector of unknowns at a given wavelength.
    idx:
        Wavelength/channel index selecting the corresponding diagonal ``T``
        blocks.

    Returns
    -------
    numpy.ndarray
        The product ``(I - T W) @ value``.

    Notes
    -----
    The diagonal application of ``T`` is implemented by elementwise
    multiplication with the raveled Mie coefficient array. This corresponds to a
    block-diagonal T-matrix for independent spherical particles
    :cite:`Bohren-1998-ID178,Mishchenko-2002-ID6`.

    If per-matvec timing is enabled (``sim._matvec_detail_enabled``), this
    function records the elapsed time under ``sim._matvec_detail['t_matrix']``.
    """

    wx = coupling_matrix_multiply(sim, value, idx)

    sim.log.debug("apply T-matrix ...")
    t_matrix_start = time()

    if sim._matvec_detail_enabled:
        t0 = perf_counter()

    twx = (
        sim.mie_coefficients[
            sim.parameters.particles.single_unique_array_idx, :, idx
        ].ravel(order="C")
        * wx
    )
    mx = value - twx

    if sim._matvec_detail_enabled:
        sim._matvec_detail["t_matrix"] = sim._matvec_detail.get("t_matrix", 0.0) + (
            perf_counter() - t0
        )

    t_matrix_stop = time()
    sim.log.debug(f"\t done in {t_matrix_stop - t_matrix_start} seconds.")

    return mx
