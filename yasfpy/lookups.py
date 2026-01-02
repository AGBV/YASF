"""Lookup-table generation for simulations.

This module contains functionality extracted from `yasfpy.simulation.Simulation`
so that `simulation.py` can stay focused on orchestration.

The lookup tables produced here are used by coupling backends to avoid repeated
special-function evaluation and to precompute geometric terms (relative
positions, azimuthal phases, and associated Legendre polynomials) for the
vector-spherical-wave-function (VSWF) translation operators.

The ``legacy_*`` helpers are preserved as-is; they originate from the Matlab
CELES framework :cite:`Egel-2017-ID1` and are not used by YASF in typical runs.
"""

from __future__ import annotations

from time import time
from typing import Any

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from yasfpy.functions.cpu_numba import compute_idx_lookups
from yasfpy.functions.misc import mutual_lookup


def legacy_compute_lookup_particle_distances(sim: Any) -> None:
    """Compute distance sampling points for a legacy Hankel lookup table.

    Parameters
    ----------
    sim:
        Simulation-like object that will receive a ``lookup_particle_distances``
        attribute.

    Notes
    -----
    This helper exists for parity with legacy CELES-style input generation and
    YAML/Matlab compatibility :cite:`Egel-2017-ID1`.

    The resulting distances include an extra leading ``0`` to keep the original
    interpolation scheme stable near the origin.
    """

    # add two zeros at the beginning to allow interpolation
    # also in the first segment
    step = sim.numerics.particle_distance_resolution
    maxdist = (
        sim.parameters.particles.max_particle_distance
        + 3 * sim.numerics.particle_distance_resolution
    )
    sim.lookup_particle_distances = np.concatenate(
        (np.array([0]), np.arange(0, maxdist + np.finfo(float).eps, step))
    )


def legacy_compute_h3_table(sim: Any) -> None:
    """Compute a legacy spherical Hankel lookup table.

    Parameters
    ----------
    sim:
        Simulation-like object with ``lookup_particle_distances`` and
        wavelength-dependent medium wavenumbers ``parameters.k_medium``.

    Notes
    -----
    Populates ``sim.h3_table`` with values of

    .. math::

        h_l^{(1)}(k r) = j_l(k r) + i y_l(k r),

    for ``l = 0 .. 2*lmax`` and the configured sampling radii.

    This is kept only for compatibility with legacy CELES-derived workflows
    :cite:`Egel-2017-ID1`.
    """

    sim.h3_table = np.zeros(
        (
            2 * sim.numerics.lmax + 1,
            sim.lookup_particle_distances.shape[0],
            sim.parameters.medium_refractive_index.shape[0],
        ),
        dtype=complex,
    )
    size_param = np.outer(sim.lookup_particle_distances, sim.parameters.k_medium)

    for p in range(2 * sim.numerics.lmax + 1):
        sim.h3_table[p, :, :] = spherical_jn(p, size_param) + 1j * spherical_yn(
            p, size_param
        )


def compute_idx_lookup(sim: Any) -> None:
    """Create the (particle, n, tau, l, m) index lookup table.

    Parameters
    ----------
    sim:
        Simulation-like object that provides ``numerics.lmax`` and
        ``parameters.particles.number`` and will receive an ``idx_lookup``
        attribute.

    Notes
    -----
    The ``idx_lookup`` table maps the flattened unknown index ``j`` to a tuple
    ``(s, n, tau, l, m)`` (particle index, VSWF mode index, polarization, degree,
    order). It is used by dense coupling backends to turn the matvec

    .. math::

        (W x)_j = \\sum_k W_{j k} x_k

    into structured loops over VSWF indices.
    """

    sim.idx_lookup = compute_idx_lookups(
        sim.numerics.lmax, sim.parameters.particles.number
    )


def compute_lookups(sim: Any) -> None:
    """Compute pairwise lookup tables used by coupling backends.

    Parameters
    ----------
    sim:
        Simulation-like object providing ``numerics.lmax``, particle positions,
        and medium wavenumbers (``parameters.k_medium``). The function writes the
        lookup arrays onto ``sim``.

    Notes
    -----
    This function calls :func:`yasfpy.functions.misc.mutual_lookup` to compute a
    set of geometric and special-function tables for all particle pairs.

    The produced attributes:

    - ``sim.sph_j``: spherical Bessel functions ``j_l(k r)``.
    - ``sim.sph_h``: spherical Hankel functions ``h_l^{(1)}(k r)``.
    - ``sim.e_j_dm_phi``: azimuthal phase factors ``exp(i (m-m') phi)``.
    - ``sim.plm``: normalized associated Legendre terms.

    These terms are building blocks for VSWF translation operators and are used
    by the dense coupling implementations (CPU/Numba and CUDA) to assemble or
    apply inter-particle interaction blocks.

    References
    ----------
    The overall CELES-style formulation using VSWFs and pairwise translation is
    described in :cite:`Egel-2017-ID1`.
    """

    lookup_computation_time_start = time()

    # mutual_lookup returns additional arrays (distance grids, etc.). The dense
    # coupling backends only need these core building blocks.
    sim.sph_j, sim.sph_h, sim.e_j_dm_phi, sim.plm = mutual_lookup(
        sim.numerics.lmax,
        sim.parameters.particles.position,
        sim.parameters.particles.position,
        sim.parameters.k_medium,
    )[:4]

    lookup_computation_time_stop = time()
    sim.log.info(
        "Computing lookup tables took %f s"
        % (lookup_computation_time_stop - lookup_computation_time_start)
    )
