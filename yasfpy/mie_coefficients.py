"""Mie coefficient generation utilities.

This module contains the implementation that used to live in
`yasfpy.simulation.Simulation.compute_mie_coefficients`, extracted to keep
`simulation.py` smaller.

The coefficients computed here are the diagonal entries of each particle's
T-matrix in a vector-spherical-wave-function (VSWF) basis for homogeneous
spheres (classical Mie scattering):

- ``tau = 1`` corresponds to one polarization family (often "TE"/"magnetic")
- ``tau = 2`` corresponds to the other polarization family (often "TM"/"electric")

The formulas follow standard Mie theory; see, e.g., :cite:`Bohren-1998-ID178` or
:cite:`Mishchenko-2002-ID6`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import spherical_jn, spherical_yn

if TYPE_CHECKING:
    from yasfpy.simulation import Simulation


def compute_mie_coefficients(sim: Simulation) -> None:
    """Compute diagonal Mie coefficients for unique (radius, material) pairs.

    Parameters
    ----------
    sim:
        Simulation instance providing medium wavenumbers and particle material
        refractive indices. The results are stored on ``sim``.

    Returns
    -------
    None

    Notes
    -----
    This routine computes the diagonal spherical-particle T-matrix entries
    (Mie coefficients) for each unique (radius, material) pair found in
    ``sim.parameters.particles``.

    The legacy implementation in YASF called ``t_entry()`` for every
    ``(pair, tau, l)`` and then repeated the scalar coefficient for all
    ``m = -l..l``. In typical geometries with many distinct radii, that created
    substantial overhead from repeated special-function evaluations.

    This implementation reduces overhead by vectorizing the special-function
    evaluation:

    - evaluates spherical Bessel/Hankel functions for all ``l = 1..lmax`` on an
      outer ``(l, channel)`` grid
    - fills the ``(tau, l, m)`` blocks by slicing (no inner ``m`` loop)

    The math is unchanged; only the computational structure differs.

    The additional array ``sim.scatter_to_internal`` stores the ratio between
    internal-field and scattered-field coefficients used elsewhere in the
    solver.

    References
    ----------
    Standard Mie scattering theory for homogeneous spheres:
    :cite:`Bohren-1998-ID178,Mishchenko-2002-ID6`.
    """

    lmax = sim.numerics.lmax
    nmax = sim.numerics.nmax
    channels = sim.parameters.wavelength.shape[0]

    sim.mie_coefficients = np.zeros(
        (
            sim.parameters.particles.num_unique_pairs,
            nmax,
            channels,
        ),
        dtype=complex,
    )

    sim.scatter_to_internal = np.zeros_like(sim.mie_coefficients)

    l_orders = np.arange(1, lmax + 1)
    k_medium = sim.parameters.k_medium

    for unique_pair_index in range(sim.parameters.particles.num_unique_pairs):
        k_sphere = (
            sim.parameters.omega
            * sim.parameters.particles.unique_radius_index_pairs[unique_pair_index, 1:]
        )
        radius = float(
            np.real(
                sim.parameters.particles.unique_radius_index_pairs[unique_pair_index, 0]
            )
        )

        m = k_sphere / k_medium
        x = np.atleast_1d(k_medium) * radius
        mx = np.atleast_1d(k_sphere) * radius

        # SciPy broadcasts `n` and `z` together. When both are 1D with the
        # same length it performs elementwise evaluation; reshape to get the
        # outer (l, channel) grid.
        l_grid = l_orders[:, None]
        x_grid = x[None, :]
        mx_grid = mx[None, :]

        jx = spherical_jn(l_grid, x_grid)
        jx_prime = spherical_jn(l_grid, x_grid, derivative=True)
        yx = spherical_yn(l_grid, x_grid)
        yx_prime = spherical_yn(l_grid, x_grid, derivative=True)

        jmx = spherical_jn(l_grid, mx_grid)
        jmx_prime = spherical_jn(l_grid, mx_grid, derivative=True)

        hx = jx + 1j * yx
        hx_prime = jx_prime + 1j * yx_prime

        # Riccati-Bessel derivatives: d/dz [z * f_l(z)]
        djx = jx + x * jx_prime
        djmx = jmx + mx * jmx_prime
        dhx = hx + x * hx_prime

        # Shape: (lmax, channels)
        numer_common = jmx * djx - jx * djmx
        denom_common = jmx * dhx - hx * djmx

        # scattered
        mie_tau1 = -numer_common / denom_common
        mie_tau2 = -(m**2 * jmx * djx - jx * djmx) / (m**2 * jmx * dhx - hx * djmx)

        # ratio (internal / scattered)
        ratio_tau1 = (jx * dhx - hx * djx) / (-numer_common)
        ratio_tau2 = (m * jx * dhx - m * hx * djx) / (-(m**2 * jmx * djx - jx * djmx))

        # Fill per-(tau, l) blocks; m just replicates the coefficient.
        for l_idx, l in enumerate(l_orders):
            width = 2 * int(l) + 1

            j_start_tau1 = 0 + int(l) * int(l) - 1
            j_end_tau1 = j_start_tau1 + width
            sim.mie_coefficients[unique_pair_index, j_start_tau1:j_end_tau1, :] = (
                mie_tau1[l_idx, :]
            )
            sim.scatter_to_internal[unique_pair_index, j_start_tau1:j_end_tau1, :] = (
                ratio_tau1[l_idx, :]
            )

            tau2_offset = lmax * (lmax + 2)
            j_start_tau2 = tau2_offset + int(l) * int(l) - 1
            j_end_tau2 = j_start_tau2 + width
            sim.mie_coefficients[unique_pair_index, j_start_tau2:j_end_tau2, :] = (
                mie_tau2[l_idx, :]
            )
            sim.scatter_to_internal[unique_pair_index, j_start_tau2:j_end_tau2, :] = (
                ratio_tau2[l_idx, :]
            )
