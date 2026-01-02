"""CPU kernels accelerated with Numba.

This module contains CPU implementations of inner loops used for coupling and
field/polarization calculations.
"""

import numpy as np
from typing import TYPE_CHECKING

from numba import jit
from scipy.special import hankel1, lpmv, spherical_jn

from yasfpy.functions.misc import single_index2multi

if TYPE_CHECKING:
    prange = range  # type: ignore[assignment]
else:
    from numba import prange


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def particle_interaction_scalar(
    lmax: int,
    particle_number: int,
    idx: np.ndarray,
    x: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi,
):
    """Compute Wx = coupling_matrix @ x for a single channel.

    This specialization avoids per-row array allocations and inner loops over
    the wavelength axis, which noticeably helps the common `channels==1` case.
    """

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax

    # Column views reduce repeated idx[j, :] row slicing overhead.
    idx_s = idx[:, 0]
    idx_n = idx[:, 1]
    idx_tau = idx[:, 2]
    idx_l = idx[:, 3]
    idx_m = idx[:, 4]

    wx = np.zeros(jmax, dtype=np.complex128)

    for j1 in prange(jmax):
        s1 = idx_s[j1]
        n1 = idx_n[j1]
        tau1 = idx_tau[j1]
        l1 = idx_l[j1]
        m1 = idx_m[j1]

        acc = 0.0j

        for s2 in range(particle_number):
            if s2 == s1:
                continue

            base = s2 * nmax
            end = base + nmax
            for j2 in range(base, end):
                n2 = idx_n[j2]
                tau2 = idx_tau[j2]
                l2 = idx_l[j2]
                m2 = idx_m[j2]

                delta_tau = abs(tau1 - tau2)
                delta_l = abs(l1 - l2)
                dm = m1 - m2
                if dm < 0:
                    dm = -dm
                delta_m = dm

                p_start = delta_m
                tmp = delta_l + delta_tau
                if tmp > p_start:
                    p_start = tmp

                phase_x = e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

                for p in range(p_start, l1 + l2 + 1):
                    acc += (
                        translation_table[n2, n1, p]
                        * plm[p * (p + 1) // 2 + delta_m, s1, s2]
                        * phase_x
                        * sph_h[p, s1, s2, 0]
                    )

        wx[j1] = acc

    return wx


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def particle_interaction(
    lmax: int,
    particle_number: int,
    idx: np.ndarray,
    x: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi,
):
    """Compute Wx = coupling_matrix @ x.

    Notes
    -----
    This routine is performance-critical. The implementation is structured so
    that the outer `prange` writes to disjoint output rows, which allows Numba
    to parallelize safely.

    The implementation avoids redundant work by:
    - computing per-(j1, j2) scalars once (not once per channel)
    - skipping same-particle blocks by construction
    """

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax
    channels = sph_h.shape[-1]

    # Column views reduce repeated idx[j, :] row slicing overhead.
    idx_s = idx[:, 0]
    idx_n = idx[:, 1]
    idx_tau = idx[:, 2]
    idx_l = idx[:, 3]
    idx_m = idx[:, 4]

    wx = np.zeros((jmax, channels), dtype=np.complex128)

    for j1 in prange(jmax):
        s1 = idx_s[j1]
        n1 = idx_n[j1]
        tau1 = idx_tau[j1]
        l1 = idx_l[j1]
        m1 = idx_m[j1]

        acc = np.zeros(channels, dtype=np.complex128)

        for s2 in range(particle_number):
            if s2 == s1:
                continue

            base = s2 * nmax
            end = base + nmax
            for j2 in range(base, end):
                n2 = idx_n[j2]
                tau2 = idx_tau[j2]
                l2 = idx_l[j2]
                m2 = idx_m[j2]

                delta_tau = abs(tau1 - tau2)
                delta_l = abs(l1 - l2)
                dm = m1 - m2
                if dm < 0:
                    dm = -dm
                delta_m = dm

                p_start = delta_m
                tmp = delta_l + delta_tau
                if tmp > p_start:
                    p_start = tmp

                phase_x = e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

                for p in range(p_start, l1 + l2 + 1):
                    coeff_phase = (
                        translation_table[n2, n1, p]
                        * plm[p * (p + 1) // 2 + delta_m, s1, s2]
                        * phase_x
                    )
                    for w in range(channels):
                        acc[w] += coeff_phase * sph_h[p, s1, s2, w]

        wx[j1, :] = acc

    return wx


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def particle_interaction_tiled(
    lmax: int,
    tile_s1_start: int,
    tile_s1_end: int,
    tile_s2_start: int,
    tile_s2_end: int,
    idx: np.ndarray,
    x: np.ndarray,
    translation_table: np.ndarray,
    plm_tile: np.ndarray,
    sph_h_tile: np.ndarray,
    e_j_dm_phi_tile: np.ndarray,
    wx: np.ndarray,
):
    """Accumulate a coupling-matvec block into wx.

    This is the tiled counterpart to `particle_interaction`.

    Parameters
    ----------
    tile_s1_start/tile_s1_end:
        Global particle index range for output rows.
    tile_s2_start/tile_s2_end:
        Global particle index range for input columns.

    Notes
    -----
    `plm_tile`, `sph_h_tile`, and `e_j_dm_phi_tile` must be computed for the
    cross product of the two particle ranges (same ordering as mutual_lookup).
    """

    nmax = 2 * lmax * (lmax + 2)
    channels = sph_h_tile.shape[-1]

    idx_n = idx[:, 1]
    idx_tau = idx[:, 2]
    idx_l = idx[:, 3]
    idx_m = idx[:, 4]

    s1_count = tile_s1_end - tile_s1_start

    for j1_local in prange(s1_count * nmax):
        j1 = tile_s1_start * nmax + j1_local
        s1 = tile_s1_start + (j1_local // nmax)
        s1_local = s1 - tile_s1_start

        n1 = idx_n[j1]
        tau1 = idx_tau[j1]
        l1 = idx_l[j1]
        m1 = idx_m[j1]

        acc = np.zeros(channels, dtype=np.complex128)

        for s2 in range(tile_s2_start, tile_s2_end):
            if s2 == s1:
                continue

            s2_local = s2 - tile_s2_start
            base = s2 * nmax
            end = base + nmax

            for j2 in range(base, end):
                n2 = idx_n[j2]
                tau2 = idx_tau[j2]
                l2 = idx_l[j2]
                m2 = idx_m[j2]

                delta_tau = abs(tau1 - tau2)
                delta_l = abs(l1 - l2)

                dm = m1 - m2
                if dm < 0:
                    dm = -dm
                delta_m = dm

                p_start = delta_m
                tmp = delta_l + delta_tau
                if tmp > p_start:
                    p_start = tmp

                phase_x = (
                    e_j_dm_phi_tile[m2 - m1 + 2 * lmax, s1_local, s2_local] * x[j2]
                )

                for p in range(p_start, l1 + l2 + 1):
                    coeff_phase = (
                        translation_table[n2, n1, p]
                        * plm_tile[p * (p + 1) // 2 + delta_m, s1_local, s2_local]
                        * phase_x
                    )
                    for w in range(channels):
                        acc[w] += coeff_phase * sph_h_tile[p, s1_local, s2_local, w]

        wx[j1, :] += acc


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def particle_interaction_sparse(
    lmax: int,
    tile_s1_start: int,
    tile_s1_end: int,
    s2_indices: np.ndarray,
    near_mask: np.ndarray,
    idx: np.ndarray,
    x: np.ndarray,
    translation_table: np.ndarray,
    plm_tile: np.ndarray,
    sph_h_tile: np.ndarray,
    e_j_dm_phi_tile: np.ndarray,
    wx: np.ndarray,
):
    """Accumulate a coupling-matvec block into wx for sparse targets.

    `s2_indices` lists which global particle blocks are included. `near_mask`
    controls which (s1_local, s2_local) pairs contribute.
    """
    nmax = 2 * lmax * (lmax + 2)
    channels = sph_h_tile.shape[-1]

    idx_n = idx[:, 1]
    idx_tau = idx[:, 2]
    idx_l = idx[:, 3]
    idx_m = idx[:, 4]

    s1_count = tile_s1_end - tile_s1_start
    s2_count = s2_indices.shape[0]

    for j1_local in prange(s1_count * nmax):
        j1 = tile_s1_start * nmax + j1_local
        s1 = tile_s1_start + (j1_local // nmax)
        s1_local = s1 - tile_s1_start

        n1 = idx_n[j1]
        tau1 = idx_tau[j1]
        l1 = idx_l[j1]
        m1 = idx_m[j1]

        acc = np.zeros(channels, dtype=np.complex128)

        for s2_local in range(s2_count):
            if not near_mask[s1_local, s2_local]:
                continue

            s2 = s2_indices[s2_local]
            if s2 == s1:
                continue

            base = s2 * nmax
            end = base + nmax

            for j2 in range(base, end):
                n2 = idx_n[j2]
                tau2 = idx_tau[j2]
                l2 = idx_l[j2]
                m2 = idx_m[j2]

                delta_tau = abs(tau1 - tau2)
                delta_l = abs(l1 - l2)

                dm = m1 - m2
                if dm < 0:
                    dm = -dm
                delta_m = dm

                p_start = delta_m
                tmp = delta_l + delta_tau
                if tmp > p_start:
                    p_start = tmp

                phase_x = (
                    e_j_dm_phi_tile[m2 - m1 + 2 * lmax, s1_local, s2_local] * x[j2]
                )

                for p in range(p_start, l1 + l2 + 1):
                    coeff_phase = (
                        translation_table[n2, n1, p]
                        * plm_tile[p * (p + 1) // 2 + delta_m, s1_local, s2_local]
                        * phase_x
                    )
                    for w in range(channels):
                        acc[w] += coeff_phase * sph_h_tile[p, s1_local, s2_local, w]

        wx[j1, :] += acc


@jit(nopython=True, parallel=True, fastmath=True)
def compute_idx_lookups(lmax: int, particle_number: int):
    r"""Create the (particle, n, tau, l, m) lookup table.

    Notes
    -----
    YASF flattens the unknown vector by stacking all vector-spherical-wave-function
    (VSWF) coefficients for each particle. For a fixed particle ``s`` we enumerate
    the modes using ``(tau, l, m)`` with ``tau in {1, 2}``, ``l = 1..lmax``, and
    ``m = -l..l``.

    The per-particle mode index ``n`` follows the CELES ordering

    .. math::

        n = (\tau - 1)\,l_{\max}(l_{\max}+2) + (l-1)(l+1) + l + m.

    The global flat index is then ``i = s * nmax + n`` with
    ``nmax = 2 * lmax * (lmax + 2)``.
    """
    nmax = 2 * lmax * (lmax + 2)
    idx = np.zeros(nmax * particle_number * 5, dtype=np.int64).reshape(
        (nmax * particle_number, 5)
    )

    for s in prange(particle_number):
        for tau in range(1, 3):
            for l in range(1, lmax + 1):
                for m in range(-l, l + 1):
                    n = (tau - 1) * lmax * (lmax + 2) + (l - 1) * (l + 1) + l + m
                    i = n + s * nmax
                    idx[i, 0] = s
                    idx[i, 1] = n
                    idx[i, 2] = tau
                    idx[i, 3] = l
                    idx[i, 4] = m

    return idx


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def compute_scattering_cross_section(
    lmax: int,
    particle_number: int,
    idx: np.ndarray,
    sfc: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
):
    """Compute the scattering cross section.

    The original implementation performed a parallel loop over all (j1, j2)
    pairs and accumulated into a shared output array, which is not a safe
    parallel reduction.

    This version keeps the parallelism over `j1` (large axis, important for
    performance) by computing per-`j1` partial sums and reducing at the end.
    """

    jmax = particle_number * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    partial = np.zeros((jmax, channels), dtype=np.complex128)

    for j1 in prange(jmax):
        s1, n1, _, _, m1 = idx[j1, :]

        for j2 in range(jmax):
            s2, n2, _, _, m2 = idx[j2, :]

            delta_m = abs(m1 - m2)
            phase = e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2]

            for w in range(channels):
                val = 0.0 + 0.0j
                for p in range(delta_m, 2 * lmax + 1):
                    val += (
                        translation_table[n2, n1, p]
                        * plm[p * (p + 1) // 2 + delta_m, s1, s2]
                        * sph_h[p, s1, s2, w]
                    )

                val *= np.conj(sfc[s1, n1, w]) * phase * sfc[s2, n2, w]
                partial[j1, w] += val

    return np.sum(partial, axis=0)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def compute_radial_independent_scattered_field_legacy(
    lmax: int,
    particles_position: np.ndarray,
    idx: np.ndarray,
    sfc: np.ndarray,
    k_medium: np.ndarray,
    azimuthal_angles: np.ndarray,
    e_r: np.ndarray,
    e_phi: np.ndarray,
    e_theta: np.ndarray,
    pilm: np.ndarray,
    taulm: np.ndarray,
):
    """Calculates the scattered field for a given set of parameters and returns the result.

    Args:
        lmax (int): The maximum value of the angular momentum quantum number `l`. It determines the maximum order of the spherical harmonics used in the computation.
        particles_position (np.ndarray): An array representing the positions of particles. It has shape `(num_particles, 3)`, where `num_particles` is the number of particles and each row represents the x, y, and z coordinates of a particle.
        idx (np.ndarray): An array containing the indices of the particles. It has shape `(jmax, 5)` where `jmax` is the total number of particles. Each row of `idx` represents a particle and contains the following information:
        sfc (np.ndarray): A 3-dimensional array representing the scattering form factors. It has dimensions `(s, n, w)`, where:
        k_medium (np.ndarray): An array representing the wave number in the medium. It is used in the calculation of the scattered field.
        azimuthal_angles (np.ndarray): An array of azimuthal angles, representing the angles at which the scattered field is computed.
        e_r (np.ndarray): An array representing the radial component of the electric field. It has shape `(azimuthal_angles.size, 3)`, where `azimuthal_angles.size` is the number of azimuthal angles and 3 represents the three Cartesian components of the electric field.
        e_phi (np.ndarray): An array representing the electric field component in the azimuthal direction. It has a shape of `(azimuthal_angles.size, 3)`, where `azimuthal_angles.size` is the number of azimuthal angles and `3` represents the three components of the electric field.
        e_theta (np.ndarray): An array representing the electric field component in the theta direction. It has a shape of `(azimuthal_angles.size, 3)`, where `azimuthal_angles.size` is the number of azimuthal angles and `3` represents the three components of the electric field.
        pilm (np.ndarray): An array representing the matrix of spherical harmonics coefficients. It has a shape of `(lmax+1, lmax+1, azimuthal_angles.size)`. Each element `pilm[l, m, a]` represents the coefficient of the spherical harmonics for a given `l`, `m`, and azimuthal angle `a`.
        taulm (np.ndarray): An array representing the scattering coefficients for each combination of `l`, `m`, and azimuthal angle `a`. It has a shape of `(lmax+1, lmax+1, azimuthal_angles.size)`. The values in `taulm` represent the scattering coefficients.

    Returns:
        e_1_sca (np.ndarray): An array of complex numbers representing the scattered field.

    """
    e_1_sca = np.zeros(
        azimuthal_angles.size * 3 * k_medium.size, dtype=np.complex128
    ).reshape((azimuthal_angles.size, 3, k_medium.size))
    jmax = particles_position.shape[0] * 2 * lmax * (lmax + 2)

    for global_idx in prange(jmax * azimuthal_angles.size * k_medium.size):
        w_idx = global_idx % (jmax * k_medium.size)
        g_idx = global_idx // (jmax * k_medium.size)

        a = g_idx

        w = w_idx % k_medium.size
        j_idx = w_idx // k_medium.size
        s, n, tau, l, m = idx[j_idx, :]

        t = (
            np.power(1j, tau - l - 2)
            * sfc[s, n, w]
            / np.sqrt(2 * l * (l + 1))
            * np.exp(
                1j
                * (
                    m * azimuthal_angles[a]
                    - k_medium[w] * np.sum(particles_position[s, :] * e_r[a, :])
                )
            )
        )

        if tau == 1:
            e_1_sca[a, :, w] += t * (
                e_theta[a, :] * pilm[l, np.abs(m), a] * 1j * m
                - e_phi[a, :] * taulm[l, np.abs(m), a]
            )
        else:
            e_1_sca[a, :, w] += t * (
                e_phi[a, :] * pilm[l, np.abs(m), a] * 1j * m
                + e_theta[a, :] * taulm[l, np.abs(m), a]
            )

    return e_1_sca


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def compute_electric_field_angle_components(
    lmax: int,
    particles_position: np.ndarray,
    idx: np.ndarray,
    sfc: np.ndarray,
    k_medium: np.ndarray,
    azimuthal_angles: np.ndarray,
    e_r: np.ndarray,
    pilm: np.ndarray,
    taulm: np.ndarray,
):
    """Calculates the electric field components in the theta and phi directions for given input parameters.

    Args:
        lmax (int): The maximum value of the angular momentum quantum number `l`. It determines the maximum value of `l` for which the calculations will be performed.
        particles_position (np.ndarray): The positions of particles. It has shape `(num_particles, 3)`, where `num_particles` is the number of particles and each particle has 3 coordinates (x, y, z).
        idx (np.ndarray): A numpy array of shape `(jmax, 5)`, where `jmax` is the total number of particles multiplied by `2 * lmax * (lmax + 2)`. Each row of `idx` represents the indices `(s, n, tau, l, m)`.
        sfc (np.ndarray): A 3-dimensional numpy array representing the scattering form factors. It has dimensions `(s, n, w)`.
        k_medium (np.ndarray): The wave vector in the medium. It is a numpy array that contains the wave vector values for different frequencies or wavelengths.
        azimuthal_angles (np.ndarray): An array representing the azimuthal angles at which the electric field components are computed. It specifies the angles at which the electric field is measured in the azimuthal direction.
        e_r (np.ndarray): The unit vector pointing in the direction of the electric field. It is a numpy array of shape `(azimuthal_angles.size, 3)`, where each row corresponds to a different azimuthal angle and the three columns represent the x, y, and z components.
        pilm (np.ndarray): A 3-dimensional numpy array of shape `(lmax+1, lmax+1, azimuthal_angles.size)`. It represents the matrix elements of the electric field expansion coefficients for the theta component. The indices `(l, m, a)` correspond to the spherical harmonics.
        taulm (np.ndarray): A numpy array that represents the angular momentum coupling coefficients. It has a shape of `(lmax+1, lmax+1, azimuthal_angles.size)`. The first dimension represents the value of `l`, the second dimension represents the value of `m`, and the third dimension represents the azimuthal angle.

    Returns:
        e_field_theta (np.ndarray): The electric field component in the theta direction.
        e_field_phi (np.ndarray): The electric field component in the phi direction.
    """
    e_field_theta = np.zeros(
        azimuthal_angles.size * k_medium.size, dtype=np.complex128
    ).reshape((azimuthal_angles.size, k_medium.size))
    e_field_phi = np.zeros_like(e_field_theta)

    jmax = particles_position.shape[0] * 2 * lmax * (lmax + 2)

    for global_idx in prange(jmax * azimuthal_angles.size * k_medium.size):
        w_idx = global_idx % (jmax * k_medium.size)
        g_idx = global_idx // (jmax * k_medium.size)

        a = g_idx

        w = w_idx % k_medium.size
        j_idx = w_idx // k_medium.size
        s, n, tau, l, m = idx[j_idx, :]

        t = (
            np.power(1j, tau - l - 2)
            * sfc[s, n, w]
            / np.sqrt(2 * l * (l + 1))
            * np.exp(
                1j
                * (
                    m * azimuthal_angles[a]
                    - k_medium[w] * np.sum(particles_position[s, :] * e_r[a, :])
                )
            )
        )

        if tau == 1:
            e_field_theta[a, w] += t * pilm[l, np.abs(m), a] * 1j * m
            e_field_phi[a, w] -= t * taulm[l, np.abs(m), a]
        else:
            e_field_theta[a, w] += t * taulm[l, np.abs(m), a]
            e_field_phi[a, w] += t * pilm[l, np.abs(m), a] * 1j * m

    return e_field_theta, e_field_phi


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def compute_polarization_components(
    number_of_wavelengths: int,
    number_of_angles: int,
    e_field_theta: np.ndarray,
    e_field_phi: np.ndarray,
):
    """
    Compute the polarization components of electromagnetic fields.

    Args:
        number_of_wavelengths (int): The number of wavelengths.
        number_of_angles (int): The number of angles.
        e_field_theta (np.ndarray): The electric field component in the theta direction.
        e_field_phi (np.ndarray): The electric field component in the phi direction.

    Returns:
        degree_of_polarization_tuple (tuple): A tuple containing the following polarization components:
            - I (np.ndarray): The total intensity.
            - degree_of_polarization (np.ndarray): The degree of polarization.
            - degree_of_linear_polarization (np.ndarray): The degree of linear polarization.
            - degree_of_linear_polarization_q (np.ndarray): The degree of linear polarization in the Q direction.
            - degree_of_linear_polarization_u (np.ndarray): The degree of linear polarization in the U direction.
            - degree_of_circular_polarization (np.ndarray): The degree of circular polarization.
    """
    # Stokes components
    # S = np.zeros(4 * number_of_angles * number_of_wavelengths, dtype=np.complex128).reshape((4, number_of_angles, number_of_wavelengths))
    I = np.zeros(number_of_angles * number_of_wavelengths, dtype=np.float64).reshape(
        (number_of_angles, number_of_wavelengths)
    )
    Q = np.zeros_like(I)
    U = np.zeros_like(I)
    V = np.zeros_like(I)

    for global_idx in prange(number_of_angles * number_of_wavelengths):
        w_idx = global_idx % number_of_wavelengths
        a_idx = global_idx // number_of_wavelengths

        e_field_theta_abs = (
            e_field_theta[a_idx, w_idx].real ** 2
            + e_field_theta[a_idx, w_idx].imag ** 2
        )
        e_field_phi_abs = (
            e_field_phi[a_idx, w_idx].real ** 2 + e_field_phi[a_idx, w_idx].imag ** 2
        )
        e_field_angle_interaction = (
            e_field_theta[a_idx, w_idx] * e_field_phi[a_idx, w_idx].conjugate()
        )

        I[a_idx, w_idx] = e_field_theta_abs + e_field_phi_abs
        Q[a_idx, w_idx] = e_field_theta_abs - e_field_phi_abs
        U[a_idx, w_idx] = -2 * e_field_angle_interaction.real
        V[a_idx, w_idx] = 2 * e_field_angle_interaction.imag

    degree_of_polarization = np.sqrt(Q**2 + U**2 + V**2) / I
    degree_of_linear_polarization = np.sqrt(Q**2 + U**2) / I
    degree_of_linear_polarization_q = -Q / I
    degree_of_linear_polarization_u = U / I
    degree_of_circular_polarization = V / I

    return (
        I,
        degree_of_polarization,
        degree_of_linear_polarization,
        degree_of_linear_polarization_q,
        degree_of_linear_polarization_u,
        degree_of_circular_polarization,
    )


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def compute_radial_independent_scattered_field(
    number_of_wavelengths: int,
    number_of_angles: int,
    e_phi: np.ndarray,
    e_theta: np.ndarray,
    e_field_theta: np.ndarray,
    e_field_phi: np.ndarray,
):
    """
    Compute the radial independent scattered field.

    Args:
        number_of_wavelengths (int): The number of wavelengths.
        number_of_angles (int): The number of angles.
        e_phi (np.ndarray): The electric field in the phi direction.
        e_theta (np.ndarray): The electric field in the theta direction.
        e_field_theta (np.ndarray): The electric field theta component.
        e_field_phi (np.ndarray): The electric field phi component.

    Returns:
        e_1_sca (np.ndarray): The computed radial independent scattered field.
    """
    e_1_sca = np.zeros(
        number_of_angles * 3 * number_of_wavelengths, dtype=np.complex128
    ).reshape((number_of_angles, 3, number_of_wavelengths))

    for global_idx in prange(number_of_angles * number_of_wavelengths):
        w = global_idx % number_of_wavelengths
        a = global_idx // number_of_wavelengths

        e_1_sca[a, :, w] = (
            e_field_theta[a, w] * e_theta[a, :] + e_field_phi[a, w] * e_phi[a, :]
        )

    return e_1_sca


@jit(parallel=True, forceobj=True)
def compute_lookup_tables(
    lmax: int, size_parameter: np.ndarray, phi: np.ndarray, cosine_theta: np.ndarray
):
    """
    Compute lookup tables for spherical computations.

    Args:
        lmax (int): The maximum degree of the spherical harmonics.
        size_parameter (np.ndarray): Array of size parameters.
        phi (np.ndarray): Array of azimuthal angles.
        cosine_theta (np.ndarray): Array of cosine of polar angles.

    Returns:
        spherical_bessel (np.ndarray): Array of spherical Bessel functions.
        spherical_hankel (np.ndarray): Array of spherical Hankel functions.
        e_j_dm_phi (np.ndarray): Array of exponential terms.
        p_lm (np.ndarray): Array of associated Legendre polynomials.
    """
    spherical_hankel = np.zeros(
        (2 * lmax + 1) * np.prod(size_parameter.shape), dtype=complex
    ).reshape((2 * lmax + 1,) + size_parameter.shape)
    spherical_bessel = np.zeros_like(spherical_hankel)
    e_j_dm_phi = np.zeros(
        (4 * lmax + 1) * np.prod(size_parameter.shape[:2]), dtype=complex
    ).reshape((4 * lmax + 1,) + size_parameter.shape[:2])
    p_lm = np.zeros(
        (lmax + 1) * (2 * lmax + 1) * np.prod(size_parameter.shape[:2])
    ).reshape(((lmax + 1) * (2 * lmax + 1),) + size_parameter.shape[:2])

    for p in prange(2 * lmax + 1):
        spherical_hankel[p, :, :, :] = np.sqrt(
            np.divide(
                np.pi / 2,
                size_parameter,
                out=np.zeros_like(size_parameter),
                where=size_parameter != 0,
            )
        ) * hankel1(p + 1 / 2, size_parameter)
        spherical_bessel[p, :, :, :] = spherical_jn(p, size_parameter)
        e_j_dm_phi[p, :, :] = np.exp(1j * (p - 2 * lmax) * phi)
        e_j_dm_phi[p + 2 * lmax, :, :] = np.exp(1j * p * phi)
        for absdm in range(p + 1):
            cml = np.sqrt(
                (2 * p + 1) / 2 / np.prod(np.arange(p - absdm + 1, p + absdm + 1))
            )
            p_lm[p * (p + 1) // 2 + absdm, :, :] = (
                cml * np.power(-1.0, absdm) * lpmv(absdm, p, cosine_theta)
            )

    return spherical_bessel, spherical_hankel, e_j_dm_phi, p_lm


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def _compute_field_with_coefficients(
    lmax: int,
    idx: np.ndarray,
    size_parameter: np.ndarray,
    sph_h: np.ndarray,
    derivative: np.ndarray,
    e_j_dm_phi: np.ndarray,
    p_lm: np.ndarray,
    pi_lm: np.ndarray,
    tau_lm: np.ndarray,
    e_r: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    field_coefficients: np.ndarray,
):
    """Assemble the scattered field from precomputed angular/radial coefficients.

    Parameters
    ----------
    lmax
        Maximum multipole degree.
    idx
        Index lookup table encoding rows ``(s, n, tau, l, m)``.
    size_parameter, sph_h, derivative, e_j_dm_phi, p_lm, pi_lm, tau_lm
        Precomputed special-function lookup tables used by the field evaluator.
    e_r, e_theta, e_phi
        Unit-vector components at sampling points.
    field_coefficients
        Coefficients per particle/order/channel.

    Returns
    -------
    numpy.ndarray
        Complex field with shape ``(channels, n_points, 3)``.
    """

    jmax = sph_h.shape[1] * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]
    n_points = sph_h.shape[2]

    field = np.zeros((channels, n_points, 3), dtype=np.complex128)

    for out_idx in prange(channels * n_points):
        w = out_idx % channels
        sampling_idx = out_idx // channels

        fx = 0j
        fy = 0j
        fz = 0j

        for j_idx in range(jmax):
            s, n, tau, l, m = idx[j_idx, :]
            absm = np.abs(m)

            inv = (
                1.0
                / np.sqrt(2.0 * (l + 1) * l)
                * e_j_dm_phi[m + 2 * lmax, s, sampling_idx]
            )

            coeff = field_coefficients[s, n, w]

            h = sph_h[l, s, sampling_idx, w]
            pi_val = pi_lm[l, absm, s, sampling_idx]
            tau_val = tau_lm[l, absm, s, sampling_idx]

            c1 = 1j * m * pi_val

            if tau == 1:
                fx += (
                    coeff
                    * inv
                    * h
                    * (
                        c1 * e_theta[s, sampling_idx, 0]
                        - tau_val * e_phi[s, sampling_idx, 0]
                    )
                )
                fy += (
                    coeff
                    * inv
                    * h
                    * (
                        c1 * e_theta[s, sampling_idx, 1]
                        - tau_val * e_phi[s, sampling_idx, 1]
                    )
                )
                fz += (
                    coeff
                    * inv
                    * h
                    * (
                        c1 * e_theta[s, sampling_idx, 2]
                        - tau_val * e_phi[s, sampling_idx, 2]
                    )
                )
            else:
                sp = size_parameter[s, sampling_idx, w]
                pref = (l * (l + 1)) / sp * h * p_lm[l, absm, s, sampling_idx]

                d = derivative[l, s, sampling_idx, w] / sp

                fx += (
                    coeff
                    * inv
                    * (
                        pref * e_r[s, sampling_idx, 0]
                        + d
                        * (
                            tau_val * e_theta[s, sampling_idx, 0]
                            + c1 * e_phi[s, sampling_idx, 0]
                        )
                    )
                )
                fy += (
                    coeff
                    * inv
                    * (
                        pref * e_r[s, sampling_idx, 1]
                        + d
                        * (
                            tau_val * e_theta[s, sampling_idx, 1]
                            + c1 * e_phi[s, sampling_idx, 1]
                        )
                    )
                )
                fz += (
                    coeff
                    * inv
                    * (
                        pref * e_r[s, sampling_idx, 2]
                        + d
                        * (
                            tau_val * e_theta[s, sampling_idx, 2]
                            + c1 * e_phi[s, sampling_idx, 2]
                        )
                    )
                )

        field[w, sampling_idx, 0] = fx
        field[w, sampling_idx, 1] = fy
        field[w, sampling_idx, 2] = fz

    return field


def compute_field(
    lmax: int,
    idx: np.ndarray,
    size_parameter: np.ndarray,
    sph_h: np.ndarray,
    derivative: np.ndarray,
    e_j_dm_phi: np.ndarray,
    p_lm: np.ndarray,
    pi_lm: np.ndarray,
    tau_lm: np.ndarray,
    e_r: np.ndarray,
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    scattered_field_coefficients: np.ndarray | None = None,
    initial_field_coefficients: np.ndarray | None = None,
    scatter_to_internal: np.ndarray | None = None,
):
    """Compute the field for scattered or initial coefficients.

    This is a lightweight Python dispatcher around a Numba-jitted core.
    Numba doesn't reliably support `np.ndarray | None` arguments in nopython
    mode, so the jitted function always receives a concrete coefficient array.

    `scatter_to_internal` is currently unused but kept for API compatibility.
    """

    if scattered_field_coefficients is not None:
        coefficients = scattered_field_coefficients
    elif initial_field_coefficients is not None:
        coefficients = initial_field_coefficients
    else:
        channels = sph_h.shape[-1]
        n_points = sph_h.shape[2]
        return np.zeros((channels, n_points, 3), dtype=np.complex128)

    return _compute_field_with_coefficients(
        lmax,
        idx,
        size_parameter,
        sph_h,
        derivative,
        e_j_dm_phi,
        p_lm,
        pi_lm,
        tau_lm,
        e_r,
        e_theta,
        e_phi,
        coefficients,
    )
