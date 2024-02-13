from numba import jit, prange, complex128, float64, int64

import numpy as np
from scipy.special import spherical_jn, hankel1, lpmv


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
    """The function `particle_interaction` calculates the interaction between particles based on their
    properties and returns the result.

    Parameters
    ----------
    lmax : int
        The parameter `lmax` represents the maximum value of the angular momentum quantum number `l`. It
    determines the size of the arrays `plm` and `sph_h`.
    particle_number : int
        The `particle_number` parameter represents the number of particles in the system.
    idx : np.ndarray
        The `idx` parameter is a numpy array of shape `(jmax, 5)`, where `jmax` is the total number of
    interactions between particles. Each row of `idx` represents an interaction and contains the
    following information:
    x : np.ndarray
        The parameter `x` is a numpy array representing the positions of the particles. It has shape
    `(particle_number,)` and contains the x-coordinates of the particles.
    translation_table : np.ndarray
        The `translation_table` parameter is a 3-dimensional numpy array that stores the translation
    coefficients used in the calculation. It has shape `(n2, n1, p)` where `n2` and `n1` are the indices
    of the translation coefficients, and `p` is the maximum
    plm : np.ndarray
        The parameter `plm` is a numpy array representing the associated Legendre polynomials. It has shape
    `(pmax * (pmax + 1) // 2, s1max, s2max)`, where `pmax` is the maximum degree of the Legendre
    polynomials
    sph_h : np.ndarray
        The `sph_h` parameter is a numpy array representing the spherical harmonics. It has shape `(jmax,
    jmax, channels)`, where `jmax` is the maximum number of particles, `channels` is the number of
    channels, and `jmax` is the maximum number
    e_j_dm_phi
        The parameter `e_j_dm_phi` is not defined in the code snippet you provided. Could you please
    provide the definition or explanation of what `e_j_dm_phi` represents?

    Returns
    -------
        the array `wx`, which represents the result of the particle interaction calculations.

    """
    jmax = particle_number * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    wx = np.zeros(x.size * channels, dtype=complex128).reshape(x.shape + (channels,))

    for w_idx in prange(jmax * jmax * channels):
        w = w_idx % channels
        j_idx = w_idx // channels
        j1 = j_idx // jmax
        j2 = j_idx % jmax
        s1, n1, tau1, l1, m1 = idx[j1, :]
        s2, n2, tau2, l2, m2 = idx[j2, :]

        if s1 == s2:
            continue

        delta_tau = np.absolute(tau1 - tau2)
        delta_l = np.absolute(l1 - l2)
        delta_m = np.absolute(m1 - m2)

        val = 0j
        for p in range(np.maximum(delta_m, delta_l + delta_tau), l1 + l2 + 1):
            val += (
                translation_table[n2, n1, p]
                * plm[p * (p + 1) // 2 + delta_m, s1, s2]
                * sph_h[p, s1, s2, w]
            )
        val *= e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

        wx[j1, w] += val

    return wx


@jit(nopython=True, parallel=True, fastmath=True)
def compute_idx_lookups(lmax: int, particle_number: int):
    """The function `compute_idx_lookups` generates an index lookup table for a given `lmax` and
    `particle_number` using parallel processing.

    Parameters
    ----------
    lmax : int
        The parameter `lmax` represents the maximum value of the angular momentum quantum number `l`. It
    determines the range of values for `l` in the nested loop.
    particle_number : int
        The parameter `particle_number` represents the number of particles in the system.

    Returns
    -------
        The function `compute_idx_lookups` returns a NumPy array `idx` which contains the computed index
    lookups.

    """
    nmax = 2 * lmax * (lmax + 2)
    idx = np.zeros(nmax * particle_number * 5, dtype=int64).reshape(
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
    """The function `compute_scattering_cross_section` calculates the scattering cross section for a given
    set of input parameters.

    Parameters
    ----------
    lmax : int
        The parameter `lmax` represents the maximum angular momentum quantum number. It determines the
    maximum value of `l` in the calculations.
    particle_number : int
        The `particle_number` parameter represents the number of particles in the system.
    idx : np.ndarray
        `idx` is a numpy array of shape `(jmax, 5)`, where `jmax` is the total number of particle pairs.
    Each row of `idx` represents a particle pair and contains the following information:
    sfc : np.ndarray
        The `sfc` parameter is a numpy array of shape `(s, n, channels)`, where:
    translation_table : np.ndarray
        The `translation_table` parameter is a 3-dimensional numpy array that stores the translation
    coefficients used in the computation of the scattering cross section. It has shape `(n2, n1, p)`
    where `n2` and `n1` are the number of radial functions for the second and
    plm : np.ndarray
        The parameter `plm` is a numpy array representing the associated Legendre polynomials. It has shape
    `(pmax * (pmax + 1) // 2, 2, 2)`, where `pmax` is the maximum value of `p` in the loop.
    sph_h : np.ndarray
        The `sph_h` parameter is a numpy array of shape `(pmax, s1max, s2max, channels)`. It represents the
    scattering matrix elements for each combination of `s1`, `s2`, and `p`, where `p` is the order of
    the Legend
    e_j_dm_phi : np.ndarray
        The parameter `e_j_dm_phi` is a numpy array representing the scattering phase function. It has
    shape `(2*lmax+1, channels, channels)` and contains complex values. The indices `(j, s1, s2)`
    represent the angular momentum index `j`, and the spin indices

    Returns
    -------
        The function `compute_scattering_cross_section` returns the complex scattering cross section
    `c_sca_complex`.

    """
    jmax = particle_number * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    c_sca_complex = np.zeros(channels, dtype=complex128)

    for j_idx in prange(jmax * jmax):
        j1 = j_idx // jmax
        j2 = j_idx % jmax
        s1, n1, _, _, m1 = idx[j1, :]
        s2, n2, _, _, m2 = idx[j2, :]

        delta_m = np.absolute(m1 - m2)

        p_dependent = np.zeros(channels, dtype=complex128)
        for p in range(delta_m, 2 * lmax + 1):
            p_dependent += (
                translation_table[n2, n1, p]
                * plm[p * (p + 1) // 2 + delta_m, s1, s2]
                * sph_h[p, s1, s2, :]
            )
        p_dependent *= (
            np.conj(sfc[s1, n1, :])
            * e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2]
            * sfc[s2, n2, :]
        )

        c_sca_complex += p_dependent

    return c_sca_complex


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
    '''The function `compute_radial_independent_scattered_field_legacy` calculates the scattered field for
    a given set of parameters and returns the result.
    
    Parameters
    ----------
    lmax : int
        The parameter `lmax` represents the maximum value of the angular momentum quantum number `l`. It
    determines the maximum order of the spherical harmonics used in the computation.
    particles_position : np.ndarray
        The `particles_position` parameter is a numpy array that represents the positions of particles. It
    has shape `(num_particles, 3)`, where `num_particles` is the number of particles and each row
    represents the x, y, and z coordinates of a particle.
    idx : np.ndarray
        The `idx` parameter is a numpy array that contains the indices of the particles. It has shape
    `(jmax, 5)` where `jmax` is the total number of particles. Each row of `idx` represents a particle
    and contains the following information:
    sfc : np.ndarray
        The parameter `sfc` is a 3-dimensional numpy array representing the scattering form factors. It has
    dimensions `(s, n, w)`, where:
    k_medium : np.ndarray
        The parameter `k_medium` is a numpy array representing the wave number in the medium. It is used in
    the calculation of the scattered field.
    azimuthal_angles : np.ndarray
        An array of azimuthal angles, representing the angles at which the scattered field is computed.
    e_r : np.ndarray
        e_r is a numpy array representing the radial component of the electric field. It has shape
    (azimuthal_angles.size, 3), where azimuthal_angles.size is the number of azimuthal angles and 3
    represents the three Cartesian components of the electric field.
    e_phi : np.ndarray
        `e_phi` is a numpy array representing the electric field component in the azimuthal direction. It
    has a shape of `(azimuthal_angles.size, 3)`, where `azimuthal_angles.size` is the number of
    azimuthal angles and `3` represents the three components of the
    e_theta : np.ndarray
        `e_theta` is a numpy array representing the electric field component in the theta direction. It has
    a shape of `(azimuthal_angles.size, 3)`, where `azimuthal_angles.size` is the number of azimuthal
    angles and `3` represents the three components of the electric
    pilm : np.ndarray
        The parameter `pilm` is a numpy array that represents the matrix of spherical harmonics
    coefficients. It has a shape of `(lmax+1, lmax+1, azimuthal_angles.size)`. Each element `pilm[l, m,
    a]` represents the coefficient of the spherical
    taulm : np.ndarray
        `taulm` is a numpy array that represents the scattering coefficients for each combination of `l`,
    `m`, and azimuthal angle `a`. It has a shape of `(lmax+1, lmax+1, azimuthal_angles.size)`. The
    values in `taulm
    
    Returns
    -------
        The function `compute_radial_independent_scattered_field_legacy` returns the variable `e_1_sca`,
    which is a numpy array of complex numbers.
    
    '''
    e_1_sca = np.zeros(
        azimuthal_angles.size * 3 * k_medium.size, dtype=complex128
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
    '''The function `compute_electric_field_angle_components` calculates the electric field components in
    the theta and phi directions for given input parameters.
    
    Parameters
    ----------
    lmax : int
        The parameter `lmax` represents the maximum value of the angular momentum quantum number `l`. It
    determines the maximum value of `l` for which the calculations will be performed.
    particles_position : np.ndarray
        The `particles_position` parameter is a numpy array that represents the positions of particles. It
    has shape `(num_particles, 3)`, where `num_particles` is the number of particles and each particle
    has 3 coordinates (x, y, z).
    idx : np.ndarray
        The `idx` parameter is a numpy array of shape `(jmax, 5)`, where `jmax` is the total number of
    particles multiplied by `2 * lmax * (lmax + 2)`. Each row of `idx` represents the indices `(s, n,
    sfc : np.ndarray
        The parameter `sfc` is a 3-dimensional numpy array representing the scattering form factors. It has
    dimensions `(s, n, w)`, where:
    k_medium : np.ndarray
        The parameter `k_medium` represents the wave vector in the medium. It is a numpy array that
    contains the wave vector values for different frequencies or wavelengths.
    azimuthal_angles : np.ndarray
        The parameter `azimuthal_angles` is an array that represents the azimuthal angles at which the
    electric field components are computed. It specifies the angles at which the electric field is
    measured in the azimuthal direction.
    e_r : np.ndarray
        The parameter `e_r` represents the unit vector pointing in the direction of the electric field. It
    is a numpy array of shape `(azimuthal_angles.size, 3)`, where each row corresponds to a different
    azimuthal angle and the three columns represent the x, y, and z components
    pilm : np.ndarray
        `pilm` is a 3-dimensional numpy array of shape `(lmax+1, lmax+1, azimuthal_angles.size)`. It
    represents the matrix elements of the electric field expansion coefficients for the theta component.
    The indices `(l, m, a)` correspond to the spherical harmon
    taulm : np.ndarray
        `taulm` is a numpy array that represents the angular momentum coupling coefficients. It has a shape
    of `(lmax+1, lmax+1, azimuthal_angles.size)`. The first dimension represents the value of `l`, the
    second dimension represents the value of `m`, and
    
    Returns
    -------
        The function `compute_electric_field_angle_components` returns two arrays: `e_field_theta` and
    `e_field_phi`.
    
    '''
    e_field_theta = np.zeros(
        azimuthal_angles.size * k_medium.size, dtype=complex128
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
        tuple: A tuple containing the following polarization components:
            - I (np.ndarray): The total intensity.
            - degree_of_polarization (np.ndarray): The degree of polarization.
            - degree_of_linear_polarization (np.ndarray): The degree of linear polarization.
            - degree_of_linear_polarization_q (np.ndarray): The degree of linear polarization in the Q direction.
            - degree_of_linear_polarization_u (np.ndarray): The degree of linear polarization in the U direction.
            - degree_of_circular_polarization (np.ndarray): The degree of circular polarization.
    """
    # Stokes components
    # S = np.zeros(4 * number_of_angles * number_of_wavelengths, dtype=complex128).reshape((4, number_of_angles, number_of_wavelengths))
    I = np.zeros(number_of_angles * number_of_wavelengths, dtype=float64).reshape(
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
        np.ndarray: The computed radial independent scattered field.
    """
    e_1_sca = np.zeros(
        number_of_angles * 3 * number_of_wavelengths, dtype=complex128
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
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the computed lookup tables:
            - spherical_bessel (np.ndarray): Array of spherical Bessel functions.
            - spherical_hankel (np.ndarray): Array of spherical Hankel functions.
            - e_j_dm_phi (np.ndarray): Array of exponential terms.
            - p_lm (np.ndarray): Array of associated Legendre polynomials.
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
    scattered_field_coefficients: np.ndarray = None,
    initial_field_coefficients: np.ndarray = None,
    scatter_to_internal: np.ndarray = None,
):
    """
    Compute the field using the given parameters and coefficients.

    Parameters:
    - lmax (int): The maximum degree of the spherical harmonics.
    - idx (np.ndarray): The index array containing the values of s, n, tau, l, and m.
    - size_parameter (np.ndarray): The size parameter array.
    - sph_h (np.ndarray): The spherical harmonics array.
    - derivative (np.ndarray): The derivative array.
    - e_j_dm_phi (np.ndarray): The e_j_dm_phi array.
    - p_lm (np.ndarray): The p_lm array.
    - pi_lm (np.ndarray): The pi_lm array.
    - tau_lm (np.ndarray): The tau_lm array.
    - e_r (np.ndarray): The e_r array.
    - e_theta (np.ndarray): The e_theta array.
    - e_phi (np.ndarray): The e_phi array.
    - scattered_field_coefficients (np.ndarray, optional): The scattered field coefficients array. Defaults to None.
    - initial_field_coefficients (np.ndarray, optional): The initial field coefficients array. Defaults to None.
    - scatter_to_internal (np.ndarray, optional): The scatter to internal array. Defaults to None.

    Returns:
    - field (np.ndarray): The computed field array.
    """
    jmax = sph_h.shape[1] * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    field = np.zeros(channels * sph_h.shape[2] * 3, dtype=complex128).reshape(
        (channels, sph_h.shape[2], 3)
    )

    if (scattered_field_coefficients is None) and (initial_field_coefficients is None):
        print(
            "At least one, scattered field or initial field coefficients, need to be given."
        )
        print("Returning a zero array")
        return field

    for w_idx in prange(2 * lmax * (lmax + 2) * np.prod(np.array(sph_h.shape[1:]))):
        w = w_idx % channels
        j_idx = w_idx // channels
        sampling_idx = j_idx // jmax
        j_idx = j_idx % jmax
        s, n, tau, l, m = idx[j_idx, :]

        invariant = (
            1 / np.sqrt(2 * (l + 1) * l) * e_j_dm_phi[m + 2 * lmax, s, sampling_idx]
        )
        # Calculate M
        if tau == 1:
            c_term_1 = (
                1j
                * m
                * pi_lm[l, np.abs(m), s, sampling_idx]
                * e_theta[s, sampling_idx, :]
            )
            c_term_2 = tau_lm[l, np.abs(m), s, sampling_idx] * e_phi[s, sampling_idx, :]
            c_term = sph_h[l, s, sampling_idx, w] * (c_term_1 - c_term_2)

            field[w, sampling_idx, :] += (
                scattered_field_coefficients[s, n, w] * invariant * c_term
            )
        # Calculate N
        else:
            p_term = (
                l
                * (l + 1)
                / size_parameter[s, sampling_idx, w]
                * sph_h[l, s, sampling_idx, w]
                * p_lm[l, np.abs(m), s, sampling_idx]
                * e_r[s, sampling_idx, :]
            )
            #   p_term = l * (l + 1) / size_parameter[s, sampling_idx, w]
            #   p_term *= sph_h[l, s, sampling_idx, w]
            #   p_term *= p_lm[l, np.abs(m), s, sampling_idx]
            #   p_term *= e_r[s, sampling_idx, :]

            b_term_1 = (
                derivative[l, s, sampling_idx, w] / size_parameter[s, sampling_idx, w]
            )
            b_term_2 = (
                tau_lm[l, np.abs(m), s, sampling_idx] * e_theta[s, sampling_idx, :]
            )
            b_term_3 = (
                1j
                * m
                * pi_lm[l, np.abs(m), s, sampling_idx]
                * e_phi[s, sampling_idx, :]
            )
            b_term = b_term_1 * (b_term_2 + b_term_3)

            field[w, sampling_idx, :] += (
                scattered_field_coefficients[s, n, w] * invariant * (p_term + b_term)
            )

    return field
