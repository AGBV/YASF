import numpy as np
from numba import cuda
from cmath import exp, sqrt


# TODO: Implement data batching for GPUs with smaller memory
@cuda.jit(fastmath=True)
def particle_interaction_gpu(
    lmax: int,
    particle_number: int,
    idx: np.ndarray,
    x: np.ndarray,
    wx_real: np.ndarray,
    wx_imag: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi,
):
    """
    Perform particle interaction calculations on the GPU.

    Args:
        lmax (int): Maximum angular momentum quantum number.
        particle_number (int): Number of particles.
        idx (np.ndarray): Array containing particle indices.
        x (np.ndarray): Array of particle positions.
        wx_real (np.ndarray): Array to store the real part of the result.
        wx_imag (np.ndarray): Array to store the imaginary part of the result.
        translation_table (np.ndarray): Array containing translation table.
        plm (np.ndarray): Array containing associated Legendre polynomials.
        sph_h (np.ndarray): Array containing spherical harmonics.
        e_j_dm_phi (np.ndarray): Additional parameter for the calculation.

    Todo:
        Implement data batching for GPUs with smaller memory
    """
    j1, j2, w = cuda.grid(3)

    jmax = particle_number * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    if (j1 >= jmax) or (j2 >= jmax) or (w >= channels):
        return

    s1, n1, tau1, l1, m1 = idx[j1, :]
    s2, n2, tau2, l2, m2 = idx[j2, :]

    if s1 == s2:
        return

    delta_tau = abs(tau1 - tau2)
    delta_l = abs(l1 - l2)
    delta_m = abs(m1 - m2)

    p_dependent = complex(0)
    for p in range(max(delta_m, delta_l + delta_tau), l1 + l2 + 1):
        p_dependent += (
            translation_table[n2, n1, p]
            * plm[p * (p + 1) // 2 + delta_m, s1, s2]
            * sph_h[p, s1, s2, w]
        )
    p_dependent *= e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

    # atomic.add performs the += operation in sync
    cuda.atomic.add(wx_real, (j1, w), p_dependent.real)
    cuda.atomic.add(wx_imag, (j1, w), p_dependent.imag)


@cuda.jit(fastmath=True)
def compute_scattering_cross_section_gpu(
    lmax: int,
    particle_number: int,
    idx: np.ndarray,
    sfc: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
    c_sca_real: np.ndarray,
    c_sca_imag: np.ndarray,
):
    """
    Compute the scattering cross section on the GPU using CUDA.

    Args:
        lmax (int): The maximum degree of the spherical harmonics expansion.
        particle_number (int): The number of particles.
        idx (np.ndarray): The index array.
        sfc (np.ndarray): The scattering form factor array.
        translation_table (np.ndarray): The translation table array.
        plm (np.ndarray): The associated Legendre polynomials array.
        sph_h (np.ndarray): The spherical harmonics array.
        e_j_dm_phi (np.ndarray): The phase factor array.
        c_sca_real (np.ndarray): The real part of the scattering cross section array.
        c_sca_imag (np.ndarray): The imaginary part of the scattering cross section array.
    """
    jmax = particle_number * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    j1, j2, w = cuda.grid(3)

    if (j1 >= jmax) or (j2 >= jmax) or (w >= channels):
        return

    s1, n1, _, _, m1 = idx[j1, :]
    s2, n2, _, _, m2 = idx[j2, :]

    delta_m = abs(m1 - m2)
    f = sfc[:, :, w]

    p_dependent = complex(0)
    for p in range(delta_m, 2 * lmax + 1):
        p_dependent += (
            translation_table[n2, n1, p]
            * plm[p * (p + 1) // 2 + delta_m, s1, s2]
            * sph_h[p, s1, s2, w]
        )
    p_dependent *= (
        f[s1, n1].conjugate() * e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * f[s2, n2]
    )

    # atomic.add performs the += operation in sync
    cuda.atomic.add(c_sca_real, w, p_dependent.real)
    cuda.atomic.add(c_sca_imag, w, p_dependent.imag)


@cuda.jit(fastmath=True)
def compute_radial_independent_scattered_field_gpu(
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
    e_1_sca_real: np.ndarray,
    e_1_sca_imag: np.ndarray,
):
    """
    Compute the radial independent scattered field using GPU acceleration.

    Args:
        lmax (int): The maximum degree of the spherical harmonics expansion.
        particles_position (np.ndarray): Array of particle positions.
        idx (np.ndarray): Array of indices for particle properties.
        sfc (np.ndarray): Array of scattering form factors.
        k_medium (np.ndarray): Array of wave numbers in the medium.
        azimuthal_angles (np.ndarray): Array of azimuthal angles.
        e_r (np.ndarray): Array of radial electric field components.
        e_phi (np.ndarray): Array of azimuthal electric field components.
        e_theta (np.ndarray): Array of polar electric field components.
        pilm (np.ndarray): Array of associated Legendre polynomials.
        taulm (np.ndarray): Array of tau coefficients.
        e_1_sca_real (np.ndarray): Array of real parts of the scattered electric field.
        e_1_sca_imag (np.ndarray): Array of imaginary parts of the scattered electric field.
    """
    j_idx, a_idx, w_idx = cuda.grid(3)

    jmax = particles_position.shape[0] * 2 * lmax * (lmax + 2)

    if (j_idx >= jmax) or (a_idx >= azimuthal_angles.size) or (w_idx >= k_medium.size):
        return

    s, n, tau, l, m = idx[j_idx, :]

    # Temporary variable
    # If tau = 1 -> 1j**(tau-1) = 1, if tau = 2 -> 1j**(tau-1) = 1j
    # 1j**(-l-1) = (-1j)**(l+1) => both lead to the coefficient 1j**(tau-l-2)
    # k * <particle_position, e_r> is the phase shift due to the distance and relative position
    t = (
        1j ** (tau - l - 2)
        * sfc[s, n, w_idx]
        / sqrt(2 * l * (l + 1))
        * exp(
            1j
            * (
                m * azimuthal_angles[a_idx]
                - k_medium[w_idx]
                * (
                    particles_position[s, 0] * e_r[a_idx, 0]
                    + particles_position[s, 1] * e_r[a_idx, 1]
                    + particles_position[s, 2] * e_r[a_idx, 2]
                )
            )
        )
    )

    for c in range(3):
        if tau == 1:
            e_1_sca = t * (
                e_theta[a_idx, c] * pilm[l, abs(m), a_idx] * 1j * m
                - e_phi[a_idx, c] * taulm[l, abs(m), a_idx]
            )
        else:
            e_1_sca = t * (
                e_phi[a_idx, c] * pilm[l, abs(m), a_idx] * 1j * m
                + e_theta[a_idx, c] * taulm[l, abs(m), a_idx]
            )

        cuda.atomic.add(e_1_sca_real, (a_idx, c, w_idx), e_1_sca.real)
        cuda.atomic.add(e_1_sca_imag, (a_idx, c, w_idx), e_1_sca.imag)


@cuda.jit(fastmath=True)
def compute_electric_field_angle_components_gpu(
    lmax: int,
    particles_position: np.ndarray,
    idx: np.ndarray,
    sfc: np.ndarray,
    k_medium: np.ndarray,
    azimuthal_angles: np.ndarray,
    e_r: np.ndarray,
    pilm: np.ndarray,
    taulm: np.ndarray,
    e_field_theta_real: np.ndarray,
    e_field_theta_imag: np.ndarray,
    e_field_phi_real: np.ndarray,
    e_field_phi_imag: np.ndarray,
):
    """
    Compute the electric field angle components on the GPU.

    Args:
        lmax (int): The maximum angular momentum quantum number.
        particles_position (np.ndarray): Array of particle positions.
        idx (np.ndarray): Array of indices.
        sfc (np.ndarray): Array of scattering form factors.
        k_medium (np.ndarray): Array of medium wavevectors.
        azimuthal_angles (np.ndarray): Array of azimuthal angles.
        e_r (np.ndarray): Array of radial unit vectors.
        pilm (np.ndarray): Array of associated Legendre polynomials.
        taulm (np.ndarray): Array of tau coefficients.
        e_field_theta_real (np.ndarray): Array of real parts of electric field theta component.
        e_field_theta_imag (np.ndarray): Array of imaginary parts of electric field theta component.
        e_field_phi_real (np.ndarray): Array of real parts of electric field phi component.
        e_field_phi_imag (np.ndarray): Array of imaginary parts of electric field phi component.
    """
    j_idx, a_idx, w_idx = cuda.grid(3)
    jmax = particles_position.shape[0] * 2 * lmax * (lmax + 2)
    if (j_idx >= jmax) or (a_idx >= azimuthal_angles.size) or (w_idx >= k_medium.size):
        return

    s, n, tau, l, m = idx[j_idx, :]

    t = (
        1j ** (tau - l - 2)
        * sfc[s, n, w_idx]
        / sqrt(2 * l * (l + 1))
        * exp(
            1j
            * (
                m * azimuthal_angles[a_idx]
                - k_medium[w_idx]
                * (
                    particles_position[s, 0] * e_r[a_idx, 0]
                    + particles_position[s, 1] * e_r[a_idx, 1]
                    + particles_position[s, 2] * e_r[a_idx, 2]
                )
            )
        )
    )

    if tau == 1:
        e_field_theta = t * pilm[l, abs(m), a_idx] * 1j * m
        e_field_phi = -t * taulm[l, abs(m), a_idx]
    else:
        e_field_theta = t * taulm[l, abs(m), a_idx]
        e_field_phi = t * pilm[l, abs(m), a_idx] * 1j * m

    cuda.atomic.add(e_field_theta_real, (a_idx, w_idx), e_field_theta.real)
    cuda.atomic.add(e_field_theta_imag, (a_idx, w_idx), e_field_theta.imag)
    cuda.atomic.add(e_field_phi_real, (a_idx, w_idx), e_field_phi.real)
    cuda.atomic.add(e_field_phi_imag, (a_idx, w_idx), e_field_phi.imag)


@cuda.jit(fastmath=True)
def compute_polarization_components_gpu(
    number_of_wavelengths: int,
    number_of_angles: int,
    e_field_theta_real: np.ndarray,
    e_field_theta_imag: np.ndarray,
    e_field_phi_real: np.ndarray,
    e_field_phi_imag: np.ndarray,
    intensity: np.ndarray,
    degree_of_polarization: np.ndarray,
    degree_of_linear_polarization: np.ndarray,
    degree_of_linear_polarization_q: np.ndarray,
    degree_of_linear_polarization_u: np.ndarray,
    degree_of_circular_polarization: np.ndarray,
):
    """
    Compute the polarization components using GPU acceleration.

    Args:
        number_of_wavelengths (int): Number of wavelengths.
        number_of_angles (int): Number of angles.
        e_field_theta_real (np.ndarray): Real part of the electric field in the theta direction.
        e_field_theta_imag (np.ndarray): Imaginary part of the electric field in the theta direction.
        e_field_phi_real (np.ndarray): Real part of the electric field in the phi direction.
        e_field_phi_imag (np.ndarray): Imaginary part of the electric field in the phi direction.
        intensity (np.ndarray): Array to store the intensity component.
        degree_of_polarization (np.ndarray): Array to store the degree of polarization component.
        degree_of_linear_polarization (np.ndarray): Array to store the degree of linear polarization component.
        degree_of_linear_polarization_q (np.ndarray): Array to store the degree of linear polarization (Q) component.
        degree_of_linear_polarization_u (np.ndarray): Array to store the degree of linear polarization (U) component.
        degree_of_circular_polarization (np.ndarray): Array to store the degree of circular polarization component.
    """
    a_idx, w_idx = cuda.grid(2)
    if (w_idx >= number_of_wavelengths) or (a_idx >= number_of_angles):
        return

    # Jones vector components (1,2,4)
    e_field_theta_abs = (
        e_field_theta_real[a_idx, w_idx] ** 2 + e_field_theta_imag[a_idx, w_idx] ** 2
    )
    e_field_phi_abs = (
        e_field_phi_real[a_idx, w_idx] ** 2 + e_field_phi_imag[a_idx, w_idx] ** 2
    )
    e_field_angle_interaction_real = (
        e_field_theta_real[a_idx, w_idx] * e_field_phi_real[a_idx, w_idx]
        + e_field_theta_imag[a_idx, w_idx] * e_field_phi_imag[a_idx, w_idx]
    )
    e_field_angle_interaction_imag = (
        e_field_theta_imag[a_idx, w_idx] * e_field_phi_real[a_idx, w_idx]
        - e_field_theta_real[a_idx, w_idx] * e_field_phi_imag[a_idx, w_idx]
    )

    # Stokes components S = (I, Q, U, V)
    I = e_field_theta_abs + e_field_phi_abs
    Q = e_field_theta_abs - e_field_phi_abs
    U = -2 * e_field_angle_interaction_real
    V = 2 * e_field_angle_interaction_imag

    intensity[a_idx, w_idx] = I
    degree_of_polarization[a_idx, w_idx] = sqrt(Q**2 + U**2 + V**2).real / I
    degree_of_linear_polarization[a_idx, w_idx] = sqrt(Q**2 + U**2).real / I
    degree_of_linear_polarization_q[a_idx, w_idx] = -Q.real / I
    degree_of_linear_polarization_u[a_idx, w_idx] = U.real / I
    degree_of_circular_polarization[a_idx, w_idx] = V / I


@cuda.jit(fastmath=True)
def compute_field_gpu(
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
    scattered_field_coefficients: np.ndarray,
    field_real: np.ndarray,
    field_imag: np.ndarray,
):  # , initial_field_coefficients: np.ndarray, scatter_to_internal: np.ndarray):
    """
    Compute the field on the GPU using CUDA.

    Args:
        lmax (int): Maximum degree of the spherical harmonics.
        idx (np.ndarray): Array of indices.
        size_parameter (np.ndarray): Array of size parameters.
        sph_h (np.ndarray): Array of spherical harmonics.
        derivative (np.ndarray): Array of derivatives.
        e_j_dm_phi (np.ndarray): Array of phi-dependent terms.
        p_lm (np.ndarray): Array of Legendre polynomials.
        pi_lm (np.ndarray): Array of pi-dependent terms.
        tau_lm (np.ndarray): Array of tau-dependent terms.
        e_r (np.ndarray): Array of r-dependent terms.
        e_theta (np.ndarray): Array of theta-dependent terms.
        e_phi (np.ndarray): Array of phi-dependent terms.
        scattered_field_coefficients (np.ndarray): Array of scattered field coefficients.
        field_real (np.ndarray): Array to store the real part of the field.
        field_imag (np.ndarray): Array to store the imaginary part of the field.
    """
    jmax = sph_h.shape[1] * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    sampling_idx, j_idx, w = cuda.grid(3)

    if (sampling_idx >= sph_h.shape[2]) or (j_idx >= jmax) or (w >= channels):
        return

    particle_idx, n, tau, l, m = idx[j_idx, :]

    invariant = (
        1 / sqrt(2 * (l + 1) * l) * e_j_dm_phi[m + 2 * lmax, particle_idx, sampling_idx]
    )

    for c in range(3):
        term = scattered_field_coefficients[particle_idx, n, w] * invariant

        # Calculate M
        if tau == 1:
            c_term_1 = (
                pi_lm[l, abs(m), particle_idx, sampling_idx]
                * e_theta[particle_idx, sampling_idx, c]
                * 1j
                * m
            )
            c_term_2 = (
                tau_lm[l, abs(m), particle_idx, sampling_idx]
                * e_phi[particle_idx, sampling_idx, c]
            )
            c_term = sph_h[l, particle_idx, sampling_idx, w] * (c_term_1 - c_term_2)

            term *= c_term

        # Calculate N
        else:
            p_term = (
                l
                * (l + 1)
                / size_parameter[particle_idx, sampling_idx, w]
                * sph_h[l, particle_idx, sampling_idx, w]
            )
            p_term *= (
                p_lm[l, abs(m), particle_idx, sampling_idx]
                * e_r[particle_idx, sampling_idx, c]
            )

            b_term_1 = (
                derivative[l, particle_idx, sampling_idx, w]
                / size_parameter[particle_idx, sampling_idx, w]
            )
            b_term_2 = (
                tau_lm[l, abs(m), particle_idx, sampling_idx]
                * e_theta[particle_idx, sampling_idx, c]
            )
            b_term_3 = (
                pi_lm[l, abs(m), particle_idx, sampling_idx]
                * e_phi[particle_idx, sampling_idx, c]
                * 1j
                * m
            )
            b_term = b_term_1 * (b_term_2 + b_term_3)

            term *= p_term + b_term

        cuda.atomic.add(field_real, (w, sampling_idx, c), term.real)
        cuda.atomic.add(field_imag, (w, sampling_idx, c), term.imag)
