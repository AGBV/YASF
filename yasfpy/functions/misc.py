import numpy as np

from scipy.special import spherical_jn
from scipy.special import hankel1, lpmv

from yasfpy.functions.legendre_normalized_trigon import legendre_normalized_trigon


def jmult_max(num_part, lmax):
    """
    Calculate the maximum value of jmult.

    Parameters:
    num_part (int): The number of particles.
    lmax (int): The maximum value of l.

    Returns:
        (int): The maximum value of jmult.
    """
    return 2 * num_part * lmax * (lmax + 2)


def multi2single_index(j_s, tau, l, m, lmax):
    """
    Converts the multi-index (j_s, tau, l, m) to a single index.

    Args:
        j_s (int): The value of j_s.
        tau (int): The value of tau.
        l (int): The value of l.
        m (int): The value of m.
        lmax (int): The maximum value of l.

    Returns:
        (int): The single index corresponding to the multi-index (j_s, tau, l, m).
    """
    return (
        j_s * 2 * lmax * (lmax + 2)
        + (tau - 1) * lmax * (lmax + 2)
        + (l - 1) * (l + 1)
        + m
        + l
    )


def single_index2multi(idx, lmax):
    """
    Convert a single index to multi-indices (j_s, tau, l, m) for spherical harmonics.

    Args:
        idx (int): The single index.
        lmax (int): The maximum angular momentum quantum number.

    Returns:
        j_s (int): The spin index.
        tau (int): The isospin index.
        l (float): The orbital angular momentum quantum number.
        m (int): The magnetic quantum number.
    """
    j_s = idx // (2 * lmax * (lmax + 2))
    idx_new = idx % (2 * lmax * (lmax + 2))
    tau = idx_new // (lmax * (lmax + 2)) + 1
    idx_new = idx_new % (lmax * (lmax + 2))
    l = np.floor(np.sqrt(idx_new + 1))
    m = idx_new - (l * l + l - 1)
    return j_s, tau, l, m


def transformation_coefficients(pilm, taulm, tau, l, m, pol, dagger: bool = False):
    """
    Calculate the transformation coefficients for spherical harmonics.

    Args:
        pilm (ndarray): Array of spherical harmonics.
        taulm (ndarray): Array of spherical harmonics.
        tau (int): Polarization state.
        l (int): Degree of the spherical harmonics.
        m (int): Order of the spherical harmonics.
        pol (int): Polarization state.
        dagger (bool, optional): Whether to apply the dagger operation. Defaults to False.

    Returns:
        (float): The transformation coefficient.

    """
    ifac = 1j
    if dagger:
        ifac *= -1

    # Polarized light
    if np.any(np.equal(pol, [1, 2])):
        if tau == pol:
            spher_fun = taulm[l, np.abs(m)]
        else:
            spher_fun = m * pilm[l, np.abs(m)]

        return (
            -1
            / np.power(ifac, l + 1)
            / np.sqrt(2 * l * (l + 1))
            * (ifac * (pol == 1) + (pol == 2))
            * spher_fun
        )

    # Unpolarized light
    return (
        -1
        / np.power(ifac, l + 1)
        / np.sqrt(2 * l * (l + 1))
        * (ifac + 1)
        * (taulm[l, np.abs(m)] + m * pilm[l, np.abs(m)])
        / 2
    )


def mutual_lookup(
    lmax: int,
    positions_1: np.ndarray,
    positions_2: np.ndarray,
    refractive_index: np.ndarray,
    derivatives: bool = False,
    parallel: bool = False,
):
    """
    Calculate mutual lookup tables for scattering calculations.

    Args:
        lmax (int): The maximum degree of the spherical harmonics expansion.
        positions_1 (np.ndarray): The positions of the first set of particles.
        positions_2 (np.ndarray): The positions of the second set of particles.
        refractive_index (np.ndarray): The refractive indices of the particles.
        derivatives (bool, optional): Whether to calculate the derivatives of the lookup tables. Defaults to False.
        parallel (bool, optional): Whether to use parallel computation. Defaults to False.

    Returns:
        spherical_bessel (np.ndarray): The spherical Bessel functions.
        spherical_hankel (np.ndarray): The spherical Hankel functions.
        e_j_dm_phi (np.ndarray): The exponential term in the scattering calculation.
        p_lm (np.ndarray): The normalized Legendre polynomials.
        e_r (np.ndarray): The unit vectors in the radial direction.
        e_theta (np.ndarray): The unit vectors in the polar direction.
        e_phi (np.ndarray): The unit vectors in the azimuthal direction.
        cosine_theta (np.ndarray): The cosine of the polar angle.
        sine_theta (np.ndarray): The sine of the polar angle.
        size_parameter (np.ndarray): The size parameter of the particles.
        spherical_hankel_derivative (np.ndarray): The derivative of the spherical Hankel functions.
    """
    differences = positions_1[:, np.newaxis, :] - positions_2[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    distances = distances[:, :, np.newaxis]
    e_r = np.divide(
        differences, distances, out=np.zeros_like(differences), where=distances != 0
    )
    cosine_theta = e_r[:, :, 2]
    # cosine_theta = np.divide(
    #   differences[:, :, 2],
    #   distances,
    #   out = np.zeros_like(distances),
    #   where = distances != 0
    # )
    # correct possible rounding errors
    cosine_theta[cosine_theta < -1] = -1
    cosine_theta[cosine_theta > 1] = 1
    sine_theta = np.sqrt(1 - cosine_theta**2)
    phi = np.arctan2(differences[:, :, 1], differences[:, :, 0])
    e_theta = np.stack(
        [cosine_theta * np.cos(phi), cosine_theta * np.sin(phi), -sine_theta], axis=-1
    )
    e_phi = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)

    size_parameter = distances * refractive_index[np.newaxis, np.newaxis, :]

    if parallel:
        from yasfpy.functions.cpu_numba import compute_lookup_tables

        spherical_bessel, spherical_hankel, e_j_dm_phi, p_lm = compute_lookup_tables(
            lmax, size_parameter, phi, cosine_theta
        )
    else:
        p_range = np.arange(2 * lmax + 1)
        p_range = p_range[:, np.newaxis, np.newaxis, np.newaxis]
        size_parameter_extended = size_parameter[np.newaxis, :, :, :]
        spherical_hankel = np.sqrt(
            np.divide(
                np.pi / 2,
                size_parameter_extended,
                out=np.zeros_like(size_parameter_extended),
                where=size_parameter_extended != 0,
            )
        ) * hankel1(p_range + 1 / 2, size_parameter_extended)
        spherical_bessel = spherical_jn(p_range, size_parameter_extended)

        if derivatives:
            spherical_hankel_lower = np.sqrt(
                np.divide(
                    np.pi / 2,
                    size_parameter_extended,
                    out=np.zeros_like(size_parameter_extended),
                    where=size_parameter_extended != 0,
                )
            ) * hankel1(-1 / 2, size_parameter_extended)
            spherical_hankel_lower = np.vstack(
                (spherical_hankel_lower, spherical_hankel[:-1, :, :, :])
            )
            spherical_hankel_derivative = (
                size_parameter_extended * spherical_hankel_lower
                - p_range * spherical_hankel
            )

            # p_range = np.arange(2 * lmax + 2) - 1
            # p_range = p_range[:, np.newaxis, np.newaxis, np.newaxis]
            # spherical_hankel = np.sqrt(np.divide(np.pi / 2, size_parameter_extended, out = np.zeros_like(size_parameter_extended), where = size_parameter_extended != 0)) * hankel1(p_range + 1/2, size_parameter_extended)
            # spherical_hankel_derivative = size_parameter_extended * spherical_hankel[:-1, :, :, :] - p_range[1:, :, :, :] * spherical_hankel[1:, :, :, :]

            p_lm = legendre_normalized_trigon(lmax, cosine_theta, sine_theta)
        else:
            spherical_hankel_derivative = None

            p_lm = np.zeros(
                (lmax + 1) * (2 * lmax + 1) * np.prod(size_parameter.shape[:2])
            ).reshape(((lmax + 1) * (2 * lmax + 1),) + size_parameter.shape[:2])
            for p in range(2 * lmax + 1):
                for absdm in range(p + 1):
                    cml = np.sqrt(
                        (2 * p + 1)
                        / 2
                        * np.prod(1 / np.arange(p - absdm + 1, p + absdm + 1))
                    )
                    # if np.isnan(cml):
                    #     print(p)
                    #     print(absdm)
                    p_lm[p * (p + 1) // 2 + absdm, :, :] = (
                        cml * np.power(-1.0, absdm) * lpmv(absdm, p, cosine_theta)
                    )

        phi = phi[np.newaxis, :, :]
        p_range = np.arange(-2 * lmax, 2 * lmax + 1)
        p_range = p_range[:, np.newaxis, np.newaxis]
        e_j_dm_phi = np.exp(1j * p_range * phi)

    return (
        spherical_bessel,
        spherical_hankel,
        e_j_dm_phi,
        p_lm,
        e_r,
        e_theta,
        e_phi,
        cosine_theta,
        sine_theta,
        size_parameter,
        spherical_hankel_derivative,
    )
