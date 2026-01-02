"""Translation-entry helper routines.

Implements utilities for constructing translation operator entries used by the
multiple-scattering coupling.
"""

import sys, os

sys.path.append(os.getcwd())
import yasfpy.log as log

from scipy.special import spherical_jn, spherical_yn


def t_entry(tau, l, k_medium, k_sphere, radius, field_type="scattered"):
    """
    Computes an entry in the T Matrix for a given k, l, and tau

    Args:
        tau (float): The value of tau.
        l (int): The value of l.
        k_medium (float): The value of k_medium.
        k_sphere (float): The value of k_sphere.
        radius (float): The value of radius.
        field_type (str, optional): The type of field. Defaults to "scattered".

    Returns:
        (float): The computed entry in the T Matrix.

    """
    m = k_sphere / k_medium
    x = k_medium * radius
    mx = k_sphere * radius

    # Use SciPy derivatives to avoid extra Bessel evaluations.
    jx = spherical_jn(l, x)
    jx_prime = spherical_jn(l, x, derivative=True)
    yx = spherical_yn(l, x)
    yx_prime = spherical_yn(l, x, derivative=True)

    jmx = spherical_jn(l, mx)
    jmx_prime = spherical_jn(l, mx, derivative=True)

    hx = jx + 1j * yx
    hx_prime = jx_prime + 1j * yx_prime

    # Riccati-Bessel derivatives: d/dz [z * f_l(z)]
    djx = jx + x * jx_prime
    djmx = jmx + mx * jmx_prime
    dhx = hx + x * hx_prime

    # if (field_type, tau) == ("scattered", 1):
    #     return -(jmx * djx - jx * djmx) / (jmx * dhx - hx * djmx)  # -b
    # if (field_type, tau) == ("scattered", 2):
    #     return -(m**2 * jmx * djx - jx * djmx) / (m**2 * jmx * dhx - hx * djmx)  # -a
    # if (field_type, tau) == ("internal", 1):
    #     return (jx * dhx - hx * djx) / (jmx * dhx - hx * djmx)  # c
    # if (field_type, tau) == ("internal", 2):
    #     return (m * jx * dhx - m * hx * djx) / (m**2 * jmx * dhx - hx * djmx)  # d
    # if (field_type, tau) == ("ratio", 1):
    #     return (jx * dhx - hx * djx) / -(jmx * djx - jx * djmx)  # c / -b
    # if (field_type, tau) == ("ratio", 2):
    #     return (m * jx * dhx - m * hx * djx) / -(m**2 * jmx * djx - jx * djmx)  # d / -a
    match (field_type, tau):
        case ("scattered", 1):
            return -(jmx * djx - jx * djmx) / (jmx * dhx - hx * djmx)  # -b
        case ("scattered", 2):
            return -(m**2 * jmx * djx - jx * djmx) / (
                m**2 * jmx * dhx - hx * djmx
            )  # -a
        case ("internal", 1):
            return (jx * dhx - hx * djx) / (jmx * dhx - hx * djmx)  # c
        case ("internal", 2):
            return (m * jx * dhx - m * hx * djx) / (m**2 * jmx * dhx - hx * djmx)  # d
        case ("ratio", 1):
            return (jx * dhx - hx * djx) / -(jmx * djx - jx * djmx)  # c / -b
        case ("ratio", 2):
            return (m * jx * dhx - m * hx * djx) / -(
                m**2 * jmx * djx - jx * djmx
            )  # d / -a
        case _:
            raise ValueError("Not a valid field type provided. Returning None!")
