import sys, os

sys.path.append(os.getcwd())
import yasfpy.log as log

from scipy.special import spherical_jn, spherical_yn


def t_entry(tau, l, k_medium, k_sphere, radius, field_type="scattered"):
    """
    Computes an entry in the T Matrix for a given l, k, and tau

    **Note**: scipy.special has also derivative function. Why is it not the same?
    Example:
      Now:    djx  = x *  spherical_jn(l-1, x)  - l * jx
      Possible: djx  = spherical_jn(l, x, derivative=True)
    """

    m = k_sphere / k_medium
    x = k_medium * radius
    mx = k_sphere * radius

    jx = spherical_jn(l, x)
    jmx = spherical_jn(l, mx)
    hx = spherical_jn(l, x) + 1j * spherical_yn(l, x)

    djx = x * spherical_jn(l - 1, x) - l * jx
    djmx = mx * spherical_jn(l - 1, mx) - l * jmx
    dhx = x * (spherical_jn(l - 1, x) + 1j * spherical_yn(l - 1, x)) - l * hx

    if (field_type, tau) == ("scattered", 1):
        return -(jmx * djx - jx * djmx) / (jmx * dhx - hx * djmx)  # -b
    elif (field_type, tau) == ("scattered", 2):
        return -(m**2 * jmx * djx - jx * djmx) / (
            m**2 * jmx * dhx - hx * djmx
        )  # -a
    elif (field_type, tau) == ("internal", 1):
        return (jx * dhx - hx * djx) / (jmx * dhx - hx * djmx)  # c
    elif (field_type, tau) == ("internal", 2):
        return (m * jx * dhx - m * hx * djx) / (m**2 * jmx * dhx - hx * djmx)  # d
    elif (field_type, tau) == ("ratio", 1):
        return (jx * dhx - hx * djx) / -(jmx * djx - jx * djmx)  # c / -b
    elif (field_type, tau) == ("ratio", 2):
        return (m * jx * dhx - m * hx * djx) / -(
            m**2 * jmx * djx - jx * djmx
        )  # d / -a
    else:
        logger = log.scattering_logger("t_entry")
        logger.warning("Not a valid field type provided. Returning None!")
        return None
