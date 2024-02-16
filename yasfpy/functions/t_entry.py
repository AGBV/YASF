import sys, os

sys.path.append(os.getcwd())
import yasfpy.log as log

from scipy.special import spherical_jn, spherical_yn


def t_entry(tau, l, k_medium, k_sphere, radius, field_type="scattered"):
    """
    Computes an entry in the T Matrix for a given l, k, and tau

    Args:
        tau (float): The value of tau.
        l (int): The value of l.
        k_medium (float): The value of k_medium.
        k_sphere (float): The value of k_sphere.
        radius (float): The value of radius.
        field_type (str, optional): The type of field. Defaults to "scattered".

    Returns:
        (float): The computed entry in the T Matrix.

    Note:
        [scipy.special](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.spherical_jn.html#scipy.special.spherical_jn){:target="_blank"}
        has also derivative function. Why is it not the same?

        Now:      `djx  = x *  spherical_jn(l-1, x)  - l * jx`<br>
        Possible: `djx  = spherical_jn(l, x, derivative=True)`

    Raises:
        ValueError: If an invalid field type is provided.

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
        logger.error("Not a valid field type provided. Returning None!")
        raise ValueError("Not a valid field type provided. Returning None!")
