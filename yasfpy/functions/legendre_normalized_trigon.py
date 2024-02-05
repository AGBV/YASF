import numpy as np

# https://github.com/disordered-photonics/celes/blob/master/src/mathematics/legendre_normalized_trigon.m


def legendre_normalized_trigon(lmax, x: np.ndarray, y: np.ndarray = None):
    #   if not(isinstance(x, np.ndarray) or np.isscalar(x) or isinstance(x, list)):
    #     return legendre_normalized_trigon_legacy(x, y, lmax)

    #   if np.isscalar(x):
    #     x = np.array([x])
    #   elif isinstance(x, list):
    #     x = np.array(x)
    size = x.shape
    if y is None:
        x = np.ravel(x)
        ct = np.cos(x)
        st = np.sin(x)
    else:
        ct = x.ravel()
        st = y.ravel()

    plm = np.zeros((lmax + 1, lmax + 1, ct.size)) * np.nan

    plm[0, 0, :] = np.sqrt(1 / 2) * np.ones_like(ct)
    plm[1, 0, :] = np.sqrt(3 / 2) * ct

    for l in range(1, lmax):
        plm[l + 1, 0, :] = (
            1 / (l + 1) * np.sqrt((2 * l + 1) * (2 * l + 3)) * plm[l, 0, :] * ct
            - l / (l + 1) * np.sqrt((2 * l + 3) / (2 * l - 1)) * plm[l - 1, 0, :]
        )

    for m in range(1, lmax + 1):
        plm[m - 1, m, :] = np.zeros_like(ct)
        plm[m, m, :] = (
            np.sqrt((2 * m + 1) / 2 / np.math.factorial(2 * m))
            * np.prod(np.arange(1, 2 * m, 2))
            * np.power(st, m)
        )
        for l in range(m, lmax):
            plm[l + 1, m, :] = (
                np.sqrt((2 * l + 1) * (2 * l + 3) / (l + 1 - m) / (l + 1 + m))
                * ct
                * plm[l, m, :]
                - np.sqrt(
                    (2 * l + 3)
                    * (l - m)
                    * (l + m)
                    / (2 * l - 1)
                    / (l + 1 - m)
                    / (l + 1 + m)
                )
                * plm[l - 1, m, :]
            )

    plm = np.reshape(plm, np.concatenate(([lmax + 1, lmax + 1], size)))

    return plm


def legendre_normalized_trigon_legacy(x, y=None, lmax=4):
    import sympy as sym

    plm = sym.zeros(lmax + 1, lmax + 1)
    if y == None:
        ct = sym.cos(x)
        st = sym.sin(x)
    elif isinstance(x, sym.core.symbol.Symbol) and isinstance(
        y, sym.core.symbol.Symbol
    ):
        ct = x
        st = y
    else:
        ct = sym.Symbol("ct")
        st = sym.Symbol("st")

    plm[0, 0] = np.sqrt(1 / 2)
    plm[1, 0] = np.sqrt(3 / 2) * ct

    for l in range(1, lmax):
        plm[l + 1, 0] = (
            1 / (l + 1) * np.sqrt((2 * l + 1) * (2 * l + 3)) * plm[l, 0] * ct
            - l / (l + 1) * np.sqrt((2 * l + 3) / (2 * l - 1)) * plm[l - 1, 0]
        )

    for m in range(1, lmax + 1):
        plm[m - 1, m] = np.zeros_like(ct)
        plm[m, m] = (
            np.sqrt((2 * m + 1) / 2 / np.math.factorial(2 * m))
            * np.prod(np.arange(1, 2 * m, 2))
            * st**m
        )
        for l in range(m, lmax):
            plm[l + 1, m] = (
                np.sqrt((2 * l + 1) * (2 * l + 3) / (l + 1 - m) / (l + 1 + m))
                * ct
                * plm[l, m]
                - np.sqrt(
                    (2 * l + 3)
                    * (l - m)
                    * (l + m)
                    / (2 * l - 1)
                    / (l + 1 - m)
                    / (l + 1 + m)
                )
                * plm[l - 1, m]
            )

    return plm
