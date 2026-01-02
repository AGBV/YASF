"""Jones/Mueller matrix utilities.

This module contains helpers to convert between polarization representations,
including Jones and Mueller matrices.
"""

from __future__ import annotations

import numpy as np

try:
    import numba as nb
    from numba import prange
except ModuleNotFoundError:  # pragma: no cover
    nb = None
    prange = range  # type: ignore[assignment]


_SIGMA = np.asarray(
    [
        [[1.0, 0.0], [0.0, 1.0]],  # I
        [[1.0, 0.0], [0.0, -1.0]],  # Q
        [[0.0, 1.0], [1.0, 0.0]],  # U
        [[0.0, -1.0j], [1.0j, 0.0]],  # V
    ],
    dtype=np.complex128,
)


def jones_to_mueller(jones: np.ndarray) -> np.ndarray:
    """Convert a Jones matrix to a Mueller matrix.

    Parameters
    ----------
    jones:
        Complex Jones matrix with shape `(..., 2, 2)`.

    Returns
    -------
    np.ndarray
        Real Mueller matrix with shape `(..., 4, 4)`, using Stokes order (I,Q,U,V).

    Notes
    -----
    Uses the Pauli-matrix trace identity:

        M_ij = 0.5 * Re( Tr( σ_i J σ_j J† ) )

    where σ_i are (I, Q, U, V) Pauli matrices.
    """

    jones = np.asarray(jones)
    if jones.shape[-2:] != (2, 2):
        raise ValueError(f"jones must end with shape (2,2), got {jones.shape}")

    j_dagger = np.conjugate(np.swapaxes(jones, -1, -2))

    out_shape = jones.shape[:-2] + (4, 4)
    mueller = np.empty(out_shape, dtype=float)

    # Tr(σ_i J σ_j J†) = σ_i_ab J_bc σ_j_cd J†_da
    for i in range(4):
        for j in range(4):
            trace = np.einsum(
                "ab,...bc,cd,...da->...", _SIGMA[i], jones, _SIGMA[j], j_dagger
            )
            mueller[..., i, j] = 0.5 * np.real(trace)

    return mueller


if nb is not None:

    @nb.njit(parallel=True, nogil=True, fastmath=True)
    def _jones_to_mueller_kernel(jones_flat: np.ndarray) -> np.ndarray:
        n = jones_flat.shape[0]
        mueller_flat = np.empty((n, 4, 4), dtype=np.float64)

        sigma = _SIGMA

        for k in prange(n):
            a00 = jones_flat[k, 0, 0]
            a01 = jones_flat[k, 0, 1]
            a10 = jones_flat[k, 1, 0]
            a11 = jones_flat[k, 1, 1]

            # Conjugate-transpose (dagger)
            jdag00 = np.conjugate(a00)
            jdag01 = np.conjugate(a10)
            jdag10 = np.conjugate(a01)
            jdag11 = np.conjugate(a11)

            # Expand J and J† for indexed access.
            j00 = a00
            j01 = a01
            j10 = a10
            j11 = a11

            for i in range(4):
                for j in range(4):
                    trace = 0.0 + 0.0j

                    # Explicitly unrolled indices for 2x2 matrices.
                    # trace = σ_i_ab J_bc σ_j_cd J†_da
                    for a in range(2):
                        for b in range(2):
                            sigma_i = sigma[i, a, b]
                            if sigma_i == 0.0:
                                continue

                            if b == 0:
                                jb0 = j00
                                jb1 = j01
                            else:
                                jb0 = j10
                                jb1 = j11

                            # c = 0
                            for d in range(2):
                                sigma_j = sigma[j, 0, d]
                                if sigma_j != 0.0:
                                    if d == 0:
                                        jdag = jdag00 if a == 0 else jdag01
                                    else:
                                        jdag = jdag10 if a == 0 else jdag11
                                    trace += sigma_i * jb0 * sigma_j * jdag

                            # c = 1
                            for d in range(2):
                                sigma_j = sigma[j, 1, d]
                                if sigma_j != 0.0:
                                    if d == 0:
                                        jdag = jdag00 if a == 0 else jdag01
                                    else:
                                        jdag = jdag10 if a == 0 else jdag11
                                    trace += sigma_i * jb1 * sigma_j * jdag

                    mueller_flat[k, i, j] = 0.5 * np.real(trace)

        return mueller_flat


def jones_to_mueller_numba(jones: np.ndarray) -> np.ndarray:
    """Convert a Jones matrix to a Mueller matrix (Numba-accelerated).

    This is a drop-in replacement for `jones_to_mueller`, but much faster for
    large batches.

    Parameters
    ----------
    jones:
        Complex Jones matrix with shape `(..., 2, 2)`.

    Returns
    -------
    np.ndarray
        Real Mueller matrix with shape `(..., 4, 4)`, using Stokes order (I,Q,U,V).
    """

    jones = np.asarray(jones)
    if jones.shape[-2:] != (2, 2):
        raise ValueError(f"jones must end with shape (2,2), got {jones.shape}")

    if nb is None:
        return jones_to_mueller(jones)

    batch_shape = jones.shape[:-2]
    jones_flat = np.ascontiguousarray(jones.reshape((-1, 2, 2)), dtype=np.complex128)

    mueller_flat = _jones_to_mueller_kernel(jones_flat)
    return mueller_flat.reshape(batch_shape + (4, 4))
