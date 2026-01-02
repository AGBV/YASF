"""Sampling-point field evaluation.

This module contains the implementation that used to live in
`yasfpy.simulation.Simulation.compute_fields`, extracted to keep
`yasfpy/simulation.py` smaller.

Given computed scattered-field expansion coefficients (VSWF coefficients about
particle centers), the main routine :func:`compute_fields` evaluates the
scattered field at user-provided sampling points. For plane-wave illumination it
also evaluates the incident field and reports the total field as incident plus
scattered.

Notes
-----
The evaluation relies on precomputed lookup tables for particle-to-point VSWF
translations (computed via :func:`yasfpy.functions.misc.mutual_lookup`) and then
dispatches to a CPU (Numba) or CUDA implementation.

References
----------
The use of VSWFs and translation operators for multiple scattering is described
in :cite:`Waterman-1971-ID50` and in the CELES implementation context
:cite:`Egel-2017-ID1`.
"""

from __future__ import annotations

from math import ceil
from time import time
from typing import TYPE_CHECKING

import numpy as np
from numba import cuda
from scipy.special import spherical_jn, spherical_yn

from yasfpy.functions.cpu_numba import compute_field
from yasfpy.functions.cuda_numba import compute_field_gpu
from yasfpy.functions.misc import mutual_lookup
from yasfpy.functions.spherical_functions_trigon import spherical_functions_trigon

if TYPE_CHECKING:
    from yasfpy.simulation import Simulation


def compute_fields(sim: "Simulation", sampling_points: np.ndarray) -> None:
    """Compute scattered (and optionally total) fields at sampling points.

    Parameters
    ----------
    sim:
        Simulation instance providing particle positions, the computed scattered
        field coefficients (``sim.scattered_field_coefficients``), and
        configuration flags such as ``sim.numerics.gpu``.
    sampling_points:
        Array of Cartesian sampling points with shape ``(N, 3)``.

    Returns
    -------
    None

    Notes
    -----
    This routine mutates ``sim`` by setting:

    - ``sim.sampling_points``
    - ``sim.scattered_field`` (evaluated field at sampling points)
    - ``sim.initial_field_electric`` and ``sim.initial_field_magnetic`` for the
      plane-wave illumination case
    - ``sim.total_field_electric`` equal to incident plus scattered when an
      incident field is evaluated

    The implementation computes particle-to-point lookup tables using
    :func:`yasfpy.functions.misc.mutual_lookup` (including Hankel-function
    derivatives) and then evaluates the field either via CPU/Numba
    (:func:`yasfpy.functions.cpu_numba.compute_field`) or CUDA
    (:func:`yasfpy.functions.cuda_numba.compute_field_gpu`).

    When ``sim.numerics.particle_distance_resolution > 0``, a radial
    interpolation scheme is used to reduce repeated special-function evaluation
    for large sampling grids.

    References
    ----------
    VSWF translation-based field evaluation is consistent with CELES-style
    formulations :cite:`Egel-2017-ID1`.
    """

    if sampling_points.shape[0] < 1:
        sim.log.error("Number of sampling points must be bigger than zero!")
        return
    if sampling_points.shape[1] != 3:
        sim.log.error("The points have to have three coordinates (x,y,z)!")
        return

    sim.initial_field_electric = None
    sim.initial_field_magnetic = None

    if (sim.parameters.initial_field.beam_width == 0) or np.isinf(
        sim.parameters.initial_field.beam_width
    ):
        alpha = float(sim.parameters.initial_field.azimuthal_angle)
        beta = float(sim.parameters.initial_field.polar_angle)
        cb = np.cos(beta)
        sb = np.sin(beta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        direction = np.array([sb * ca, sb * sa, cb], dtype=float)

        channels = sim.parameters.k_medium.size
        npts = sampling_points.shape[0]
        sim.initial_field_electric = np.zeros((channels, npts, 3), dtype=complex)
        sim.initial_field_magnetic = np.zeros_like(sim.initial_field_electric)

        if sim.numerics.gpu:
            from yasfpy.functions.cuda_numba import compute_plane_wave_field_gpu

            E_real = np.zeros((channels, npts, 3), dtype=float)
            E_imag = np.zeros_like(E_real)
            H_real = np.zeros_like(E_real)
            H_imag = np.zeros_like(E_real)

            points_device = cuda.to_device(np.ascontiguousarray(sampling_points))
            focal_device = cuda.to_device(
                np.ascontiguousarray(
                    sim.parameters.initial_field.focal_point.astype(float)
                )
            )
            direction_device = cuda.to_device(np.ascontiguousarray(direction))
            k_device = cuda.to_device(
                np.ascontiguousarray(sim.parameters.k_medium.astype(float))
            )
            n_m = np.asarray(sim.parameters.medium_refractive_index)
            n_real_device = cuda.to_device(np.ascontiguousarray(n_m.real.astype(float)))
            n_imag_device = cuda.to_device(np.ascontiguousarray(n_m.imag.astype(float)))

            E_real_device = cuda.to_device(E_real)
            E_imag_device = cuda.to_device(E_imag)
            H_real_device = cuda.to_device(H_real)
            H_imag_device = cuda.to_device(H_imag)

            threads = (16, 16)
            blocks = (
                ceil(npts / threads[0]),
                ceil(channels / threads[1]),
            )
            compute_plane_wave_field_gpu[blocks, threads](
                points_device,
                focal_device,
                direction_device,
                k_device,
                n_real_device,
                n_imag_device,
                int(sim.parameters.initial_field.pol),
                float(sim.parameters.initial_field.amplitude),
                float(sa),
                float(ca),
                float(sb),
                float(cb),
                E_real_device,
                E_imag_device,
                H_real_device,
                H_imag_device,
            )

            sim.initial_field_electric = (
                E_real_device.copy_to_host() + 1j * E_imag_device.copy_to_host()
            )
            sim.initial_field_magnetic = (
                H_real_device.copy_to_host() + 1j * H_imag_device.copy_to_host()
            )
        else:
            R = sampling_points - sim.parameters.initial_field.focal_point
            phase = R @ direction
            E = (
                sim.parameters.initial_field.amplitude
                * np.exp(
                    1j * phase[:, np.newaxis] * sim.parameters.k_medium[np.newaxis, :]
                )
            ).T  # (channels, N)

            n_medium = np.asarray(sim.parameters.medium_refractive_index)
            H = (-1j * n_medium[:, np.newaxis]) * E

            if sim.parameters.initial_field.pol == 1:  # TE
                sim.initial_field_electric[:, :, 0] = -sa * E
                sim.initial_field_electric[:, :, 1] = ca * E

                hx_fac = -ca * cb
                hy_fac = -sa * cb
                hz_fac = sb
                sim.initial_field_magnetic[:, :, 0] = 1j * hx_fac * H
                sim.initial_field_magnetic[:, :, 1] = 1j * hy_fac * H
                sim.initial_field_magnetic[:, :, 2] = 1j * hz_fac * H
            else:  # TM
                sim.initial_field_electric[:, :, 0] = ca * cb * E
                sim.initial_field_electric[:, :, 1] = sa * cb * E
                sim.initial_field_electric[:, :, 2] = -sb * E

                hx_fac = -sa
                hy_fac = ca
                sim.initial_field_magnetic[:, :, 0] = 1j * hx_fac * H
                sim.initial_field_magnetic[:, :, 1] = 1j * hy_fac * H

    sim.log.info("Computing mutual lookup")
    lookup_computation_time_start = time()

    (
        _,
        sph_h,
        e_j_dm_phi,
        p_lm,
        e_r,
        e_theta,
        e_phi,
        cosine_theta,
        sine_theta,
        size_parameter,
        sph_h_derivative,
    ) = mutual_lookup(
        sim.numerics.lmax,
        sampling_points,
        sim.parameters.particles.position,
        sim.parameters.k_medium,
        derivatives=True,
        parallel=False,
    )

    sph_h = np.swapaxes(sph_h, 1, 2)
    sph_h_derivative = np.swapaxes(sph_h_derivative, 1, 2)
    e_j_dm_phi = np.swapaxes(e_j_dm_phi, 1, 2)
    if p_lm.ndim == 4:
        p_lm = np.swapaxes(p_lm, 2, 3)
    e_r = np.swapaxes(e_r, 0, 1)
    e_theta = np.swapaxes(e_theta, 0, 1)
    e_phi = np.swapaxes(e_phi, 0, 1)
    cosine_theta = cosine_theta.T
    sine_theta = sine_theta.T
    size_parameter = np.swapaxes(size_parameter, 0, 1)

    resol = float(sim.numerics.particle_distance_resolution)
    if resol > 0:
        diffs = (
            sampling_points[np.newaxis, :, :]
            - sim.parameters.particles.position[:, np.newaxis, :]
        )
        r = np.sqrt(np.sum(diffs**2, axis=2))
        rmax = float(np.max(r))
        ri = np.arange(0.0, rmax + resol, resol)
        if ri.size < 2:
            ri = np.array([0.0, resol])

        kr = ri[:, np.newaxis] * sim.parameters.k_medium[np.newaxis, :]
        kr[0, :] = 1e-30

        channels = sim.parameters.k_medium.size
        sph_h_i = np.zeros((sim.numerics.lmax + 1, ri.size, channels), dtype=complex)
        sph_d_i = np.zeros_like(sph_h_i)
        for l in range(1, sim.numerics.lmax + 1):
            h_l = spherical_jn(l, kr) + 1j * spherical_yn(l, kr)
            h_lm1 = spherical_jn(l - 1, kr) + 1j * spherical_yn(l - 1, kr)
            sph_h_i[l, :, :] = h_l
            sph_d_i[l, :, :] = kr * h_lm1 - l * h_l

        idx0 = np.floor(r / resol).astype(int)
        idx0 = np.clip(idx0, 0, ri.size - 2)
        frac = (r / resol) - idx0

        sph_h_interp = np.zeros(
            (sim.numerics.lmax + 1, r.shape[0], r.shape[1], channels), dtype=complex
        )
        sph_d_interp = np.zeros_like(sph_h_interp)
        for l in range(1, sim.numerics.lmax + 1):
            for w in range(channels):
                h_w = sph_h_i[l, :, w]
                d_w = sph_d_i[l, :, w]
                h0 = h_w[idx0]
                h1 = h_w[idx0 + 1]
                d0 = d_w[idx0]
                d1 = d_w[idx0 + 1]
                sph_h_interp[l, :, :, w] = h0 + frac * (h1 - h0)
                sph_d_interp[l, :, :, w] = d0 + frac * (d1 - d0)

        sph_h = sph_h_interp
        sph_h_derivative = sph_d_interp
        size_parameter = (
            r[:, :, np.newaxis] * sim.parameters.k_medium[np.newaxis, np.newaxis, :]
        )

    lookup_computation_time_stop = time()
    sim.log.info(
        "Computing lookup tables took %f s",
        lookup_computation_time_stop - lookup_computation_time_start,
    )

    pi_lm, tau_lm = spherical_functions_trigon(
        sim.numerics.lmax, cosine_theta, sine_theta
    )

    sim.log.info("Computing field...")
    field_time_start = time()
    sim.sampling_points = sampling_points

    if sim.numerics.gpu:
        sim.log.info("\t...using GPU")

        field_real = np.zeros(
            (sim.parameters.k_medium.size, sampling_points.shape[0], 3), dtype=float
        )
        field_imag = np.zeros_like(field_real)

        idx_device = cuda.to_device(sim.idx_lookup)
        size_parameter_device = cuda.to_device(np.ascontiguousarray(size_parameter))
        sph_h_device = cuda.to_device(np.ascontiguousarray(sph_h))
        sph_h_derivative_device = cuda.to_device(np.ascontiguousarray(sph_h_derivative))
        e_j_dm_phi_device = cuda.to_device(np.ascontiguousarray(e_j_dm_phi))
        p_lm_device = cuda.to_device(np.ascontiguousarray(p_lm))
        pi_lm_device = cuda.to_device(np.ascontiguousarray(pi_lm))
        tau_lm_device = cuda.to_device(np.ascontiguousarray(tau_lm))
        e_r_device = cuda.to_device(np.ascontiguousarray(e_r))
        e_theta_device = cuda.to_device(np.ascontiguousarray(e_theta))
        e_phi_device = cuda.to_device(np.ascontiguousarray(e_phi))
        sfc_device = cuda.to_device(
            np.ascontiguousarray(sim.scattered_field_coefficients)
        )

        field_real_device = cuda.to_device(field_real)
        field_imag_device = cuda.to_device(field_imag)

        threads_per_block = (16, 16, 2)
        blocks_per_grid = (
            sampling_points.shape[0],
            sph_h.shape[1] * 2 * sim.numerics.lmax * (sim.numerics.lmax + 2),
            sim.parameters.k_medium.size,
        )
        blocks_per_grid = tuple(
            ceil(blocks_per_grid[i] / threads_per_block[i])
            for i in range(len(threads_per_block))
        )

        compute_field_gpu[blocks_per_grid, threads_per_block](
            sim.numerics.lmax,
            idx_device,
            size_parameter_device,
            sph_h_device,
            sph_h_derivative_device,
            e_j_dm_phi_device,
            p_lm_device,
            pi_lm_device,
            tau_lm_device,
            e_r_device,
            e_theta_device,
            e_phi_device,
            sfc_device,
            field_real_device,
            field_imag_device,
        )

        field_real = field_real_device.copy_to_host()
        field_imag = field_imag_device.copy_to_host()
        sim.scattered_field = field_real + 1j * field_imag
    else:
        sim.log.info("\t...using CPU")
        sim.scattered_field = compute_field(
            sim.numerics.lmax,
            sim.idx_lookup,
            size_parameter,
            sph_h,
            sph_h_derivative,
            e_j_dm_phi,
            p_lm,
            pi_lm,
            tau_lm,
            e_r,
            e_theta,
            e_phi,
            scattered_field_coefficients=sim.scattered_field_coefficients,
        )

    if sim.initial_field_electric is not None:
        sim.total_field_electric = sim.initial_field_electric + sim.scattered_field
    else:
        sim.total_field_electric = sim.scattered_field

    field_time_stop = time()
    sim.log.info(
        "\t Time taken for field calculation: %s", field_time_stop - field_time_start
    )
