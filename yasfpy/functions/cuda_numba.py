# pyright: ignore

"""CUDA kernels accelerated with Numba.

This module contains GPU implementations of inner loops used for coupling and
field/polarization calculations.
"""



import numpy as np
from numba import cuda
from cmath import exp, sqrt
from math import cos, sin


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
    """Dense coupling kernel using atomics.

    This kernel launches one thread per (j1, j2, w) contribution and accumulates
    into `wx_*` using atomic adds. It is simple but can be slow due to the huge
    number of threads and atomic contention.

    For the solver's single-wavelength matvec path, prefer
    `particle_interaction_gpu_single_wavelength()`.
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

    p_real = 0.0
    p_imag = 0.0
    for p in range(max(delta_m, delta_l + delta_tau), l1 + l2 + 1):
        term = (
            translation_table[n2, n1, p]
            * plm[p * (p + 1) // 2 + delta_m, s1, s2]
            * sph_h[p, s1, s2, w]
        )
        p_real += term.real
        p_imag += term.imag

    phase_x = e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

    out_real = p_real * phase_x.real - p_imag * phase_x.imag
    out_imag = p_real * phase_x.imag + p_imag * phase_x.real

    # `wx_real` and `wx_imag` are expected to be 1D flattened buffers of size
    # (jmax * channels) to avoid reshaping inside the CUDA kernel.
    flat_idx = j1 * channels + w
    cuda.atomic.add(wx_real, flat_idx, out_real)
    cuda.atomic.add(wx_imag, flat_idx, out_imag)


@cuda.jit(fastmath=True)
def particle_interaction_gpu_single_wavelength(
    lmax: int,
    particle_number: int,
    w_idx: int,
    idx: np.ndarray,
    x: np.ndarray,
    wx: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
):
    """Dense coupling matvec for a single wavelength channel.

    This is the baseline GPU kernel: one thread computes one output row.

    For small/moderate problems this can underutilize the GPU, so we also provide
    a block-per-row reduction kernel: `particle_interaction_gpu_single_wavelength_block_row()`.
    """

    j1 = cuda.grid(1)

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax

    if j1 >= jmax:
        return

    s1 = idx[j1, 0]
    n1 = idx[j1, 1]
    tau1 = idx[j1, 2]
    l1 = idx[j1, 3]
    m1 = idx[j1, 4]

    acc_real = 0.0
    acc_imag = 0.0

    for s2 in range(particle_number):
        if s2 == s1:
            continue

        base = s2 * nmax
        end = base + nmax
        for j2 in range(base, end):
            n2 = idx[j2, 1]
            tau2 = idx[j2, 2]
            l2 = idx[j2, 3]
            m2 = idx[j2, 4]

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
            phase_x_real = phase_x.real
            phase_x_imag = phase_x.imag

            for p in range(p_start, l1 + l2 + 1):
                trans = translation_table[n2, n1, p]
                leg = plm[p * (p + 1) // 2 + delta_m, s1, s2]
                hank = sph_h[p, s1, s2, w_idx]

                # coeff = trans * leg * hank (manual complex multiply)
                tmp_real = trans.real * leg.real - trans.imag * leg.imag
                tmp_imag = trans.real * leg.imag + trans.imag * leg.real
                coeff_real = tmp_real * hank.real - tmp_imag * hank.imag
                coeff_imag = tmp_real * hank.imag + tmp_imag * hank.real

                # term = coeff * phase_x
                term_real = coeff_real * phase_x_real - coeff_imag * phase_x_imag
                term_imag = coeff_real * phase_x_imag + coeff_imag * phase_x_real

                acc_real += term_real
                acc_imag += term_imag

    wx[j1] = acc_real + 1j * acc_imag


@cuda.jit(fastmath=True)
def particle_interaction_gpu_single_wavelength_block_row_256(
    lmax: int,
    particle_number: int,
    w_idx: int,
    idx: np.ndarray,
    x: np.ndarray,
    wx: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
):
    """Dense coupling matvec for a single wavelength using block-per-row reduction.

    Mapping:
    - Each CUDA block computes one output row j1.
    - Threads in the block iterate over j2 in a strided loop and accumulate partial sums.
    - Partial sums are reduced in shared memory and written once.

    This generally improves GPU occupancy for small/moderate jmax by increasing
    parallelism per row.

    Notes
    -----
    - Requires blocks of exactly 256 threads.
    """

    t = cuda.threadIdx.x
    j1 = cuda.blockIdx.x

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax
    if j1 >= jmax:
        return

    s1 = idx[j1, 0]
    n1 = idx[j1, 1]
    tau1 = idx[j1, 2]
    l1 = idx[j1, 3]
    m1 = idx[j1, 4]

    acc_real = 0.0
    acc_imag = 0.0

    # Flatten j2 across all particles except self.
    # Iterate in stripes over the full j2 range.
    for j2 in range(t, jmax, 256):
        s2 = idx[j2, 0]
        if s2 == s1:
            continue

        n2 = idx[j2, 1]
        tau2 = idx[j2, 2]
        l2 = idx[j2, 3]
        m2 = idx[j2, 4]

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
        phase_x_real = phase_x.real
        phase_x_imag = phase_x.imag

        for p in range(p_start, l1 + l2 + 1):
            trans = translation_table[n2, n1, p]
            leg = plm[p * (p + 1) // 2 + delta_m, s1, s2]
            hank = sph_h[p, s1, s2, w_idx]

            tmp_real = trans.real * leg.real - trans.imag * leg.imag
            tmp_imag = trans.real * leg.imag + trans.imag * leg.real
            coeff_real = tmp_real * hank.real - tmp_imag * hank.imag
            coeff_imag = tmp_real * hank.imag + tmp_imag * hank.real

            term_real = coeff_real * phase_x_real - coeff_imag * phase_x_imag
            term_imag = coeff_real * phase_x_imag + coeff_imag * phase_x_real

            acc_real += term_real
            acc_imag += term_imag

    smem_real = cuda.shared.array(256, dtype=np.float64)
    smem_imag = cuda.shared.array(256, dtype=np.float64)
    smem_real[t] = acc_real
    smem_imag[t] = acc_imag
    cuda.syncthreads()

    offset = 128
    while offset > 0:
        if t < offset:
            smem_real[t] += smem_real[t + offset]
            smem_imag[t] += smem_imag[t + offset]
        cuda.syncthreads()
        offset //= 2

    if t == 0:
        wx[j1] = smem_real[0] + 1j * smem_imag[0]


@cuda.jit(fastmath=True)
def particle_interaction_gpu_single_wavelength_lut(
    lmax: int,
    particle_number: int,
    w_idx: int,
    tau_lut: np.ndarray,
    l_lut: np.ndarray,
    m_lut: np.ndarray,
    x: np.ndarray,
    wx: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
):
    """Dense coupling matvec for a single wavelength using compact nmax lookup tables.

    Compared to `particle_interaction_gpu_single_wavelength`, this avoids loading
    a large `idx` matrix from global memory. Instead we compute (s, n) from the
    flattened index j and use small lookup tables of size nmax for (tau, l, m).

    Mapping: one thread computes one output row.
    """

    j1 = cuda.grid(1)

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax

    if j1 >= jmax:
        return

    s1 = j1 // nmax
    n1 = j1 - s1 * nmax
    tau1 = tau_lut[n1]
    l1 = l_lut[n1]
    m1 = m_lut[n1]

    acc_real = 0.0
    acc_imag = 0.0

    for s2 in range(particle_number):
        if s2 == s1:
            continue

        base = s2 * nmax
        end = base + nmax
        for j2 in range(base, end):
            n2 = j2 - base
            tau2 = tau_lut[n2]
            l2 = l_lut[n2]
            m2 = m_lut[n2]

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
            phase_x_real = phase_x.real
            phase_x_imag = phase_x.imag

            for p in range(p_start, l1 + l2 + 1):
                trans = translation_table[n2, n1, p]
                leg = plm[p * (p + 1) // 2 + delta_m, s1, s2]
                hank = sph_h[p, s1, s2, w_idx]

                tmp_real = trans.real * leg.real - trans.imag * leg.imag
                tmp_imag = trans.real * leg.imag + trans.imag * leg.real
                coeff_real = tmp_real * hank.real - tmp_imag * hank.imag
                coeff_imag = tmp_real * hank.imag + tmp_imag * hank.real

                term_real = coeff_real * phase_x_real - coeff_imag * phase_x_imag
                term_imag = coeff_real * phase_x_imag + coeff_imag * phase_x_real

                acc_real += term_real
                acc_imag += term_imag

    wx[j1] = acc_real + 1j * acc_imag


@cuda.jit(fastmath=True)
def particle_interaction_gpu_single_wavelength_block_row_256_lut(
    lmax: int,
    particle_number: int,
    w_idx: int,
    tau_lut: np.ndarray,
    l_lut: np.ndarray,
    m_lut: np.ndarray,
    x: np.ndarray,
    wx: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
):
    """Block-per-row single wavelength matvec using compact nmax lookup tables.

    This is the LUT-based counterpart to
    `particle_interaction_gpu_single_wavelength_block_row_256`.

    Notes
    -----
    - Requires blocks of exactly 256 threads.
    """

    t = cuda.threadIdx.x
    j1 = cuda.blockIdx.x

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax
    if j1 >= jmax:
        return

    s1 = j1 // nmax
    n1 = j1 - s1 * nmax
    tau1 = tau_lut[n1]
    l1 = l_lut[n1]
    m1 = m_lut[n1]

    acc_real = 0.0
    acc_imag = 0.0

    for j2 in range(t, jmax, 256):
        s2 = j2 // nmax
        if s2 == s1:
            continue

        base = s2 * nmax
        n2 = j2 - base
        tau2 = tau_lut[n2]
        l2 = l_lut[n2]
        m2 = m_lut[n2]

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
        phase_x_real = phase_x.real
        phase_x_imag = phase_x.imag

        for p in range(p_start, l1 + l2 + 1):
            trans = translation_table[n2, n1, p]
            leg = plm[p * (p + 1) // 2 + delta_m, s1, s2]
            hank = sph_h[p, s1, s2, w_idx]

            tmp_real = trans.real * leg.real - trans.imag * leg.imag
            tmp_imag = trans.real * leg.imag + trans.imag * leg.real
            coeff_real = tmp_real * hank.real - tmp_imag * hank.imag
            coeff_imag = tmp_real * hank.imag + tmp_imag * hank.real

            term_real = coeff_real * phase_x_real - coeff_imag * phase_x_imag
            term_imag = coeff_real * phase_x_imag + coeff_imag * phase_x_real

            acc_real += term_real
            acc_imag += term_imag

    smem_real = cuda.shared.array(256, dtype=np.float64)
    smem_imag = cuda.shared.array(256, dtype=np.float64)
    smem_real[t] = acc_real
    smem_imag[t] = acc_imag
    cuda.syncthreads()

    offset = 128
    while offset > 0:
        if t < offset:
            smem_real[t] += smem_real[t + offset]
            smem_imag[t] += smem_imag[t + offset]
        cuda.syncthreads()
        offset //= 2

    if t == 0:
        wx[j1] = smem_real[0] + 1j * smem_imag[0]


@cuda.jit(fastmath=True)
def particle_interaction_gpu_single_wavelength_lut_chunk(
    lmax: int,
    particle_number: int,
    w_idx: int,
    j1_offset: int,
    j1_count: int,
    tau_lut: np.ndarray,
    l_lut: np.ndarray,
    m_lut: np.ndarray,
    x: np.ndarray,
    wx: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
):
    """Single-wavelength LUT matvec for a contiguous chunk of output rows.

    Computes rows `[j1_offset, j1_offset + j1_count)` and stores them into
    `wx[0:j1_count]`.

    Intended for pipelining kernels and host transfers across chunks.
    """

    j_local = cuda.grid(1)
    if j_local >= j1_count:
        return

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax

    j1 = j1_offset + j_local
    if j1 >= jmax:
        return

    s1 = j1 // nmax
    n1 = j1 - s1 * nmax
    tau1 = tau_lut[n1]
    l1 = l_lut[n1]
    m1 = m_lut[n1]

    acc_real = 0.0
    acc_imag = 0.0

    for s2 in range(particle_number):
        if s2 == s1:
            continue

        base = s2 * nmax
        end = base + nmax
        for j2 in range(base, end):
            n2 = j2 - base
            tau2 = tau_lut[n2]
            l2 = l_lut[n2]
            m2 = m_lut[n2]

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
            phase_x_real = phase_x.real
            phase_x_imag = phase_x.imag

            for p in range(p_start, l1 + l2 + 1):
                trans = translation_table[n2, n1, p]
                leg = plm[p * (p + 1) // 2 + delta_m, s1, s2]
                hank = sph_h[p, s1, s2, w_idx]

                tmp_real = trans.real * leg.real - trans.imag * leg.imag
                tmp_imag = trans.real * leg.imag + trans.imag * leg.real
                coeff_real = tmp_real * hank.real - tmp_imag * hank.imag
                coeff_imag = tmp_real * hank.imag + tmp_imag * hank.real

                term_real = coeff_real * phase_x_real - coeff_imag * phase_x_imag
                term_imag = coeff_real * phase_x_imag + coeff_imag * phase_x_real

                acc_real += term_real
                acc_imag += term_imag

    wx[j_local] = acc_real + 1j * acc_imag


@cuda.jit(fastmath=True)
def particle_interaction_gpu_single_wavelength_block_row_256_lut_chunk(
    lmax: int,
    particle_number: int,
    w_idx: int,
    j1_offset: int,
    j1_count: int,
    tau_lut: np.ndarray,
    l_lut: np.ndarray,
    m_lut: np.ndarray,
    x: np.ndarray,
    wx: np.ndarray,
    translation_table: np.ndarray,
    plm: np.ndarray,
    sph_h: np.ndarray,
    e_j_dm_phi: np.ndarray,
):
    """Block-per-row LUT matvec for a contiguous output chunk.

    Notes
    -----
    - Requires blocks of exactly 256 threads.
    - Launch with `blocks_per_grid = j1_count`.
    """

    t = cuda.threadIdx.x
    j_local = cuda.blockIdx.x
    if j_local >= j1_count:
        return

    nmax = 2 * lmax * (lmax + 2)
    jmax = particle_number * nmax

    j1 = j1_offset + j_local
    if j1 >= jmax:
        return

    s1 = j1 // nmax
    n1 = j1 - s1 * nmax
    tau1 = tau_lut[n1]
    l1 = l_lut[n1]
    m1 = m_lut[n1]

    acc_real = 0.0
    acc_imag = 0.0

    for j2 in range(t, jmax, 256):
        s2 = j2 // nmax
        if s2 == s1:
            continue

        base = s2 * nmax
        n2 = j2 - base
        tau2 = tau_lut[n2]
        l2 = l_lut[n2]
        m2 = m_lut[n2]

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
        phase_x_real = phase_x.real
        phase_x_imag = phase_x.imag

        for p in range(p_start, l1 + l2 + 1):
            trans = translation_table[n2, n1, p]
            leg = plm[p * (p + 1) // 2 + delta_m, s1, s2]
            hank = sph_h[p, s1, s2, w_idx]

            tmp_real = trans.real * leg.real - trans.imag * leg.imag
            tmp_imag = trans.real * leg.imag + trans.imag * leg.real
            coeff_real = tmp_real * hank.real - tmp_imag * hank.imag
            coeff_imag = tmp_real * hank.imag + tmp_imag * hank.real

            term_real = coeff_real * phase_x_real - coeff_imag * phase_x_imag
            term_imag = coeff_real * phase_x_imag + coeff_imag * phase_x_real

            acc_real += term_real
            acc_imag += term_imag

    smem_real = cuda.shared.array(256, dtype=np.float64)
    smem_imag = cuda.shared.array(256, dtype=np.float64)
    smem_real[t] = acc_real
    smem_imag[t] = acc_imag
    cuda.syncthreads()

    offset = 128
    while offset > 0:
        if t < offset:
            smem_real[t] += smem_real[t + offset]
            smem_imag[t] += smem_imag[t + offset]
        cuda.syncthreads()
        offset //= 2

    if t == 0:
        wx[j_local] = smem_real[0] + 1j * smem_imag[0]


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

    p_real = 0.0
    p_imag = 0.0
    # for p in range(delta_m, 2 * lmax + 1):
    for p in range(delta_m, m1 + m2 + 1):
        term = (
            translation_table[n1, n2, p]
            * plm[p * (p + 1) // 2 + delta_m, s1, s2]
            * sph_h[p, s1, s2, w]
        )
        p_real += term.real
        p_imag += term.imag

    a = sfc[s1, n1, w]
    b = e_j_dm_phi[m1 - m2 + 2 * lmax, s1, s2]
    c = sfc[s2, n2, w]

    # conj(a) * b * c
    tmp = b * c
    scale_real = a.real * tmp.real + a.imag * tmp.imag
    scale_imag = a.real * tmp.imag - a.imag * tmp.real

    out_real = p_real * scale_real - p_imag * scale_imag
    out_imag = p_real * scale_imag + p_imag * scale_real

    flat_idx = j1 * channels + w
    cuda.atomic.add(c_sca_real.ravel(), flat_idx, out_real)
    cuda.atomic.add(c_sca_imag.ravel(), flat_idx, out_imag)


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

        flat_idx = (a_idx * 3 + c) * k_medium.size + w_idx
        cuda.atomic.add(e_1_sca_real.ravel(), flat_idx, e_1_sca.real)
        cuda.atomic.add(e_1_sca_imag.ravel(), flat_idx, e_1_sca.imag)


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

    flat_idx = a_idx * k_medium.size + w_idx
    cuda.atomic.add(e_field_theta_real.ravel(), flat_idx, e_field_theta.real)
    cuda.atomic.add(e_field_theta_imag.ravel(), flat_idx, e_field_theta.imag)
    cuda.atomic.add(e_field_phi_real.ravel(), flat_idx, e_field_phi.real)
    cuda.atomic.add(e_field_phi_imag.ravel(), flat_idx, e_field_phi.imag)


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
def compute_plane_wave_field_gpu(
    points: np.ndarray,
    focal_point: np.ndarray,
    direction: np.ndarray,
    k_medium: np.ndarray,
    n_medium_real: np.ndarray,
    n_medium_imag: np.ndarray,
    pol: int,
    amplitude: float,
    sin_alpha: float,
    cos_alpha: float,
    sin_beta: float,
    cos_beta: float,
    E_real: np.ndarray,
    E_imag: np.ndarray,
    H_real: np.ndarray,
    H_imag: np.ndarray,
):
    """Compute incident plane-wave E/H fields at points (CELES PVWF_components)."""
    p_idx, w_idx = cuda.grid(2)
    if (p_idx >= points.shape[0]) or (w_idx >= k_medium.size):
        return

    rx = points[p_idx, 0] - focal_point[0]
    ry = points[p_idx, 1] - focal_point[1]
    rz = points[p_idx, 2] - focal_point[2]
    phase = k_medium[w_idx] * (
        rx * direction[0] + ry * direction[1] + rz * direction[2]
    )

    c = cos(phase)
    s = sin(phase)
    e_re = amplitude * c
    e_im = amplitude * s

    # H = -1i * n * E
    nre = n_medium_real[w_idx]
    nim = n_medium_imag[w_idx]
    h_re = nre * e_im + nim * e_re
    h_im = -nre * e_re + nim * e_im

    if pol == 1:  # TE
        ex_fac = -sin_alpha
        ey_fac = cos_alpha
        ez_fac = 0.0

        hx_fac = -cos_alpha * cos_beta
        hy_fac = -sin_alpha * cos_beta
        hz_fac = sin_beta
    else:  # TM
        ex_fac = cos_alpha * cos_beta
        ey_fac = sin_alpha * cos_beta
        ez_fac = -sin_beta

        hx_fac = -sin_alpha
        hy_fac = cos_alpha
        hz_fac = 0.0

    # E components
    E_real[w_idx, p_idx, 0] = ex_fac * e_re
    E_imag[w_idx, p_idx, 0] = ex_fac * e_im
    E_real[w_idx, p_idx, 1] = ey_fac * e_re
    E_imag[w_idx, p_idx, 1] = ey_fac * e_im
    E_real[w_idx, p_idx, 2] = ez_fac * e_re
    E_imag[w_idx, p_idx, 2] = ez_fac * e_im

    # H components are multiplied by 1j*fac (as in CELES PVWF_components)
    # (1j*fac)*(h_re + 1j*h_im) => real=-fac*h_im, imag=fac*h_re
    H_real[w_idx, p_idx, 0] = -hx_fac * h_im
    H_imag[w_idx, p_idx, 0] = hx_fac * h_re
    H_real[w_idx, p_idx, 1] = -hy_fac * h_im
    H_imag[w_idx, p_idx, 1] = hy_fac * h_re
    H_real[w_idx, p_idx, 2] = -hz_fac * h_im
    H_imag[w_idx, p_idx, 2] = hz_fac * h_re


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

        flat_idx = (w * sph_h.shape[2] + sampling_idx) * 3 + c
        cuda.atomic.add(field_real.ravel(), flat_idx, term.real)
        cuda.atomic.add(field_imag.ravel(), flat_idx, term.imag)
