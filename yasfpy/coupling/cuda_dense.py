"""CUDA implementation of the dense coupling matvec.

This module implements :class:`yasfpy.coupling.ops.DenseCouplingOps` using Numba's
CUDA backend. It caches device buffers (lookup tables, input/output vectors) and
can autotune between kernel strategies for the single-wavelength matvec.

Environment Variables
---------------------
The following environment variables influence kernel selection and data movement:

- ``YASF_GPU_COUPLING_STRATEGY``: ``auto`` | ``thread_row`` | ``block_row``
- ``YASF_GPU_COUPLING_AUTOTUNE``: enable/disable autotuning
- ``YASF_GPU_COUPLING_AUTOTUNE_REPS``: repetitions per strategy candidate
- ``YASF_GPU_COUPLING_PIPELINE``: enable chunked D2H overlap
- ``YASF_GPU_COUPLING_CHUNK_ROWS``: chunk size (rows) for pipelining

Notes
-----
The coupling operator is based on VSWF translations as used in multiple-scattering
approaches :cite:`Waterman-1971-ID50` and in CELES :cite:`Egel-2017-ID1`.
"""

from __future__ import annotations

import os
from math import ceil
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numba import cuda

from yasfpy.coupling.env import (
    normalize_gpu_strategy,
    normalize_multi_wavelength_mode,
    parse_bool_env,
    parse_int_env,
)
from yasfpy.coupling.ops import DenseCouplingOps
from yasfpy.functions.cuda_numba import (
    particle_interaction_gpu,
    particle_interaction_gpu_single_wavelength_block_row_256_lut,
    particle_interaction_gpu_single_wavelength_block_row_256_lut_chunk,
    particle_interaction_gpu_single_wavelength_lut,
    particle_interaction_gpu_single_wavelength_lut_chunk,
)

if TYPE_CHECKING:  # pragma: no cover
    from yasfpy.simulation import Simulation


class CudaDenseCouplingOps(DenseCouplingOps):
    """CUDA dense coupling linear operator.

    Parameters
    ----------
    sim:
        Simulation instance providing precomputed pairwise lookup tables.

    Notes
    -----
    This implementation caches device memory for lookup tables and uses pinned
    host buffers for asynchronous copies.
    """

    def __init__(self, sim: "Simulation"):
        """Construct the operator and upload static lookup buffers."""

        self._sim = sim

        if sim.plm is None or sim.sph_h is None or sim.e_j_dm_phi is None:
            raise RuntimeError(
                "Dense coupling backend requires pairwise lookup tables but they are missing."
            )

        lmax = sim.numerics.lmax
        particle_number = sim.parameters.particles.number
        jmax = particle_number * 2 * lmax * (lmax + 2)
        self._jmax = int(jmax)
        self._particle_number = int(particle_number)
        self._lmax = int(lmax)

        idx_lookup = sim.idx_lookup

        # We treat the lookup tables as static for the lifetime of the Simulation.
        # If the Simulation is reconfigured, a new instance should be created.
        nmax = 2 * lmax * (lmax + 2)
        tau_lut = np.ascontiguousarray(idx_lookup[:nmax, 2].astype(np.int64))
        l_lut = np.ascontiguousarray(idx_lookup[:nmax, 3].astype(np.int64))
        m_lut = np.ascontiguousarray(idx_lookup[:nmax, 4].astype(np.int64))

        self._cache: dict[str, Any] = {
            "idx_device": cuda.to_device(np.ascontiguousarray(idx_lookup)),
            "tau_lut_device": cuda.to_device(tau_lut),
            "l_lut_device": cuda.to_device(l_lut),
            "m_lut_device": cuda.to_device(m_lut),
            "translation_device": cuda.to_device(
                np.ascontiguousarray(sim.numerics.translation_ab5)
            ),
            "plm_device": cuda.to_device(np.ascontiguousarray(sim.plm)),
            "e_j_dm_phi_device": cuda.to_device(np.ascontiguousarray(sim.e_j_dm_phi)),
            "matvec_calls": 0,
            "uploads": 0,
            "sph_h_uploads": 0,
            "autotune": {},
        }

    def _get_sph_h_device(self, idx: int | None) -> Any:
        """Return cached device memory for ``sph_h``.

        Parameters
        ----------
        idx:
            Optional wavelength/channel index. If provided, uploads/caches a
            single-channel slice.

        Returns
        -------
        Any
            Numba CUDA device array holding the lookup table.
        """

        sph_h = self._sim.sph_h
        assert sph_h is not None

        if idx is not None:
            sph_h_host = np.ascontiguousarray(sph_h[:, :, :, idx : idx + 1])
            key = f"sph_h_device_{idx}"
        else:
            sph_h_host = np.ascontiguousarray(sph_h)
            key = "sph_h_device_all"

        if key not in self._cache:
            self._cache[key] = cuda.to_device(sph_h_host)
            self._cache["uploads"] = int(self._cache.get("uploads", 0)) + 1
            self._cache["sph_h_uploads"] = int(self._cache.get("sph_h_uploads", 0)) + 1
        return self._cache[key]

    def _ensure_x_buffers(self, x: np.ndarray) -> None:
        """Ensure cached host/device buffers for the input vector.

        Parameters
        ----------
        x:
            Input array for the matvec.

        Notes
        -----
        We keep a pinned host buffer to enable asynchronous H2D copies and a
        device buffer to avoid reallocation on each :meth:`matvec` call.
        """

        if ("x_device" not in self._cache) or (self._cache["x_device"].size != x.size):
            self._cache["x_device"] = cuda.device_array(x.shape, dtype=x.dtype)
            self._cache["uploads"] = int(self._cache.get("uploads", 0)) + 1

        if ("x_host" not in self._cache) or (self._cache["x_host"].size != x.size):
            self._cache["x_host"] = cuda.pinned_array(x.shape, dtype=x.dtype)

    def matvec(self, x: np.ndarray, idx: int | None = None) -> np.ndarray:
        """Compute the dense coupling matvec on the GPU.

        Parameters
        ----------
        x:
            Input array. For multi-wavelength usage, the trailing dimension is the
            wavelength/channel axis.
        idx:
            Optional wavelength/channel index. When provided, uses a
            single-wavelength kernel (avoids atomics).

        Returns
        -------
        numpy.ndarray
            The product ``W @ x``. If ``idx`` is not ``None``, the result is
            squeezed to drop the singleton channel axis.
        """

        sim = self._sim
        jmax = self._jmax
        particle_number = self._particle_number
        lmax = self._lmax

        # Keep host and device buffers stable.
        self._ensure_x_buffers(x)

        x_device = self._cache["x_device"]
        x_host = cast(np.ndarray, self._cache["x_host"])
        np.copyto(x_host, x)

        self._cache["matvec_calls"] = int(self._cache.get("matvec_calls", 0)) + 1

        sph_h_device = self._get_sph_h_device(idx)

        # For solver-like usage (idx provided), we use the single-wavelength kernel
        # to avoid atomics.
        if idx is not None:
            if "stream" not in self._cache:
                self._cache["stream"] = cuda.stream()
            stream = cast(cuda.stream, self._cache["stream"])

            if ("wx_device" not in self._cache) or (
                self._cache["wx_device"].size != x.size
            ):
                self._cache["wx_device"] = cuda.device_array(
                    x.shape, dtype=np.complex128, stream=stream
                )

            wx_device = self._cache["wx_device"]

            if ("wx_host" not in self._cache) or (
                self._cache["wx_host"].size != x.size
            ):
                self._cache["wx_host"] = cuda.pinned_array(x.shape, dtype=np.complex128)

            wx_host = cast(np.ndarray, self._cache["wx_host"])

            # Async H2D of input (pinned host buffer).
            x_device.copy_to_device(x_host, stream=stream)

            # Record an event so other streams can depend on the upload
            # without blocking the host thread.
            if "x_ready_event" not in self._cache:
                self._cache["x_ready_event"] = cuda.event(timing=False)
            x_ready_event = cast(Any, self._cache["x_ready_event"])
            x_ready_event.record(stream)

            strategy_env = normalize_gpu_strategy(
                os.environ.get("YASF_GPU_COUPLING_STRATEGY", "auto")
            )

            threads_per_block = 256

            autotune = parse_bool_env(
                "YASF_GPU_COUPLING_AUTOTUNE", default=(idx is None)
            )
            autotune_reps = parse_int_env(
                "YASF_GPU_COUPLING_AUTOTUNE_REPS", default=2, minimum=1
            )

            strategy_key = ("single", int(jmax), str(x.dtype))
            cached = cast(dict[Any, Any], self._cache.setdefault("autotune", {}))

            strategy: str
            if strategy_env != "auto":
                strategy = strategy_env
            elif autotune:
                strategy = cast(str, cached.get(strategy_key, "auto"))
            else:
                strategy = "auto"

            if strategy == "auto":
                if jmax < 20000:
                    strategy = "block_row"
                else:
                    strategy = "thread_row"

            if (strategy_env == "auto") and autotune and (strategy_key not in cached):
                timings: dict[str, float] = {}
                for candidate in ("thread_row", "block_row"):
                    if candidate == "block_row":
                        blocks = jmax
                        kernel = (
                            particle_interaction_gpu_single_wavelength_block_row_256_lut
                        )
                    else:
                        blocks = ceil(jmax / threads_per_block)
                        kernel = particle_interaction_gpu_single_wavelength_lut

                    wx_device_tmp = cuda.device_array(x.shape, dtype=np.complex128)
                    start = perf_counter()
                    for _ in range(autotune_reps):
                        kernel[blocks, threads_per_block, stream](
                            lmax,
                            particle_number,
                            0,
                            self._cache["tau_lut_device"],
                            self._cache["l_lut_device"],
                            self._cache["m_lut_device"],
                            x_device,
                            wx_device_tmp,
                            self._cache["translation_device"],
                            self._cache["plm_device"],
                            sph_h_device,
                            self._cache["e_j_dm_phi_device"],
                        )
                    stream.synchronize()
                    timings[candidate] = perf_counter() - start

                best = (
                    "thread_row"
                    if timings["thread_row"] <= timings["block_row"]
                    else "block_row"
                )
                cached[strategy_key] = best
                strategy = best

            if strategy == "block_row":
                blocks_per_grid = jmax
                kernel = particle_interaction_gpu_single_wavelength_block_row_256_lut
            else:
                blocks_per_grid = ceil(jmax / threads_per_block)
                kernel = particle_interaction_gpu_single_wavelength_lut

            kernel[blocks_per_grid, threads_per_block, stream](
                lmax,
                particle_number,
                0,
                self._cache["tau_lut_device"],
                self._cache["l_lut_device"],
                self._cache["m_lut_device"],
                x_device,
                wx_device,
                self._cache["translation_device"],
                self._cache["plm_device"],
                sph_h_device,
                self._cache["e_j_dm_phi_device"],
            )

            # Optionally pipeline the matvec in chunks so D2H overlaps with compute.
            pipeline = os.environ.get("YASF_GPU_COUPLING_PIPELINE", "1") not in (
                "0",
                "",
            )
            chunk_rows_env = os.environ.get("YASF_GPU_COUPLING_CHUNK_ROWS", "").strip()
            if chunk_rows_env:
                try:
                    chunk_rows = max(1, int(chunk_rows_env))
                except ValueError:
                    chunk_rows = 16384
            else:
                chunk_rows = 16384

            if (not pipeline) or (jmax <= chunk_rows):
                wx_device.copy_to_host(wx_host, stream=stream)
                stream.synchronize()
                wx = np.asarray(wx_host)
            else:
                if "streams" not in self._cache:
                    self._cache["streams"] = [cuda.stream(), cuda.stream()]

                streams = cast(list[cuda.stream], self._cache["streams"])

                if x_ready_event is not None:
                    x_ready_event.wait(streams[0])
                    x_ready_event.wait(streams[1])

                wx_chunk_device = cast(
                    list[Any], self._cache.get("wx_chunk_device", [])
                )
                if (len(wx_chunk_device) != 2) or (
                    cast(Any, wx_chunk_device[0]).size != chunk_rows
                ):
                    wx_chunk_device = [
                        cuda.device_array((chunk_rows,), dtype=np.complex128),
                        cuda.device_array((chunk_rows,), dtype=np.complex128),
                    ]
                    self._cache["wx_chunk_device"] = wx_chunk_device

                wx_chunk_host = cast(
                    list[np.ndarray], self._cache.get("wx_chunk_host", [])
                )
                if (len(wx_chunk_host) != 2) or (wx_chunk_host[0].size != chunk_rows):
                    wx_chunk_host = [
                        cuda.pinned_array((chunk_rows,), dtype=np.complex128),
                        cuda.pinned_array((chunk_rows,), dtype=np.complex128),
                    ]
                    self._cache["wx_chunk_host"] = wx_chunk_host

                if ("wx_host_full" not in self._cache) or (
                    self._cache["wx_host_full"].size != x.size
                ):
                    self._cache["wx_host_full"] = cuda.pinned_array(
                        x.shape, dtype=np.complex128
                    )

                wx_host_full = cast(np.ndarray, self._cache["wx_host_full"])

                if strategy == "block_row":
                    chunk_kernel = particle_interaction_gpu_single_wavelength_block_row_256_lut_chunk
                else:
                    chunk_kernel = particle_interaction_gpu_single_wavelength_lut_chunk

                prev_offset = [-1, -1]
                prev_count = [0, 0]

                for chunk_start in range(0, jmax, chunk_rows):
                    buf = (chunk_start // chunk_rows) & 1
                    s = streams[buf]

                    if prev_offset[buf] >= 0:
                        s.synchronize()
                        off = prev_offset[buf]
                        cnt = prev_count[buf]
                        wx_host_full[off : off + cnt] = wx_chunk_host[buf][:cnt]

                    cur = min(chunk_rows, jmax - chunk_start)

                    if strategy == "block_row":
                        chunk_kernel[cur, threads_per_block, s](
                            lmax,
                            particle_number,
                            0,
                            chunk_start,
                            cur,
                            self._cache["tau_lut_device"],
                            self._cache["l_lut_device"],
                            self._cache["m_lut_device"],
                            x_device,
                            wx_chunk_device[buf],
                            self._cache["translation_device"],
                            self._cache["plm_device"],
                            sph_h_device,
                            self._cache["e_j_dm_phi_device"],
                        )
                    else:
                        blocks = ceil(cur / threads_per_block)
                        chunk_kernel[blocks, threads_per_block, s](
                            lmax,
                            particle_number,
                            0,
                            chunk_start,
                            cur,
                            self._cache["tau_lut_device"],
                            self._cache["l_lut_device"],
                            self._cache["m_lut_device"],
                            x_device,
                            wx_chunk_device[buf],
                            self._cache["translation_device"],
                            self._cache["plm_device"],
                            sph_h_device,
                            self._cache["e_j_dm_phi_device"],
                        )

                    wx_chunk_device[buf].copy_to_host(wx_chunk_host[buf], stream=s)
                    prev_offset[buf] = chunk_start
                    prev_count[buf] = cur

                for buf in (0, 1):
                    if prev_offset[buf] >= 0:
                        streams[buf].synchronize()
                        off = prev_offset[buf]
                        cnt = prev_count[buf]
                        wx_host_full[off : off + cnt] = wx_chunk_host[buf][:cnt]

                wx = np.asarray(wx_host_full)

            return np.squeeze(wx)

        # Multi-wavelength path.
        wavelengths_size = sim.parameters.k_medium.shape[0]

        if "stream_multi" not in self._cache:
            self._cache["stream_multi"] = cuda.stream()
        stream_multi = cast(cuda.stream, self._cache["stream_multi"])

        x_device.copy_to_device(x_host, stream=stream_multi)

        if "x_ready_event" not in self._cache:
            self._cache["x_ready_event"] = cuda.event(timing=False)
        x_ready_event = cast(Any, self._cache["x_ready_event"])
        x_ready_event.record(stream_multi)

        mode_env = normalize_multi_wavelength_mode(
            os.environ.get("YASF_GPU_COUPLING_MULTI_WAVELENGTH", "auto")
        )

        autotune = parse_bool_env("YASF_GPU_COUPLING_AUTOTUNE", default=True)
        autotune_reps = parse_int_env(
            "YASF_GPU_COUPLING_AUTOTUNE_REPS", default=2, minimum=1
        )

        mode_key = ("multi", int(jmax), int(wavelengths_size), str(x.dtype))
        cached = cast(dict[Any, Any], self._cache.setdefault("autotune", {}))

        mode: str
        if mode_env != "auto":
            mode = mode_env
        elif autotune:
            mode = cast(str, cached.get(mode_key, "auto"))
        else:
            mode = "auto"

        if mode == "auto":
            mode = "per_wavelength" if wavelengths_size <= 8 else "atomic"

        if (mode_env == "auto") and autotune and (mode_key not in cached):
            timings: dict[str, float] = {}

            wx_real_tmp = np.zeros(jmax * wavelengths_size, dtype=float)
            wx_imag_tmp = np.zeros_like(wx_real_tmp)
            wx_real_dev = cuda.to_device(wx_real_tmp)
            wx_imag_dev = cuda.to_device(wx_imag_tmp)

            atomic_threads = (16, 16, 2)
            atomic_blocks = (
                ceil(jmax / atomic_threads[0]),
                ceil(jmax / atomic_threads[1]),
                ceil(wavelengths_size / atomic_threads[2]),
            )

            pw_out = cuda.device_array(x.shape, dtype=np.complex128)

            strategy_env = normalize_gpu_strategy(
                os.environ.get("YASF_GPU_COUPLING_STRATEGY", "auto")
            )
            threads_per_block = 256
            if strategy_env == "block_row" or (strategy_env == "auto" and jmax < 20000):
                pw_kernel = particle_interaction_gpu_single_wavelength_block_row_256_lut
                pw_blocks = jmax
            else:
                pw_kernel = particle_interaction_gpu_single_wavelength_lut
                pw_blocks = ceil(jmax / threads_per_block)

            for candidate in ("atomic", "per_wavelength"):
                start = perf_counter()
                if candidate == "atomic":
                    for _ in range(autotune_reps):
                        wx_real_dev.copy_to_device(wx_real_tmp)
                        wx_imag_dev.copy_to_device(wx_imag_tmp)
                        particle_interaction_gpu[
                            atomic_blocks, atomic_threads, stream_multi
                        ](
                            lmax,
                            particle_number,
                            self._cache["idx_device"],
                            x_device,
                            wx_real_dev,
                            wx_imag_dev,
                            self._cache["translation_device"],
                            self._cache["plm_device"],
                            sph_h_device,
                            self._cache["e_j_dm_phi_device"],
                        )
                    stream_multi.synchronize()
                else:
                    for _ in range(autotune_reps):
                        pw_kernel[pw_blocks, threads_per_block, stream_multi](
                            lmax,
                            particle_number,
                            0,
                            self._cache["tau_lut_device"],
                            self._cache["l_lut_device"],
                            self._cache["m_lut_device"],
                            x_device,
                            pw_out,
                            self._cache["translation_device"],
                            self._cache["plm_device"],
                            sph_h_device,
                            self._cache["e_j_dm_phi_device"],
                        )
                    stream_multi.synchronize()
                timings[candidate] = perf_counter() - start

            best = (
                "atomic"
                if timings["atomic"] <= timings["per_wavelength"]
                else "per_wavelength"
            )
            cached[mode_key] = best
            mode = best

        if mode in {"per_wavelength", "per_wavelength_chunked"}:
            if "streams_wavelength" not in self._cache:
                self._cache["streams_wavelength"] = [cuda.stream(), cuda.stream()]

            streams_wl = cast(list[cuda.stream], self._cache["streams_wavelength"])
            x_ready_event.wait(streams_wl[0])
            x_ready_event.wait(streams_wl[1])

            wx_wavelength_device = cast(
                list[Any], self._cache.get("wx_wavelength_device", [])
            )
            if (len(wx_wavelength_device) != 2) or (
                cast(Any, wx_wavelength_device[0]).size != x.size
            ):
                wx_wavelength_device = [
                    cuda.device_array(x.shape, dtype=np.complex128),
                    cuda.device_array(x.shape, dtype=np.complex128),
                ]
                self._cache["wx_wavelength_device"] = wx_wavelength_device

            wx_wavelength_host = cast(
                list[np.ndarray], self._cache.get("wx_wavelength_host", [])
            )
            if (len(wx_wavelength_host) != 2) or (wx_wavelength_host[0].size != x.size):
                wx_wavelength_host = [
                    cuda.pinned_array(x.shape, dtype=np.complex128),
                    cuda.pinned_array(x.shape, dtype=np.complex128),
                ]
                self._cache["wx_wavelength_host"] = wx_wavelength_host

            wx_wl_dev = wx_wavelength_device
            wx_wl_host = wx_wavelength_host

            strategy_env = normalize_gpu_strategy(
                os.environ.get("YASF_GPU_COUPLING_STRATEGY", "auto")
            )
            threads_per_block = 256

            if strategy_env == "block_row" or (strategy_env == "auto" and jmax < 20000):
                wl_kernel = particle_interaction_gpu_single_wavelength_block_row_256_lut
                wl_strategy = "block_row"
            else:
                wl_kernel = particle_interaction_gpu_single_wavelength_lut
                wl_strategy = "thread_row"

            chunk_rows = parse_int_env(
                "YASF_GPU_COUPLING_CHUNK_ROWS", default=16384, minimum=1
            )

            if mode == "per_wavelength_chunked" and jmax > chunk_rows:
                if "wx_tile_device" not in self._cache or (
                    cast(list[Any], self._cache.get("wx_tile_device", []))[0].size
                    != chunk_rows
                ):
                    self._cache["wx_tile_device"] = [
                        cuda.device_array((chunk_rows,), dtype=np.complex128),
                        cuda.device_array((chunk_rows,), dtype=np.complex128),
                    ]

                if "wx_tile_host" not in self._cache or (
                    cast(list[np.ndarray], self._cache.get("wx_tile_host", []))[0].size
                    != chunk_rows
                ):
                    self._cache["wx_tile_host"] = [
                        cuda.pinned_array((chunk_rows,), dtype=np.complex128),
                        cuda.pinned_array((chunk_rows,), dtype=np.complex128),
                    ]

                wx_tile_dev = cast(list[Any], self._cache["wx_tile_device"])
                wx_tile_host = cast(list[np.ndarray], self._cache["wx_tile_host"])

                if ("wx_host_full_multi" not in self._cache) or (
                    self._cache["wx_host_full_multi"].shape
                    != (int(x.size), int(wavelengths_size))
                ):
                    self._cache["wx_host_full_multi"] = cuda.pinned_array(
                        (x.size, wavelengths_size), dtype=np.complex128
                    )

                wx_full = cast(np.ndarray, self._cache["wx_host_full_multi"])

                if wl_strategy == "block_row":
                    chunk_kernel = particle_interaction_gpu_single_wavelength_block_row_256_lut_chunk
                else:
                    chunk_kernel = particle_interaction_gpu_single_wavelength_lut_chunk

                prev_tile_off = [-1, -1]
                prev_tile_cnt = [0, 0]
                prev_tile_w = [-1, -1]

                tile_idx = 0
                for w in range(wavelengths_size):
                    for chunk_start in range(0, jmax, chunk_rows):
                        buf = tile_idx & 1
                        s = streams_wl[buf]

                        if prev_tile_off[buf] >= 0:
                            s.synchronize()
                            off = prev_tile_off[buf]
                            cnt = prev_tile_cnt[buf]
                            ww = prev_tile_w[buf]
                            wx_full[off : off + cnt, ww] = wx_tile_host[buf][:cnt]

                        cur = min(chunk_rows, jmax - chunk_start)
                        if wl_strategy == "block_row":
                            chunk_kernel[cur, threads_per_block, s](
                                lmax,
                                particle_number,
                                w,
                                chunk_start,
                                cur,
                                self._cache["tau_lut_device"],
                                self._cache["l_lut_device"],
                                self._cache["m_lut_device"],
                                x_device,
                                wx_tile_dev[buf],
                                self._cache["translation_device"],
                                self._cache["plm_device"],
                                sph_h_device,
                                self._cache["e_j_dm_phi_device"],
                            )
                        else:
                            blocks = ceil(cur / threads_per_block)
                            chunk_kernel[blocks, threads_per_block, s](
                                lmax,
                                particle_number,
                                w,
                                chunk_start,
                                cur,
                                self._cache["tau_lut_device"],
                                self._cache["l_lut_device"],
                                self._cache["m_lut_device"],
                                x_device,
                                wx_tile_dev[buf],
                                self._cache["translation_device"],
                                self._cache["plm_device"],
                                sph_h_device,
                                self._cache["e_j_dm_phi_device"],
                            )

                        wx_tile_dev[buf].copy_to_host(wx_tile_host[buf], stream=s)
                        prev_tile_off[buf] = chunk_start
                        prev_tile_cnt[buf] = cur
                        prev_tile_w[buf] = w
                        tile_idx += 1

                for buf in (0, 1):
                    if prev_tile_off[buf] >= 0:
                        streams_wl[buf].synchronize()
                        off = prev_tile_off[buf]
                        cnt = prev_tile_cnt[buf]
                        ww = prev_tile_w[buf]
                        wx_full[off : off + cnt, ww] = wx_tile_host[buf][:cnt]

                return np.asarray(wx_full)

            if wl_strategy == "thread_row":
                wl_blocks = ceil(jmax / threads_per_block)
            else:
                wl_blocks = jmax

            wx_full = np.empty((x.size, wavelengths_size), dtype=np.complex128)

            prev_w = [-1, -1]
            for w in range(wavelengths_size):
                buf = w & 1
                s = streams_wl[buf]

                if prev_w[buf] >= 0:
                    s.synchronize()
                    wx_full[:, prev_w[buf]] = wx_wl_host[buf]

                wl_kernel[wl_blocks, threads_per_block, s](
                    lmax,
                    particle_number,
                    w,
                    self._cache["tau_lut_device"],
                    self._cache["l_lut_device"],
                    self._cache["m_lut_device"],
                    x_device,
                    wx_wl_dev[buf],
                    self._cache["translation_device"],
                    self._cache["plm_device"],
                    sph_h_device,
                    self._cache["e_j_dm_phi_device"],
                )

                wx_wl_dev[buf].copy_to_host(wx_wl_host[buf], stream=s)
                prev_w[buf] = w

            for buf in (0, 1):
                if prev_w[buf] >= 0:
                    streams_wl[buf].synchronize()
                    wx_full[:, prev_w[buf]] = wx_wl_host[buf]

            return wx_full

        # Atomic kernel across all channels.
        stream_multi.synchronize()

        wx_real = np.zeros(jmax * wavelengths_size, dtype=float)
        wx_imag = np.zeros_like(wx_real)

        wx_real_device = cuda.to_device(wx_real)
        wx_imag_device = cuda.to_device(wx_imag)

        threads_per_block = (16, 16, 2)
        blocks_per_grid_x = ceil(jmax / threads_per_block[0])
        blocks_per_grid_y = ceil(jmax / threads_per_block[1])
        blocks_per_grid_z = ceil(wavelengths_size / threads_per_block[2])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

        particle_interaction_gpu[blocks_per_grid, threads_per_block](
            lmax,
            particle_number,
            self._cache["idx_device"],
            x_device,
            wx_real_device,
            wx_imag_device,
            self._cache["translation_device"],
            self._cache["plm_device"],
            sph_h_device,
            self._cache["e_j_dm_phi_device"],
        )

        wx_real = wx_real_device.copy_to_host().reshape((jmax, wavelengths_size))
        wx_imag = wx_imag_device.copy_to_host().reshape((jmax, wavelengths_size))
        return wx_real + 1j * wx_imag
