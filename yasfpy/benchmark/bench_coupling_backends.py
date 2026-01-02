"""Benchmarks for dense coupling backends.

Compares CPU/GPU and alternative coupling implementations under controlled,
reproducible settings.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pyperf

from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver

from numba import cuda


def _set_reproducible_thread_env() -> None:
    """Set conservative thread environment variables.

    Notes
    -----
    Uses ``os.environ.setdefault`` so user-provided values win.
    """
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _add_worker_args(cmd: list[str], args: argparse.Namespace) -> None:
    """Populate pyperf worker command-line arguments."""
    cmd.extend(["--particles", str(args.particles)])
    cmd.extend(["--wavelengths", str(args.wavelengths)])
    cmd.extend(["--lmax", str(args.lmax)])
    cmd.extend(["--tile-size", str(args.tile_size)])
    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--near-radius", str(args.near_radius)])

    if args.mode != "slice":
        cmd.extend(["--mode", str(args.mode)])

    if args.scope != "steady":
        cmd.extend(["--scope", str(args.scope)])

    if args.log_quiet:
        cmd.append("--log-quiet")

    if getattr(args, "gpu", False):
        cmd.append("--gpu")

    gpu_strategy = getattr(args, "gpu_strategy", "auto")
    if gpu_strategy != "auto":
        cmd.extend(["--gpu-strategy", gpu_strategy])


def _build_runner() -> tuple[pyperf.Runner, argparse.ArgumentParser]:
    """Create the pyperf runner and CLI parser."""
    parser = argparse.ArgumentParser(
        description="Benchmark coupling matvec for dense vs tiled_dense",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=64,
        help="Number of particles (dense uses O(N^2) memory)",
    )
    parser.add_argument(
        "--wavelengths",
        type=int,
        default=8,
        help="Number of wavelengths / k_medium channels",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=1,
        help="Spherical harmonic cutoff",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=64,
        dest="tile_size",
        help="Particle tile size for tiled_dense",
    )
    parser.add_argument(
        "--near-radius",
        type=float,
        default=0.0,
        dest="near_radius",
        help="Near-field cutoff radius for nearfar backend",
    )
    parser.add_argument(
        "--mode",
        choices=("slice", "all"),
        default="slice",
        help="Benchmark per-wavelength slice (solver-like) or all channels at once",
    )
    parser.add_argument(
        "--scope",
        choices=("steady", "setup"),
        default="steady",
        help="Benchmark steady-state matvecs or setup + first matvec",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic geometry/vector",
    )
    parser.add_argument(
        "--log-quiet",
        dest="log_quiet",
        action="store_true",
        help="Suppress noisy Python-side logging",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable Numba CUDA for dense backend (skips if unavailable)",
    )
    parser.add_argument(
        "--gpu-strategy",
        choices=("auto", "thread_row", "block_row"),
        default="auto",
        dest="gpu_strategy",
        help="Dense GPU single-wavelength kernel strategy",
    )

    runner = pyperf.Runner(
        _argparser=parser,
        add_cmdline_args=_add_worker_args,
        processes=1,
        warmups=1,
    )
    return runner, parser


def _make_simulation(
    *,
    particle_number: int,
    wavelengths: int,
    lmax: int,
    coupling_backend: str,
    coupling_tile_size: int,
    coupling_near_field_radius: float | None,
    seed: int,
    gpu: bool,
) -> Simulation:
    """Create a small synthetic simulation for benchmarking.

    Returns
    -------
    yasfpy.simulation.Simulation
        Simulation instance configured for coupling matvec benchmarks.
    """
    rng = np.random.default_rng(seed)

    # Spread particles out a bit to avoid r=0 singularities.
    position = rng.normal(size=(particle_number, 3))
    position *= 2.0

    particles = Particles(
        position=position,
        r=np.full(particle_number, 0.1, dtype=float),
        refractive_index=np.full(particle_number, 1.5 + 0.0j, dtype=complex),
    )

    initial_field = InitialField(
        beam_width=0,
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="TE",
    )

    wavelength = np.linspace(1.0, 1.0 + 0.02 * (wavelengths - 1), wavelengths)
    parameters = Parameters(
        wavelength=wavelength,
        medium_refractive_index=np.full(wavelengths, 1.0 + 0.0j, dtype=complex),
        particles=particles,
        initial_field=initial_field,
    )

    solver = Solver(
        solver_type="gmres",
        tolerance=1e-6,
        max_iter=5,
        restart=5,
    )

    numerics = Numerics(
        lmax=lmax,
        sampling_points_number=np.array([10]),
        particle_distance_resolution=1.0,
        gpu=gpu,
        solver=solver,
        coupling_backend=coupling_backend,
        coupling_tile_size=coupling_tile_size,
        coupling_near_field_radius=coupling_near_field_radius,
    )
    numerics.compute_translation_table()

    return Simulation(parameters, numerics)


def main() -> None:
    """CLI entry point for the coupling backend benchmark."""
    _set_reproducible_thread_env()

    runner, _ = _build_runner()
    args = runner.parse_args()

    gpu_available = False
    if args.gpu:
        # pyperf may sanitize the worker environment; re-check after parsing args.
        if cuda.is_available():
            try:
                cuda.get_current_device()
                gpu_available = True
            except Exception:
                gpu_available = False

    if args.gpu_strategy != "auto":
        os.environ["YASF_GPU_COUPLING_STRATEGY"] = args.gpu_strategy

    if args.log_quiet:
        import logging

        logging.getLogger().setLevel(logging.ERROR)

    if args.scope == "setup":
        # Includes simulation setup (lookup tables, neighbor caching) plus first matvec.
        def _bench_setup_dense() -> np.ndarray:
            dense = _make_simulation(
                particle_number=args.particles,
                wavelengths=args.wavelengths,
                lmax=args.lmax,
                coupling_backend="dense",
                coupling_tile_size=args.tile_size,
                coupling_near_field_radius=None,
                seed=args.seed,
                gpu=args.gpu and gpu_available,
            )
            jmax = args.particles * dense.numerics.nmax
            rng = np.random.default_rng(args.seed + 1)
            x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)
            slice_idx = 0 if args.mode == "slice" else None
            return dense.coupling_matrix_multiply(x, idx=slice_idx)

        def _bench_setup_tiled() -> np.ndarray:
            tiled = _make_simulation(
                particle_number=args.particles,
                wavelengths=args.wavelengths,
                lmax=args.lmax,
                coupling_backend="tiled_dense",
                coupling_tile_size=args.tile_size,
                coupling_near_field_radius=None,
                seed=args.seed,
                gpu=False,
            )
            jmax = args.particles * tiled.numerics.nmax
            rng = np.random.default_rng(args.seed + 1)
            x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)
            slice_idx = 0 if args.mode == "slice" else None
            return tiled.coupling_matrix_multiply(x, idx=slice_idx)

        def _bench_setup_nearfar() -> np.ndarray:
            nearfar = _make_simulation(
                particle_number=args.particles,
                wavelengths=args.wavelengths,
                lmax=args.lmax,
                coupling_backend="nearfar",
                coupling_tile_size=args.tile_size,
                coupling_near_field_radius=args.near_radius,
                seed=args.seed,
                gpu=False,
            )
            jmax = args.particles * nearfar.numerics.nmax
            rng = np.random.default_rng(args.seed + 1)
            x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)
            slice_idx = 0 if args.mode == "slice" else None
            return nearfar.coupling_matrix_multiply(x, idx=slice_idx)

        runner.bench_func("coupling_setup_plus_matvec_dense", _bench_setup_dense)
        runner.bench_func("coupling_setup_plus_matvec_tiled_dense", _bench_setup_tiled)
        runner.bench_func("coupling_setup_plus_matvec_nearfar", _bench_setup_nearfar)
        return

    # Steady-state benchmark: build simulations and time matvec only.
    dense = _make_simulation(
        particle_number=args.particles,
        wavelengths=args.wavelengths,
        lmax=args.lmax,
        coupling_backend="dense",
        coupling_tile_size=args.tile_size,
        coupling_near_field_radius=None,
        seed=args.seed,
        gpu=args.gpu and gpu_available,
    )
    tiled = _make_simulation(
        particle_number=args.particles,
        wavelengths=args.wavelengths,
        lmax=args.lmax,
        coupling_backend="tiled_dense",
        coupling_tile_size=args.tile_size,
        coupling_near_field_radius=None,
        seed=args.seed,
        gpu=False,
    )
    nearfar = _make_simulation(
        particle_number=args.particles,
        wavelengths=args.wavelengths,
        lmax=args.lmax,
        coupling_backend="nearfar",
        coupling_tile_size=args.tile_size,
        coupling_near_field_radius=args.near_radius,
        seed=args.seed,
        gpu=False,
    )

    jmax = args.particles * dense.numerics.nmax
    rng = np.random.default_rng(args.seed + 1)
    x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)

    slice_idx: int | None
    if args.mode == "slice":
        slice_idx = 0
    else:
        slice_idx = None

    # Warm up Numba compilation and caches.
    dense.coupling_matrix_multiply(x, idx=slice_idx)
    tiled.coupling_matrix_multiply(x, idx=slice_idx)
    nearfar.coupling_matrix_multiply(x, idx=slice_idx)

    if args.gpu and gpu_available:
        suffix = args.gpu_strategy
        if suffix == "auto":
            suffix = os.environ.get("YASF_GPU_COUPLING_STRATEGY", "auto")
        dense_label = f"coupling_matvec_dense_gpu_{suffix}"
    else:
        dense_label = "coupling_matvec_dense_cpu"
    runner.bench_func(
        dense_label,
        lambda: dense.coupling_matrix_multiply(x, idx=slice_idx),
    )
    runner.bench_func(
        "coupling_matvec_tiled_dense",
        lambda: tiled.coupling_matrix_multiply(x, idx=slice_idx),
    )
    runner.bench_func(
        "coupling_matvec_nearfar",
        lambda: nearfar.coupling_matrix_multiply(x, idx=slice_idx),
    )


if __name__ == "__main__":
    main()
