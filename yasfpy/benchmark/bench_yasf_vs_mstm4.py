"""Benchmark script comparing YASF and MSTM4.

This module runs comparable configurations across the two solvers and collects
outputs for performance/accuracy comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyperf

from yasfpy.benchmark.mstm4 import MSTM4Manager
from yasfpy.yasf import YASF


def _make_config_paths_absolute(cfg: dict, base_dir: Path) -> None:
    """Make dataset paths in a config absolute.

    The benchmark can create a temporary configuration file. This helper ensures
    that relative paths in the config remain valid by resolving them against the
    directory of the original config file.

    Parameters
    ----------
    cfg
        Parsed configuration dictionary (mutated in-place).
    base_dir
        Base directory used to resolve relative paths.
    """

    particles = cfg.get("particles")
    if not isinstance(particles, dict):
        return

    geometry = particles.get("geometry")
    if isinstance(geometry, dict) and isinstance(geometry.get("file"), str):
        file_path = Path(geometry["file"])
        if not file_path.is_absolute():
            geometry["file"] = str((base_dir / file_path).resolve())

    materials = particles.get("material")
    if isinstance(materials, list):
        for material in materials:
            if not isinstance(material, dict):
                continue
            path = material.get("path")
            if isinstance(path, str):
                path_obj = Path(path)
                if not path_obj.is_absolute():
                    material["path"] = str((base_dir / path_obj).resolve())


def _set_reproducible_thread_env() -> None:
    """Set deterministic thread-related environment defaults.

    Notes
    -----
    This helper sets common BLAS/Numba/OpenMP thread-count environment variables
    to ``"1"`` only if they are not already defined (conservative override).
    """

    # Keep this conservative: do not override if user already set.
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _add_worker_args(cmd: list[str], args: argparse.Namespace) -> None:
    """Append benchmark-specific CLI flags for pyperf worker processes.

    Parameters
    ----------
    cmd
        Command list that will be executed by pyperf. This list is mutated
        in-place.
    args
        Parsed CLI arguments.
    """

    # pyperf spawns worker processes with only its own args.
    # Re-add our benchmark-specific args so the worker can parse them.
    cmd.extend(["--config", str(args.config)])

    if getattr(args, "cluster", ""):
        cmd.extend(["--cluster", str(args.cluster)])

    cmd.extend(["--mstm-binary", str(args.mstm_binary)])
    cmd.extend(["--mstm-parallel", str(args.mstm_parallel)])

    if getattr(args, "yasf_quiet", False):
        cmd.append("--yasf-quiet")

    if getattr(args, "mstm_silent", False):
        cmd.append("--mstm-silent")

    if getattr(args, "yasf_no_phase_function", False):
        cmd.append("--yasf-no-phase-function")

    if getattr(args, "yasf_force_cpu", False):
        cmd.append("--yasf-force-cpu")

    workdir = getattr(args, "workdir", None)
    if workdir is not None:
        cmd.extend(["--workdir", str(workdir)])


def _build_runner() -> tuple[pyperf.Runner, argparse.ArgumentParser]:
    """Construct the CLI parser and the pyperf runner.

    Returns
    -------
    runner, parser
        The configured pyperf runner and its underlying argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark YASF vs MSTM4 using pyperf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YASF config JSON",
    )
    parser.add_argument(
        "--cluster",
        default="",
        help="Optional cluster override (path), passed to both backends",
    )
    parser.add_argument(
        "--mstm-binary",
        default="mstm",
        help="MSTM4 executable name/path",
    )
    parser.add_argument(
        "--mstm-parallel",
        type=int,
        default=4,
        help="MSTM4 MPI processes (must be multiple of 4 for this wrapper)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Workdir for MSTM4 input/output files",
    )
    parser.add_argument(
        "--yasf-quiet",
        dest="yasf_quiet",
        action="store_true",
        help="Suppress noisy Python-side output/logging",
    )
    parser.add_argument(
        "--mstm-silent",
        action="store_true",
        help="Suppress MSTM stdout/stderr",
    )
    parser.add_argument(
        "--yasf-no-phase-function",
        dest="yasf_no_phase_function",
        action="store_true",
        help="Disable expensive phase function calculation (benchmark helper)",
    )
    parser.add_argument(
        "--yasf-force-cpu",
        dest="yasf_force_cpu",
        action="store_true",
        help="Force CPU path by setting numerics.gpu=false (benchmark helper)",
    )

    runner = pyperf.Runner(
        _argparser=parser,
        add_cmdline_args=_add_worker_args,
        processes=1,
        warmups=1,
    )
    return runner, parser


def main() -> None:
    """CLI entry point to benchmark YASF against MSTM4."""

    _set_reproducible_thread_env()

    runner, _ = _build_runner()
    args = runner.parse_args()

    if args.yasf_quiet:
        import logging

        logging.getLogger().setLevel(logging.ERROR)

    config_path = Path(args.config).expanduser().resolve()

    if shutil.which(str(args.mstm_binary)) is None:
        raise FileNotFoundError(f"MSTM binary '{args.mstm_binary}' not found in PATH")

    workdir_tmp: tempfile.TemporaryDirectory[str] | None = None
    workdir = args.workdir
    if workdir is not None:
        workdir = Path(workdir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir_tmp = tempfile.TemporaryDirectory()
        workdir = Path(workdir_tmp.name)

    # For large clusters, the default optics + GPU path can be very slow or
    # fail when CUDA isn't available. Keep the benchmark script robust by
    # allowing a lightweight optics path.
    config_tmp: tempfile.TemporaryDirectory[str] | None = None
    config_for_yasf = config_path
    if args.yasf_no_phase_function or args.yasf_force_cpu:
        cfg = json.loads(config_path.read_text())
        _make_config_paths_absolute(cfg, base_dir=config_path.parent)
        if args.yasf_force_cpu:
            cfg.setdefault("numerics", {})
            cfg["numerics"]["gpu"] = False
        if args.yasf_no_phase_function:
            cfg["optics"] = {
                "enabled": True,
                "cross_sections": True,
                "phase_function": False,
            }

        config_tmp = tempfile.TemporaryDirectory()
        config_for_yasf = Path(config_tmp.name) / config_path.name
        config_for_yasf.write_text(json.dumps(cfg))

    runner.bench_func(
        "yasf_init",
        lambda: YASF(
            str(config_for_yasf),
            path_cluster=args.cluster,
            quiet=bool(args.yasf_quiet),
        ),
    )

    yasf = YASF(
        str(config_for_yasf),
        path_cluster=args.cluster,
        quiet=bool(args.yasf_quiet),
    )
    runner.bench_func("yasf_run", yasf.run)

    mstm = MSTM4Manager(
        path_config=str(config_path),
        path_cluster=args.cluster,
        binary=str(args.mstm_binary),
        parallel=int(args.mstm_parallel),
        workdir=workdir,
        nix=True,
        quiet=bool(args.yasf_quiet),
        random_orientation=False,
        incidence_average=False,
        azimuthal_average=False,
    )

    # MSTM4 timings:
    # - exec_only: includes generating inputs + running binary
    # - parse_only: parse an already-produced output file
    # - run_parse: end-to-end (exec + parse)

    runner.bench_func(
        "mstm4_exec_only",
        lambda: mstm.run(
            runner=None,
            cleanup=True,
            silent=bool(args.mstm_silent),
            parse=False,
        ),
    )

    # Prepare output once; then benchmark parsing only.
    mstm.run(
        runner=None,
        cleanup=False,
        silent=bool(args.mstm_silent),
        parse=False,
    )
    runner.bench_func("mstm4_parse_only", lambda: mstm.read_output())
    mstm.clean()

    runner.bench_func(
        "mstm4_run_parse",
        lambda: mstm.run(
            runner=None,
            cleanup=True,
            silent=bool(args.mstm_silent),
            parse=True,
        ),
    )


if __name__ == "__main__":
    main()
