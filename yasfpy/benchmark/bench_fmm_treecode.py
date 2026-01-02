"""Benchmark driver for the FMM/treecode prototype.

Runs timing comparisons for the Helmholtz treecode implementation.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pyperf

from yasfpy.fmm import HelmholtzTreecode


def _set_reproducible_thread_env() -> None:
    """Set deterministic thread-related environment defaults.

    This helper sets common BLAS/Numba/OpenMP thread-count environment variables
    to ``"1"`` only if they are not already defined in the process environment.

    Notes
    -----
    This is intended for reproducible benchmarking.
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
    """Append benchmark configuration flags for pyperf worker processes.

    Parameters
    ----------
    cmd
        Command list that will be executed by pyperf. This list is mutated
        in-place.
    args
        Parsed CLI arguments.
    """

    if args.sizes is not None:
        cmd.extend(["--sizes", str(args.sizes)])
    else:
        cmd.extend(["--n", str(args.n)])

    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--order", str(args.order)])
    cmd.extend(["--leaf-size", str(args.leaf_size)])
    cmd.extend(["--theta", str(args.theta)])
    cmd.extend(["--which", str(args.which)])
    cmd.extend(["--parallel-mode", str(args.parallel_mode)])

    if args.numba_threads is not None:
        cmd.extend(["--numba-threads", str(args.numba_threads)])

    if args.scope != "steady":
        cmd.extend(["--scope", str(args.scope)])

    if args.log_quiet:
        cmd.append("--log-quiet")


def _build_runner() -> tuple[pyperf.Runner, argparse.ArgumentParser]:
    """Construct the CLI parser and the pyperf runner.

    Returns
    -------
    runner, parser
        The configured pyperf runner and its underlying argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Benchmark scalar Helmholtz treecode vs direct sum",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=256,
        help="Number of sources/targets (ignored if --sizes is used)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated sweep sizes, e.g. '64,128,256'",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--order", type=int, default=4, help="Chebyshev order per box")
    parser.add_argument(
        "--leaf-size",
        type=int,
        default=24,
        dest="leaf_size",
        help="Max sources per leaf",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.6,
        help="Multipole acceptance parameter (bigger => more aggressive)",
    )
    parser.add_argument(
        "--scope",
        choices=("steady", "setup"),
        default="steady",
        help="Benchmark steady-state apply() or build+apply()",
    )
    parser.add_argument(
        "--which",
        choices=("both", "treecode", "direct"),
        default="both",
        help="Which implementation(s) to benchmark",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=("auto", "traverse", "build", "none"),
        default="auto",
        dest="parallel_mode",
        help="Parallelization strategy for treecode.apply()",
    )
    parser.add_argument(
        "--numba-threads",
        type=int,
        default=None,
        dest="numba_threads",
        help="Set Numba thread count inside worker",
    )
    parser.add_argument(
        "--numba-threads-sweep",
        type=str,
        default=None,
        dest="numba_threads_sweep",
        help="Comma-separated Numba thread counts; runs each in a subprocess",
    )
    parser.add_argument(
        "--plot",
        choices=("none", "time", "speedup", "both"),
        default="none",
        help="Plot thread sweep results in terminal (requires plotext)",
    )
    parser.add_argument(
        "--log-quiet",
        dest="log_quiet",
        action="store_true",
        help="Suppress noisy Python-side logging",
    )

    runner = pyperf.Runner(
        _argparser=parser,
        add_cmdline_args=_add_worker_args,
        processes=1,
        warmups=1,
    )
    return runner, parser


def _direct_sum(k: complex, pos: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Evaluate the dense Helmholtz interaction by direct summation.

    Parameters
    ----------
    k
        Complex wavenumber.
    pos
        Positions with shape ``(n, 3)``.
    q
        Complex source strengths with shape ``(n,)``.

    Returns
    -------
    numpy.ndarray
        Complex potential at each position with shape ``(n,)``.

    Notes
    -----
    The diagonal term is skipped (``i == j``).
    """

    n = pos.shape[0]
    out = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        acc = 0.0 + 0.0j
        for j in range(n):
            if i == j:
                continue
            d = pos[i] - pos[j]
            r = float(np.sqrt(np.dot(d, d)))
            acc += np.exp(1j * k * r) / r * q[j]
        out[i] = acc
    return out


def _parse_int_list(spec: str, *, opt_name: str) -> list[int]:
    """Parse a comma-separated list of integers.

    Parameters
    ----------
    spec
        Comma-separated integer specification.
    opt_name
        Name of the CLI option being parsed (used in error messages).

    Returns
    -------
    list[int]
        Parsed integer values.

    Raises
    ------
    ValueError
        If no integers are provided.
    """

    values: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError(f"{opt_name} must contain at least one integer")
    return values


def _parse_sizes(spec: str) -> list[int]:
    """Parse the ``--sizes`` sweep specification."""

    return _parse_int_list(spec, opt_name="--sizes")


def _register_benchmarks_for_size(
    runner: pyperf.Runner,
    *,
    n: int,
    seed: int,
    order: int,
    leaf_size: int,
    theta: float,
    scope: str,
    which: str,
    parallel_mode: str,
    numba_threads: int | None,
) -> None:
    """Register pyperf benchmark functions for a single problem size.

    Parameters
    ----------
    runner
        Pyperf runner used to register timed callables.
    n
        Number of sources/targets.
    seed
        RNG seed used to generate positions and charges.
    order
        Chebyshev order per box for the treecode.
    leaf_size
        Maximum sources per leaf node.
    theta
        Multipole acceptance parameter.
    scope
        Either ``"steady"`` (build once, time apply) or ``"setup"`` (time
        build+apply).
    which
        Which implementation(s) to benchmark.
    parallel_mode
        Parallelization strategy passed to ``HelmholtzTreecode.apply``.
    numba_threads
        If provided, set Numba's thread count before running.
    """
    if numba_threads is not None:
        try:
            import numba

            numba.set_num_threads(int(numba_threads))
        except ImportError:  # pragma: no cover
            pass

    rng = np.random.default_rng(seed)

    pos = rng.normal(size=(n, 3))
    pos *= 0.8

    q = rng.normal(size=n) + 1j * rng.normal(size=n)
    k = 2.5 + 0.0j

    if scope == "setup":

        def _bench_setup_direct() -> np.ndarray:
            return _direct_sum(k, pos, q)

        def _bench_setup_treecode() -> np.ndarray:
            tc = HelmholtzTreecode(k=k, order=order, leaf_size=leaf_size, theta=theta)
            tc.build(pos)
            return tc.apply(q, parallel_mode=parallel_mode)

        if which in {"both", "direct"}:
            runner.bench_func(f"helmholtz_direct_sum_setup_n{n}", _bench_setup_direct)
        if which in {"both", "treecode"}:
            runner.bench_func(f"helmholtz_treecode_setup_n{n}", _bench_setup_treecode)
        return

    tc = HelmholtzTreecode(k=k, order=order, leaf_size=leaf_size, theta=theta)
    tc.build(pos)

    def _bench_direct() -> np.ndarray:
        return _direct_sum(k, pos, q)

    def _bench_treecode() -> np.ndarray:
        return tc.apply(q, parallel_mode=parallel_mode)

    if which in {"both", "direct"}:
        runner.bench_func(f"helmholtz_direct_sum_n{n}", _bench_direct)
    if which in {"both", "treecode"}:
        runner.bench_func(f"helmholtz_treecode_n{n}", _bench_treecode)


def _maybe_plot_threads_sweep(
    *,
    plot: str,
    n: int,
    parallel_mode: str,
    results: list[tuple[int, float]],
) -> None:
    """Optionally plot thread-sweep results in the terminal.

    Parameters
    ----------
    plot
        Plot selection (``"none"``, ``"time"``, ``"speedup"``, or ``"both"``).
    n
        Problem size.
    parallel_mode
        Parallelization strategy used for the treecode.
    results
        List of ``(numba_threads, mean_ms)`` pairs.

    Raises
    ------
    RuntimeError
        If plotting is requested but the optional dependency ``plotext`` is not
        available.
    """

    if plot == "none":
        return

    try:
        import plotext as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "--plot requires the 'plotext' package to be installed"
        ) from exc

    threads = [t for t, _ in results]
    mean_ms = [m for _, m in results]
    baseline = mean_ms[0]
    speedup = [baseline / m for m in mean_ms]

    title = f"Helmholtz treecode (n={n}, parallel_mode={parallel_mode})"

    def _plot_time() -> None:
        plt.clf()
        plt.title(title)
        plt.xlabel("numba_threads")
        plt.ylabel("mean (ms)")
        plt.plot(threads, mean_ms, marker="braille")
        plt.show()

    def _plot_speedup() -> None:
        plt.clf()
        plt.title(title)
        plt.xlabel("numba_threads")
        plt.ylabel("speedup")
        plt.plot(threads, speedup, marker="braille")
        plt.show()

    if plot in {"time", "both"}:
        _plot_time()
    if plot in {"speedup", "both"}:
        _plot_speedup()


def _run_threads_sweep(args: argparse.Namespace, runner: pyperf.Runner) -> None:
    """Run the benchmark in subprocesses for multiple Numba thread counts.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    runner
        Pyperf runner (used for accessing benchmark configuration flags).

    Raises
    ------
    ValueError
        If thread sweep is requested together with ``--sizes``.
    RuntimeError
        If the subprocess output cannot be parsed.
    subprocess.CalledProcessError
        If a subprocess benchmark invocation fails.
    """

    threads_list = _parse_int_list(
        args.numba_threads_sweep, opt_name="--numba-threads-sweep"
    )

    # Only makes sense for a single size at a time.
    if args.sizes is not None:
        raise ValueError("--numba-threads-sweep requires --n (not --sizes)")

    base_cmd = [sys.executable, "-m", "yasfpy.benchmark.bench_fmm_treecode"]
    # Preserve benchmark configuration, but turn off sweep and keep output parseable.
    base_cmd.extend(["--n", str(args.n)])
    base_cmd.extend(["--seed", str(args.seed)])
    base_cmd.extend(["--order", str(args.order)])
    base_cmd.extend(["--leaf-size", str(args.leaf_size)])
    base_cmd.extend(["--theta", str(args.theta)])
    base_cmd.extend(["--scope", str(args.scope)])
    base_cmd.extend(["--which", str(args.which)])
    base_cmd.extend(["--parallel-mode", str(args.parallel_mode)])

    # Keep sweep runs consistent with user-provided pyperf knobs.
    base_cmd.extend(["--values", str(args.values)])
    base_cmd.extend(["--warmups", str(args.warmups)])
    base_cmd.extend(["--min-time", str(args.min_time)])

    base_cmd.append("--log-quiet")

    env = dict(os.environ)

    results: list[tuple[int, float]] = []

    for t in threads_list:
        cmd = base_cmd + ["--numba-threads", str(t)]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        text = proc.stdout + proc.stderr

        # Parse e.g. "helmholtz_treecode_n4096: Mean +- std dev: 131 ms +- 1 ms"
        # or "...: Mean +- std dev: 1.68 sec +- 0.02 sec".
        m = re.search(r"Mean \+\- std dev: ([0-9.]+) (us|ms|sec)", text)
        if not m:
            raise RuntimeError(
                f"Could not parse mean time from output for threads={t}\n{text}"
            )
        mean = float(m.group(1))
        unit = m.group(2)
        if unit == "us":
            mean_ms = mean / 1000.0
        elif unit == "ms":
            mean_ms = mean
        else:  # sec
            mean_ms = mean * 1000.0
        results.append((t, mean_ms))

    baseline = results[0][1]
    print("numba_threads,mean_ms,speedup")
    for t, mean in results:
        print(f"{t},{mean:.3f},{baseline / mean:.3f}")

    _maybe_plot_threads_sweep(
        plot=str(args.plot),
        n=int(args.n),
        parallel_mode=str(args.parallel_mode),
        results=results,
    )


def main() -> None:
    """CLI entry point for the Helmholtz treecode benchmark."""

    _set_reproducible_thread_env()

    runner, _ = _build_runner()
    args = runner.parse_args()

    if args.numba_threads_sweep is not None:
        _run_threads_sweep(args, runner)
        return

    if args.log_quiet:
        import logging

        logging.getLogger().setLevel(logging.ERROR)

    if args.sizes is None:
        sizes = [int(args.n)]
    else:
        sizes = _parse_sizes(args.sizes)

    for n in sizes:
        _register_benchmarks_for_size(
            runner,
            n=n,
            seed=int(args.seed),
            order=int(args.order),
            leaf_size=int(args.leaf_size),
            theta=float(args.theta),
            scope=str(args.scope),
            which=str(args.which),
            parallel_mode=str(args.parallel_mode),
            numba_threads=args.numba_threads,
        )


if __name__ == "__main__":
    main()
