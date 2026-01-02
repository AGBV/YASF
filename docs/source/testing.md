# Testing

YASF uses `pytest`. The test suite is a mix of:

- **Unit tests** for small numerical helpers.
- **Regression tests** against reference datasets (MATLAB/CELES exports).
- **Backend equivalence tests** (dense vs tiled/nearfar coupling backends).
- **Optional integration tests** that require external binaries (e.g. `mstm`).

## Running tests

Run everything:

```sh
uv run pytest
```

Run quietly:

```sh
uv run pytest -q
```

Run a single file or test:

```sh
uv run pytest tests/test_fmm_treecode.py
uv run pytest tests/test_fmm_treecode.py::test_helmholtz_treecode_matches_direct_sum_reasonably
```

Run tests matching a keyword:

```sh
uv run pytest -k treecode
```

## Coverage

If you want a local coverage report:

```sh
uv run pytest --cov
```

## Optional dependencies and skips

Some tests are designed to **auto-skip** when optional tooling is missing.

- **MSTM4 integration**: `tests/test_compare_mstm4.py` skips unless the `mstm` binary
  is present in `PATH`.
- **Reference datasets**: some tests skip if reference `.mat` files are missing.
- **GPU**: many tests can run on CPU; GPU-specific execution is not required to run
  the standard suite, but GPU codepaths are exercised when available.

Tip: to keep benchmarks/tests reproducible, it can help to fix thread-related env vars:

```sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Benchmarks

YASF includes small benchmark drivers in `yasfpy/benchmark/`. These are intended
for **performance regression tracking** and for comparing algorithmic variants
(dense vs tiled/near/far coupling, treecode prototype, external solver
comparisons).

### Reproducibility checklist

Benchmarks are sensitive to background load and thread parallelism. Before
recording results, prefer pinning thread-related env vars:

```sh
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

If you are benchmarking GPU kernels, ensure your CUDA environment is stable and
avoid mixing multiple GPU workloads.

### General workflow (pyperf)

Most benchmark scripts use `pyperf`.

Run a benchmark directly:

```sh
uv run python -m yasfpy.benchmark.bench_coupling_backends
```

List benchmark options/help:

```sh
uv run python -m yasfpy.benchmark.bench_coupling_backends --help
```

Save results to a file:

```sh
uv run python -m yasfpy.benchmark.bench_coupling_backends -o coupling.json
```

Compare two runs:

```sh
uv run python -m pyperf compare_to coupling_old.json coupling_new.json
```

### Coupling backends benchmark

Module: `yasfpy.benchmark.bench_coupling_backends`

Measures the coupling matvec cost for different coupling backends and execution
modes.

Examples:

```sh
# Dense vs tiled_dense; per-wavelength slice matvec benchmark
uv run python -m yasfpy.benchmark.bench_coupling_backends --particles 128 --wavelengths 8 --lmax 1

# Include setup cost (builds simulation + first matvec)
uv run python -m yasfpy.benchmark.bench_coupling_backends --scope setup

# Run dense backend with GPU enabled (skips if CUDA unavailable)
uv run python -m yasfpy.benchmark.bench_coupling_backends --gpu

# Benchmark the near/far prototype by setting a near-field cutoff radius
uv run python -m yasfpy.benchmark.bench_coupling_backends --near-radius 1.0
```

Notes:

- `--mode slice` mimics solver usage (matvec per wavelength).
- `--mode all` benchmarks applying all wavelength channels at once.
- Dense backend memory usage scales as \(O(N^2)\); keep `--particles` modest.

### Treecode / FMM prototype benchmark

Module: `yasfpy.benchmark.bench_fmm_treecode`

Compares the scalar Helmholtz `HelmholtzTreecode` prototype against a direct
\(O(n^2)\) sum.

Examples:

```sh
# Single size
uv run python -m yasfpy.benchmark.bench_fmm_treecode --n 256

# Sweep problem sizes
uv run python -m yasfpy.benchmark.bench_fmm_treecode --sizes "64,128,256,512"

# Include build cost
uv run python -m yasfpy.benchmark.bench_fmm_treecode --scope setup

# Limit to treecode-only timings
uv run python -m yasfpy.benchmark.bench_fmm_treecode --which treecode
```

Threading note:

The treecode can use Numba parallel loops. If you benchmark threaded execution,
keep it explicit and reproducible. The benchmark supports setting Numba threads
inside the worker:

```sh
uv run python -m yasfpy.benchmark.bench_fmm_treecode --numba-threads 1
```

### External solver comparisons (MSTM4 / ADDA)

Some benchmarks/comparisons call external binaries and tend to be more brittle
(they depend on the installed executable, CPU affinity, and filesystem IO).
Prefer running these in an isolated work directory.

- `yasfpy.benchmark.compare_mstm4` (helper used by tests)
- `yasfpy.benchmark.bench_yasf_vs_mstm4`
- `yasfpy.benchmark.adda`

Tip: when a benchmark writes inputs/outputs, run it in a temporary directory
(e.g. `mktemp -d`) to avoid polluting the repo.

## Reference test data

The `tests/data/` directory contains frozen reference inputs/outputs used for
regression testing. Common formats:

- **MATLAB `.mat` files** for numerical reference arrays.
- **FITS files** produced by CELES for cross-framework comparisons.

In general:

- Prefer adding **small**, deterministic reference datasets.
- Avoid regenerating reference data unless you are explicitly updating a numerical
  baseline.

## Test suite map (what to look at)

This section acts as an index for contributors: which tests cover which parts of
YASF, and what failures typically imply.

### `tests/test_comparison.py`

Broad regression coverage for multiple building blocks, largely by comparing
Python results to MATLAB reference outputs in `tests/data/*.mat`.

Key tests:

- `test_multi2single_index`: index mapping (`yasfpy.functions.misc.multi2single_index`).
- `test_legendre_normalized_trigon`: normalized associated Legendre functions
  (`yasfpy.functions.legendre_normalized_trigon.legendre_normalized_trigon`).
- `test_spherical_functions_trigon`: spherical function lookup helpers
  (`yasfpy.functions.spherical_functions_trigon.spherical_functions_trigon`).
- `test_t_entry`: single-particle T-matrix entry (`yasfpy.functions.t_entry.t_entry`).
- `test_translation_table_ab`: translation table generation (`Numerics.compute_translation_table`).
- `test_sph_bessel`: spherical Bessel/Hankel reference comparisons (SciPy special functions).
- `test_dx_xz`: derivative identities for spherical Bessel/Hankel.
- `test_transformation_coefficients`: transformation coefficients
  (`yasfpy.functions.misc.transformation_coefficients`).
- `test_coupling_matrix_multiply`: validates coupling matrix-vector product against a
  MATLAB-exported reference dataset.

If this file fails, it usually indicates a **numerical regression** or a
**convention mismatch** (indexing/layout, degree ordering, normalization).

### `tests/test_full_simulation.py`

End-to-end pipeline checks against `tests/data/full_simulation_data.mat`.

Key tests:

- `test_initial_field_coefficients`: initial-field expansion coefficients.
- `test_mie_coefficients`: single-particle Mie/T-matrix coefficients.
- `test_right_hand_side`: RHS assembly from initial field and Mie coefficients.
- `test_scattered_field_coefficients`: linear solve for scattering coefficients.
- `test_electric_field_scattered`: scattered E-field evaluation at sampling points.
- `test_electric_field_initial`: incident E-field evaluation at sampling points.

If this file fails, it often points to a change in one of the major simulation
stages or in the ordering/layout of coefficient arrays.

### `tests/test_celes.py`

Regression tests against CELES-produced FITS datasets in `tests/data/celes_*.fits`.
This file uses `unittest.TestCase` and runs a small YASF simulation for each FITS
input, comparing computed coefficients to stored results.

Key tests:

- `TestCELES.test_initial_field_coefficients_planewave`
- `TestCELES.test_scattered_field_coefficients`

If this fails, check for:

- Polarization/angle conventions (TE/TM vs perpendicular/parallel).
- Changes in solver tolerances or convergence behavior.

### `tests/test_compare_mstm4.py`

Optional integration test that runs MSTM4 (external binary) and compares against
YASF for a **single sphere**. Skips automatically when `mstm` is unavailable.

Key test:

- `test_compare_single_sphere_qext_qsca_and_phase_and_dolp`: compares efficiencies,
  normalized phase function, and DoLP.

If this fails (when enabled), it often indicates a physics/convention mismatch
rather than a small numeric drift.

### `tests/test_miepython_single_sphere.py`

Compares single-sphere efficiencies (`qext`, `qsca`) against `miepython`.

Key test:

- `test_single_sphere_matches_miepython_efficiencies`

Useful when changing cross-section conventions or medium-index scaling.

### `tests/test_mueller.py`

Validates Mueller-matrix conversion:

- `test_jones_to_mueller_numba_matches_reference_batched`

This primarily checks that the accelerated `numba` implementation agrees with the
pure-numpy reference for batched (and non-contiguous) inputs.

### `tests/test_gpu_env_helpers.py`

Unit tests for parsing environment variables that influence coupling/GPU execution.

Key tests:

- `test_parse_bool_env_defaults`, `test_parse_bool_env_values`
- `test_parse_int_env_defaults_and_minimum`
- `test_normalize_gpu_strategy`
- `test_normalize_multi_wavelength_mode`

If you add new env vars or change accepted values, update this file.

### `tests/test_coupling_backend_selection.py`

Ensures coupling backend selection has the expected side effects (dense lookup
precomputation vs backends that avoid dense tables).

Key tests:

- `test_nearfar_backend_skips_dense_lookups`
- `test_dense_backend_still_builds_lookups`
- `test_tiled_dense_backend_skips_dense_lookups`

### `tests/test_tiled_dense_backend.py`

Backend equivalence tests for the tiled dense coupling backend.

Key tests:

- `test_tiled_dense_matches_dense_matvec`
- `test_tiled_dense_matches_single_wavelength_slice`

### `tests/test_nearfar_backend.py`

Backend equivalence tests for the near/far coupling backend.

Key tests:

- `test_nearfar_large_radius_matches_dense`: near/far should converge to dense.
- `test_nearfar_zero_radius_removes_coupling`: radius=0 should produce zero coupling.
- `test_nearfar_supports_single_wavelength_slice`: validates `idx=` slicing behavior.

### `tests/test_fmm_treecode.py`

Accuracy sanity test for the Helmholtz treecode prototype.

Key test:

- `test_helmholtz_treecode_matches_direct_sum_reasonably`: compares a treecode apply
  against a direct \(O(n^2)\) sum, asserting a bounded relative error.

### `tests/test_wavebundle.py`

Smoke and behavioral tests for the Gaussian-beam (wavebundle) initial field.

Key tests:

- `test_wavebundle_normal_incidence_basic`: coefficients compute, no NaNs, non-zero.
- `test_wavebundle_beam_focusing`: particles near focus get stronger response.
- `test_wavebundle_polarization`: TE and TM produce different coefficients.

### `tests/test_wavebundle_validation.py`

Validation of wavebundle coefficients against CELES/MATLAB reference data.
Skips if `tests/data/wavebundle_test_data.mat` is missing.

Key tests:

- `test_wavebundle_vs_matlab`: full coefficient comparison against reference.

To regenerate the reference `.mat` file, see:

- `tests/matlab/generate_wavebundle_test_data.m`
- `tests/matlab/README_WAVEBUNDLE.md`

### `tests/test_preconditioner.py`

Checks correctness of the block-diagonal preconditioner.

Key tests:

- `test_preconditioner_correctness`: solution equivalence with/without preconditioner.
- `test_preconditioner_vs_matlab`: preconditioned run matches MATLAB reference.

These tests can be relatively slow/noisy; keep problem sizes small when extending.

### `tests/test_url_handlers.py`

Currently a placeholder for URL/material handler tests (mostly commented out).
If you revive or extend URL-based material sources, this is the natural home for
network-free handler unit tests (ideally using local fixtures/mocks).

## Adding new tests (guidelines)

- Prefer **deterministic RNG seeds** (`np.random.default_rng(0)`).
- Keep array sizes small; tests should be fast on CPU.
- Use `pytest.skip(...)` for optional external tools/data.
- When a test depends on numerical tolerances, document the choice of `rtol/atol`
  in the test itself.
