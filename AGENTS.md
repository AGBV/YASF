# AGENTS.md

This file documents repository conventions for humans and coding agents.

## Environment / Running

- Use `uv` for running Python code.
  - Examples: `uv run pytest`, `uv run python -m yasfpy.cli --help`.
- External binaries (e.g. `mstm`) may be present in `PATH`.

## Tests

- Run tests with:
  - `uv run pytest`
- Some tests are optional and auto-skip if the `mstm` binary is unavailable.

## Benchmarks

- Prefer `pyperf` for benchmarking.
- Benchmarks must be reproducible:
  - Set thread-related env vars (example):
    - `OMP_NUM_THREADS=1`
    - `MKL_NUM_THREADS=1`
    - `OPENBLAS_NUM_THREADS=1`
    - `NUMEXPR_NUM_THREADS=1`
  - Avoid benchmarking from the repo root if it causes generated files.
  - MSTM runs must write inputs/outputs to an isolated working directory
    (temporary folder or user-provided `--workdir`).

## Repo Hygiene (Generated Files)

- Do not commit generated outputs from local experiments/benchmarks.
- Keep scratch outputs in dedicated directories (e.g. `.tmp_compare/`).
- `celes/` is considered local-only reference material and should not be committed.
- `examples/` may contain large datasets; avoid adding new large binary artifacts.

## Coding Conventions

- Keep changes minimal and focused on the requested task.
- Prefer `pathlib.Path` for filesystem paths.
- Use Pydantic v2 patterns when touching Pydantic models.
- Avoid adding new dependencies unless strictly needed.
