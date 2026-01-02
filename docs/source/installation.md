# Installation

## Install (uv)

YASF is available on PyPI as `yasfpy`.

```sh
uv pip install yasfpy
```

(Alternatively: `pip install yasfpy`.)

## Optional extras

- GPU support (CUDA bindings): `uv pip install "yasfpy[cuda]"`
- Interactive exploration dashboard: `uv pip install "yasfpy[explore]"`

### Refractive-index tables (`refidxdb`)

`refidxdb` is a required dependency and is used to load refractive-index tables
from local CSV files or from
[refractiveindex.info](https://refractiveindex.info) URLs directly in your config.

The `explore` extra adds only the interactive dashboard dependencies:

- **Streamlit / Plotly / PyVista** – power the `yasf explore` dashboard.

## GPU (CUDA)

YASF can optionally run on NVIDIA GPUs.

1. Install the NVIDIA driver and CUDA toolkit for your system.
2. Install the CUDA extra dependencies:

```sh
uv pip install "yasfpy[cuda]"
```

Note: the CUDA extra only pulls the Python-side CUDA bindings; you still need a working NVIDIA driver/CUDA toolkit installation.

## Development

The repository uses `uv` for development and tooling.

```sh
uv sync --group docs
```
