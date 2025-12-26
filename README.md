<p align="center" width="100%">
<img height="400" width="49%" src="docs_sphinx/source/_static/logo_white.svg#gh-dark-mode-only">
<img height="400" width="49%" src="docs_sphinx/source/_static/yasf_white.svg#gh-dark-mode-only">
</p>
<p align="center" width="100%">
<img height="400" width="49%" src="docs_sphinx/source/_static/logo_black.svg#gh-light-mode-only">
<img height="400" width="49%" src="docs_sphinx/source/_static/yasf_black.svg#gh-light-mode-only">
</p>

[![PyPI version](https://badge.fury.io/py/yasfpy.svg)](https://badge.fury.io/py/yasfpy)
[![DOI](https://zenodo.org/badge/636196317.svg)](https://zenodo.org/doi/10.5281/zenodo.11193987)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![Unit tests](https://github.com/AGBV/YASF/actions/workflows/testing.yml/badge.svg)](https://github.com/AGBV/YASF/actions/workflows/testing.yml)
![Docs](https://github.com/AGBV/YASF/actions/workflows/docs.yml/badge.svg)
![PYPI](https://github.com/AGBV/YASF/actions/workflows/pypi.yml/badge.svg)
[![codecov](https://codecov.io/gh/AGBV/YASF/graph/badge.svg?token=QUDBKGSDDB)](https://codecov.io/gh/AGBV/YASF)
[![DeepSource](https://app.deepsource.com/gh/AGBV/YASF.svg/?label=code+coverage&show_trend=true&token=qvVGCeQ5niqoLdaj12vk1hIU)](https://app.deepsource.com/gh/AGBV/YASF/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f4f8ef02c45748d9b2b477d7f29d219d)](https://app.codacy.com/gh/AGBV/YASF/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

# Yet Another Scattering Framework

YASF is a T-Matrix implementation in Python based on the Matlab framework [CELES](https://github.com/disordered-photonics/celes) developed by [Egel et al.](https://arxiv.org/abs/1706.02145).

# Install

YASF is published on PyPI as `yasfpy` (sadly `yasf` was already taken).

```sh
uv pip install yasfpy
```

(Alternatively: `pip install yasfpy`.)


## Optional extras

- GPU (CUDA): `uv pip install "yasfpy[cuda]"`
- Interactive exploration (Streamlit dashboard) dependencies: `uv pip install "yasfpy[explore]"`

## GPU (CUDA)

YASF can optionally run on NVIDIA GPUs.

- Install the NVIDIA CUDA toolkit + driver for your system.
- Install the package with the CUDA extras:

```sh
uv pip install "yasfpy[cuda]"
```

Then run your code with `uv run ...` (or inside the same `uv` environment).

Note: If you prefer `pip`, the same extras work: `pip install "yasfpy[cuda]"`.

# Examples

- Small [dashboard](https://agbv-lpsc2023-arnaut.streamlit.app/) displaying various parameters calculated using YASF

# Development

## Quickstart (uv)

```sh
uv sync --group test --group docs
uv run pytest
uv run yasf --help
```

## Documentation

```sh
uv run --group docs sphinx-build -b html docs_sphinx/source docs_sphinx/_build/html -W --keep-going
```

The project documentation is built with Sphinx (needed for BibTeX citations).

# TODO

The [`pywigxjpf`](http://fy.chalmers.se/subatom/wigxjpf/) package is not following PEP 517 and PEP 518 standards, so it may happen, that it won't install properly as a dependency of YASF. Please install it manually if that happens using `pip install pywigxjpf` (before that, run `pip install pycparser` as stated in their [pypi page](https://pypi.org/project/pywigxjpf/)).
One could convert the `setup.py` file to a `pyproject.toml` file. Providing `pycparser` as a dependency could also solve the known issue of having to preinstall it.
