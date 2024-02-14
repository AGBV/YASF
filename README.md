<p align="center" width="100%">
<img height="400" width="49%" src="docs/assets/img/logo_white.svg#gh-dark-mode-only">
<img height="400" width="49%" src="docs/assets/img/yasf_white.svg#gh-dark-mode-only">
</p>
<p align="center" width="100%">
<img height="400" width="49%" src="docs/assets/img/logo_black.svg#gh-light-mode-only">
<img height="400" width="49%" src="docs/assets/img/yasf_black.svg#gh-light-mode-only">
</p>

[![DeepSource](https://app.deepsource.com/gh/AGBV/YASF.svg/?label=code+coverage&show_trend=true&token=qvVGCeQ5niqoLdaj12vk1hIU)](https://app.deepsource.com/gh/AGBV/YASF/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f4f8ef02c45748d9b2b477d7f29d219d)](https://app.codacy.com/gh/AGBV/YASF/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![Unit tests](https://github.com/AGBV/YASF/actions/workflows/testing.yml/badge.svg)](https://github.com/AGBV/YASF/actions/workflows/testing.yml)
![Docs](https://github.com/AGBV/YASF/actions/workflows/mkdocs.yml/badge.svg)
![PYPI](https://github.com/AGBV/YASF/actions/workflows/pypi.yml/badge.svg)

# Yet Another Scattering Framework
YASF is a T-Matrix implementation in Python based on the Matlab framework [CELES](https://github.com/disordered-photonics/celes) developed by [Egel et al.](https://arxiv.org/abs/1706.02145).

# Install

## pip
```sh
pip install yasfpy
```

Sadly [`yasf`](https://pypi.org/project/yasf/) was already taken, so the package is called `yasfpy` for the Python version and can be found on [pypi](https://pypi.org/project/yasfpy/).

## conda
To run code on the GPU, the [cudetoolkit](https://developer.nvidia.com/cuda-toolkit) is needed. This can be installed using a provided package by nvidia, or by using the conda package as described by [the numba docs](https://numba.pydata.org/numba-doc/dev/cuda/overview.html#software). The repository provides a yaml environment file. To spin an environment up (and update it later on), use:
```sh
conda env create -f yasf-env.yml # install
conda env update -f yasf-env.yml # update the environment (deactivate and activate again for changes to apply)
```

# Examples
- Small [dashboard](https://agbv-lpsc2023-arnaut.streamlit.app/) displaying various parameters calculated using YASF

# Development
This repository is still under development!

# Documentation
The code is documented using [MkDocs](https://www.mkdocs.org/). If you discover mistakes, feel free to create a pull request or open up an issue.

# TODO
The [`pywigxjpf`](http://fy.chalmers.se/subatom/wigxjpf/) package is not following PEP 517 and PEP 518 standards, so it may happen, that it won't install properly as a dependency of YASF. Please install it manually if that happens using `pip install pywigxjpf` (before that, run `pip install pycparser` as stated in their [pypi page](https://pypi.org/project/pywigxjpf/)).
One could convert the `setup.py` file to a `pyproject.toml` file. Providing `pycparser` as a dependency could also solve the known issue of having to preinstall it.
