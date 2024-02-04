<p align="center" width="100%">
<img height="400" width="49%" src="docs/assets/img/logo_white.svg#gh-dark-mode-only">
<img height="400" width="49%" src="docs/assets/img/yasf_white.svg#gh-dark-mode-only">
</p>
<p align="center" width="100%">
<img height="400" width="49%" src="docs/assets/img/logo_black.svg#gh-light-mode-only">
<img height="400" width="49%" src="docs/assets/img/yasf_black.svg#gh-light-mode-only">
</p>

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f4f8ef02c45748d9b2b477d7f29d219d)](https://app.codacy.com/gh/AGBV/YASF/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

# Yet Another Scattering Framework
YASF is a T-Matrix implementation in Python based on the Matlab framework [CELES](https://github.com/disordered-photonics/celes) developed by [Egel et al.](https://arxiv.org/abs/1706.02145).

# Install
```sh
pip install yasfpy
```

Sadly [`yasf`](https://pypi.org/project/yasf/) was already taken, so the package is called `yasfpy` for the Python version and can be found on [pypi](https://pypi.org/project/yasfpy/).

# Examples
- Small [dashboard](https://agbv-lpsc2023-arnaut.streamlit.app/) displaying various parameters calculated using YASF

# Development
This repository is still under development!

# Documentation
The code is documented using [MkDocs](https://www.mkdocs.org/). If you discover mistakes, feel free to create a pull request or open up an issue.

# TODO
The [`pywigxjpf`](http://fy.chalmers.se/subatom/wigxjpf/) package is not following PEP 517 and PEP 518 standards, so it may happen, that it won't install properly as a dependency of YASF. Please install it manually if that happens using `pip install pywigxjpf` (before that, run `pip install pycparser` as stated in their [pypi page](https://pypi.org/project/pywigxjpf/)).
One could convert the `setup.py` file to a `pyproject.toml` file. Providing `pycparser` as a dependency could also solve the known issue of having to preinstall it.
