# Welcome to YASF

`YASF` is a Python implementation of the [`Celes`](https://github.com/disordered-photonics/celes) framework (based on [Matlab](https://matlab.mathworks.com/)) and extends its functionality by providing optical parameters, similar to [`MSTM`](https://github.com/dmckwski/MSTM) (based on [Fortran](https://fortran-lang.org/)).

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