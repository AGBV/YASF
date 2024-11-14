## pip

YASF is available on [pypi](https://pypi.org/project/yasfpy/). To install it, use:

```sh
pip install yasfpy
```

## Nvidia/CUDA & Conda

To run code on the GPU, the [cudetoolkit](https://developer.nvidia.com/cuda-toolkit) is needed. This can be installed using a provided package by nvidia, or by using the conda package as described by [the numba docs](https://numba.pydata.org/numba-doc/dev/cuda/overview.html#software).
An example environment file would be as follows:

```yaml
name: yasf-env
channels:
  - numba
  - conda-forge
  - defaults
variables:
  NUMBA_DISABLE_INTEL_SVML: 0
  NUMBA_CUDA_USE_NVIDIA_BINDING: 0
dependencies:
  - python=3.10
  - cudatoolkit=11.4
  - pip
  - pip:
      - -r requirements.txt
```
