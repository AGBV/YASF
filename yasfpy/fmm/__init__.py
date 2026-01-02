"""Fast multipole / treecode prototypes.

This package intentionally starts with a conservative, testable building block:

- `yasfpy.fmm.treecode.HelmholtzTreecode`: a kernel-independent treecode for the
  scalar 3D Helmholtz Green's function.

The intent is to prove feasibility (tree construction, well-separated traversal,
accuracy controls) before integrating a true VSWF/T-matrix FMM backend.
"""

from .treecode import HelmholtzTreecode

__all__ = ["HelmholtzTreecode"]
