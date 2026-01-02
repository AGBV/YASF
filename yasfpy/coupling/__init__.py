"""Coupling backends and dense-coupling operation factories.

This package contains the implementations used to apply the coupling operator
``W`` between particles in the multiple-scattering linear system.

The public API is centered around backend classes (selected by
``Numerics.coupling_backend``) and the low-level dense matvec operations returned
by :func:`yasfpy.coupling.factory.get_dense_coupling_ops`.

Notes
-----
The overall formulation using VSWF translation operators and a coupling matrix
appears in multiple-sphere scattering literature and in the CELES implementation
context :cite:`Waterman-1971-ID50,Egel-2017-ID1`.
"""

from __future__ import annotations

from yasfpy.coupling.factory import get_dense_coupling_ops
from yasfpy.coupling.backends import (
    CouplingBackend,
    DenseCouplingBackend,
    NearFarCouplingBackend,
    TiledDenseCouplingBackend,
)

__all__ = [
    "CouplingBackend",
    "DenseCouplingBackend",
    "NearFarCouplingBackend",
    "TiledDenseCouplingBackend",
    "get_dense_coupling_ops",
]
