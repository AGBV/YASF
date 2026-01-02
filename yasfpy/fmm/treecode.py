"""Helmholtz treecode / fast multipole prototype.

This module contains an experimental treecode-style accelerator for the
Helmholtz kernel used by YASF. It is primarily intended for research and
benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import time

import numpy as np

try:
    import numba
except ImportError:  # pragma: no cover
    numba = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    prange = range  # type: ignore[assignment]

    def literally(value: int) -> int:  # noqa: D401
        return value

elif numba is not None:
    from numba import literally, prange  # type: ignore[assignment]
else:  # pragma: no cover
    prange = range  # type: ignore[assignment]

    def literally(value: int) -> int:  # noqa: D401
        return value


_SQRT3 = 1.7320508075688772


def _greens_eval(k: complex, r: np.ndarray, r_src: np.ndarray) -> complex:
    """Evaluate the unnormalized outgoing 3D Helmholtz Green's function.

    Parameters
    ----------
    k:
        Wavenumber.
    r:
        Target position ``(3,)``.
    r_src:
        Source position ``(3,)``.

    Returns
    -------
    complex
        ``exp(1j*k*|r-r_src|) / |r-r_src|`` with the singular self-term treated
        as zero.
    """
    d0 = r[0] - r_src[0]
    d1 = r[1] - r_src[1]
    d2 = r[2] - r_src[2]
    dist = float(np.sqrt(d0 * d0 + d1 * d1 + d2 * d2))
    if dist == 0.0:
        return 0.0 + 0.0j
    return np.exp(1j * k * dist) / dist


def helmholtz_greens_function(k: complex, r: np.ndarray, r_src: np.ndarray) -> complex:
    """Scalar 3D Helmholtz Green's function ``G(r, r')``.

    Uses the outgoing convention

    ``G(r, r') = exp(1j * k * d) / d``

    where ``d = ||r - r'||``.

    The prefactor (1/4π) is intentionally omitted because YASF's coupling kernels
    use their own normalization; the treecode is meant as an algorithmic
    feasibility prototype.
    """

    return _greens_eval(k, r, r_src)


def _chebyshev_nodes_1d(order: int) -> np.ndarray:
    """Return first-kind Chebyshev nodes in ``[-1, 1]``.

    Parameters
    ----------
    order:
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Nodes with shape ``(order,)``.
    """
    # Chebyshev nodes of the first kind in [-1, 1]
    # x_j = cos((2j-1)π/(2n)), j=1..n
    j = np.arange(1, order + 1, dtype=float)
    return np.cos((2 * j - 1) * np.pi / (2 * order))


def _barycentric_weights_chebyshev(order: int) -> np.ndarray:
    """Return barycentric interpolation weights for Chebyshev nodes.

    Parameters
    ----------
    order:
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Alternating-sign weights with shape ``(order,)``.
    """
    # For first-kind Chebyshev nodes, barycentric weights are alternating +/- 1.
    w = np.ones(order, dtype=float)
    w[1::2] = -1.0
    return w


def _build_transfer_sign_matrices(
    nodes_1d: np.ndarray, bary_weights_1d: np.ndarray
) -> np.ndarray:
    """1D transfer matrices for child -> parent interpolation.

    Returns an array `T` with shape (2, order, order) where:

    - `T[0]` maps from a child interval at sign -1 (lower half) to parent nodes
    - `T[1]` maps from a child interval at sign +1 (upper half) to parent nodes

    The full 3D transfer for a given octant is separable, i.e. a Kronecker product
    of the corresponding x/y/z 1D matrices.
    """

    order = int(nodes_1d.shape[0])
    out = np.empty((2, order, order), dtype=np.float64)

    # Child nodes mapped into parent reference coordinates.
    for sign_idx, sign in enumerate((-1.0, 1.0)):
        xs = 0.5 * (nodes_1d + sign)
        mat = out[sign_idx]
        for j in range(order):
            basis = _lagrange_basis_1d(float(xs[j]), nodes_1d, bary_weights_1d)
            mat[:, j] = basis

    return out


def _lagrange_basis_1d(x: float, nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Evaluate all Lagrange basis polynomials at x.

    Uses barycentric interpolation weights.
    """

    diff = x - nodes
    hit = np.where(np.abs(diff) < 1e-14)[0]
    if hit.size:
        out = np.zeros(nodes.shape[0], dtype=float)
        out[int(hit[0])] = 1.0
        return out

    tmp = weights / diff
    denom = np.sum(tmp)
    return tmp / denom


def _compute_node_weights_py(
    sources_pos: np.ndarray,
    q: np.ndarray,
    indices: np.ndarray,
    center: np.ndarray,
    halfwidth: float,
    nodes_1d: np.ndarray,
    bary_weights_1d: np.ndarray,
    order: int,
) -> np.ndarray:
    """Compute equivalent-source weights on a Chebyshev grid (Python fallback).

    Parameters
    ----------
    sources_pos:
        Source positions with shape ``(N, 3)``.
    q:
        Source strengths with shape ``(N,)``.
    indices:
        Indices of sources contained in the node.
    center:
        Node center ``(3,)``.
    halfwidth:
        Half-width of the node cube.
    nodes_1d, bary_weights_1d:
        One-dimensional Chebyshev nodes and barycentric weights.
    order:
        Interpolation order per dimension.

    Returns
    -------
    numpy.ndarray
        Flattened weight vector with shape ``(order**3,)``.
    """
    weights = np.zeros(order**3, dtype=np.complex128)

    for src_idx in indices:
        idx = int(src_idx)
        rel = (sources_pos[idx] - center) / halfwidth
        bx = _lagrange_basis_1d(float(rel[0]), nodes_1d, bary_weights_1d)
        by = _lagrange_basis_1d(float(rel[1]), nodes_1d, bary_weights_1d)
        bz = _lagrange_basis_1d(float(rel[2]), nodes_1d, bary_weights_1d)

        charge = q[idx]
        out_idx = 0
        for i in range(order):
            for j in range(order):
                bij = bx[i] * by[j]
                for k in range(order):
                    weights[out_idx] += charge * bij * bz[k]
                    out_idx += 1

    return weights


def _leaf_direct_sum_range_py(
    k: complex,
    r_tgt: np.ndarray,
    sources_pos: np.ndarray,
    q: np.ndarray,
    leaf_indices: np.ndarray,
    start: int,
    count: int,
    source_index: int,
) -> complex:
    """Direct-sum contributions from a subset of leaf sources (Python fallback).

    Parameters
    ----------
    k:
        Wavenumber.
    r_tgt:
        Target position ``(3,)``.
    sources_pos:
        All source positions ``(N, 3)``.
    q:
        Source strengths ``(N,)``.
    leaf_indices:
        Indices of sources contained in the leaf.
    start, count:
        Range of entries in ``leaf_indices`` to accumulate.
    source_index:
        Optional index to skip (set to ``-1`` to disable).

    Returns
    -------
    complex
        Accumulated potential at ``r_tgt``.
    """
    acc = 0.0 + 0.0j
    for t in range(count):
        j = int(leaf_indices[start + t])
        if source_index >= 0 and j == source_index:
            continue
        acc += _greens_eval(k, r_tgt, sources_pos[j]) * q[j]
    return acc


def _equivalent_eval_py(
    k: complex,
    r_tgt: np.ndarray,
    center: np.ndarray,
    halfwidth: float,
    grid: np.ndarray,
    weights: np.ndarray,
) -> complex:
    """Evaluate the potential of a node's equivalent sources (Python fallback).

    Parameters
    ----------
    k:
        Wavenumber.
    r_tgt:
        Target position ``(3,)``.
    center:
        Node center ``(3,)``.
    halfwidth:
        Node half-width.
    grid:
        Reference node grid points with shape ``(order**3, 3)``.
    weights:
        Equivalent-source weights with shape ``(order**3,)``.

    Returns
    -------
    complex
        Potential at ``r_tgt``.
    """
    acc = 0.0 + 0.0j
    for i in range(weights.shape[0]):
        w = weights[i]
        if w != 0:
            src = center + halfwidth * grid[i]
            acc += _greens_eval(k, r_tgt, src) * w
    return acc


_compute_node_weights = _compute_node_weights_py
_leaf_direct_sum_range = _leaf_direct_sum_range_py
_equivalent_eval = _equivalent_eval_py


if numba is not None:
    _jit = numba.jit(cache=True, nogil=True)
    _jit_parallel = numba.jit(cache=True, nogil=True, parallel=True)

    @_jit
    def _greens_eval_numba(k: complex, r: np.ndarray, r_src: np.ndarray) -> complex:
        d0 = r[0] - r_src[0]
        d1 = r[1] - r_src[1]
        d2 = r[2] - r_src[2]
        dist = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if dist == 0.0:
            return 0.0 + 0.0j
        return np.exp(1j * k * dist) / dist

    @_jit
    def _greens_eval_numba_xyz(
        k: complex,
        rx: float,
        ry: float,
        rz: float,
        sx: float,
        sy: float,
        sz: float,
    ) -> complex:
        d0 = rx - sx
        d1 = ry - sy
        d2 = rz - sz
        dist = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        if dist == 0.0:
            return 0.0 + 0.0j
        return np.exp(1j * k * dist) / dist

    @_jit
    def _lagrange_basis_1d_numba(
        x: float, nodes: np.ndarray, bary_weights: np.ndarray
    ) -> np.ndarray:
        n = nodes.shape[0]
        out = np.empty(n, dtype=np.float64)

        for i in range(n):
            if abs(x - nodes[i]) < 1e-14:
                for j in range(n):
                    out[j] = 0.0
                out[i] = 1.0
                return out

        denom = 0.0
        for i in range(n):
            out[i] = bary_weights[i] / (x - nodes[i])
            denom += out[i]

        inv = 1.0 / denom
        for i in range(n):
            out[i] *= inv

        return out

    @_jit
    def _compute_node_weights_numba(
        sources_pos: np.ndarray,
        q: np.ndarray,
        indices: np.ndarray,
        center: np.ndarray,
        halfwidth: float,
        nodes_1d: np.ndarray,
        bary_weights_1d: np.ndarray,
        order: int,
    ) -> np.ndarray:
        weights = np.zeros(order * order * order, dtype=np.complex128)

        for s in range(indices.shape[0]):
            idx = int(indices[s])
            rel0 = (sources_pos[idx, 0] - center[0]) / halfwidth
            rel1 = (sources_pos[idx, 1] - center[1]) / halfwidth
            rel2 = (sources_pos[idx, 2] - center[2]) / halfwidth

            bx = _lagrange_basis_1d_numba(rel0, nodes_1d, bary_weights_1d)
            by = _lagrange_basis_1d_numba(rel1, nodes_1d, bary_weights_1d)
            bz = _lagrange_basis_1d_numba(rel2, nodes_1d, bary_weights_1d)

            charge = q[idx]

            out_idx = 0
            for i in range(order):
                for j in range(order):
                    bij = bx[i] * by[j]
                    for k in range(order):
                        weights[out_idx] += charge * bij * bz[k]
                        out_idx += 1

        return weights

    @_jit_parallel
    def _build_leaf_weights_parallel(
        sources_pos: np.ndarray,
        q: np.ndarray,
        centers: np.ndarray,
        halfwidth: np.ndarray,
        leaf_count: np.ndarray,
        node_src_start: np.ndarray,
        node_src_count: np.ndarray,
        node_src_indices: np.ndarray,
        nodes_1d: np.ndarray,
        bary_weights_1d: np.ndarray,
        order: int,
    ) -> np.ndarray:
        n_nodes = centers.shape[0]
        order3 = order * order * order
        out = np.zeros((n_nodes, order3), dtype=np.complex128)

        for node_id in prange(n_nodes):
            if leaf_count[node_id] <= 0:
                continue

            cx = centers[node_id, 0]
            cy = centers[node_id, 1]
            cz = centers[node_id, 2]
            hw = halfwidth[node_id]

            start = node_src_start[node_id]
            count = node_src_count[node_id]

            weights = out[node_id]

            for s in range(count):
                idx = int(node_src_indices[start + s])

                rel0 = (sources_pos[idx, 0] - cx) / hw
                rel1 = (sources_pos[idx, 1] - cy) / hw
                rel2 = (sources_pos[idx, 2] - cz) / hw

                bx = _lagrange_basis_1d_numba(rel0, nodes_1d, bary_weights_1d)
                by = _lagrange_basis_1d_numba(rel1, nodes_1d, bary_weights_1d)
                bz = _lagrange_basis_1d_numba(rel2, nodes_1d, bary_weights_1d)

                charge = q[idx]

                out_idx = 0
                for i in range(order):
                    for j in range(order):
                        bij = bx[i] * by[j]
                        for k in range(order):
                            weights[out_idx] += charge * bij * bz[k]
                            out_idx += 1

        return out

    @_jit
    def _build_leaf_weights_serial(
        sources_pos: np.ndarray,
        q: np.ndarray,
        centers: np.ndarray,
        halfwidth: np.ndarray,
        leaf_count: np.ndarray,
        node_src_start: np.ndarray,
        node_src_count: np.ndarray,
        node_src_indices: np.ndarray,
        nodes_1d: np.ndarray,
        bary_weights_1d: np.ndarray,
        order: int,
    ) -> np.ndarray:
        n_nodes = centers.shape[0]
        order3 = order * order * order
        out = np.zeros((n_nodes, order3), dtype=np.complex128)

        for node_id in range(n_nodes):
            if leaf_count[node_id] <= 0:
                continue

            cx = centers[node_id, 0]
            cy = centers[node_id, 1]
            cz = centers[node_id, 2]
            hw = halfwidth[node_id]

            start = node_src_start[node_id]
            count = node_src_count[node_id]

            weights = out[node_id]

            for s in range(count):
                idx = int(node_src_indices[start + s])

                rel0 = (sources_pos[idx, 0] - cx) / hw
                rel1 = (sources_pos[idx, 1] - cy) / hw
                rel2 = (sources_pos[idx, 2] - cz) / hw

                bx = _lagrange_basis_1d_numba(rel0, nodes_1d, bary_weights_1d)
                by = _lagrange_basis_1d_numba(rel1, nodes_1d, bary_weights_1d)
                bz = _lagrange_basis_1d_numba(rel2, nodes_1d, bary_weights_1d)

                charge = q[idx]

                out_idx = 0
                for i in range(order):
                    for j in range(order):
                        bij = bx[i] * by[j]
                        for k in range(order):
                            weights[out_idx] += charge * bij * bz[k]
                            out_idx += 1

        return out

    @_jit_parallel
    def _build_internal_node_weights_parallel(
        sources_pos: np.ndarray,
        q: np.ndarray,
        centers: np.ndarray,
        halfwidth: np.ndarray,
        leaf_count: np.ndarray,
        node_src_start: np.ndarray,
        node_src_count: np.ndarray,
        node_src_indices: np.ndarray,
        nodes_1d: np.ndarray,
        bary_weights_1d: np.ndarray,
        order: int,
    ) -> np.ndarray:
        n_nodes = centers.shape[0]
        order3 = order * order * order
        out = np.zeros((n_nodes, order3), dtype=np.complex128)

        for node_id in prange(n_nodes):
            # Only internal nodes are used for far-field approximations.
            if leaf_count[node_id] > 0:
                continue

            cx = centers[node_id, 0]
            cy = centers[node_id, 1]
            cz = centers[node_id, 2]
            hw = halfwidth[node_id]

            start = node_src_start[node_id]
            count = node_src_count[node_id]

            weights = out[node_id]

            for s in range(count):
                idx = int(node_src_indices[start + s])

                rel0 = (sources_pos[idx, 0] - cx) / hw
                rel1 = (sources_pos[idx, 1] - cy) / hw
                rel2 = (sources_pos[idx, 2] - cz) / hw

                bx = _lagrange_basis_1d_numba(rel0, nodes_1d, bary_weights_1d)
                by = _lagrange_basis_1d_numba(rel1, nodes_1d, bary_weights_1d)
                bz = _lagrange_basis_1d_numba(rel2, nodes_1d, bary_weights_1d)

                charge = q[idx]

                out_idx = 0
                for i in range(order):
                    for j in range(order):
                        bij = bx[i] * by[j]
                        for k in range(order):
                            weights[out_idx] += charge * bij * bz[k]
                            out_idx += 1

        return out

    @_jit_parallel
    def _merge_internal_weights_separable_parallel(
        node_ids: np.ndarray,
        child_start: np.ndarray,
        child_count: np.ndarray,
        children: np.ndarray,
        child_octant: np.ndarray,
        transfer_sign_1d: np.ndarray,
        weights: np.ndarray,
        order: int,
    ) -> None:
        order = literally(order)
        order2 = order * order
        order3 = order2 * order

        for t in prange(node_ids.shape[0]):
            node_id = int(node_ids[t])

            out = weights[node_id]
            for p in range(order3):
                out[p] = 0.0 + 0.0j

            # Local temporaries reused for each child.
            tmp_x = np.empty((order, order, order), dtype=np.complex128)
            tmp_xy = np.empty((order, order, order), dtype=np.complex128)

            start = int(child_start[node_id])
            ccount = int(child_count[node_id])
            for c in range(ccount):
                child_id = int(children[start + c])
                octant = int(child_octant[start + c])

                sx = 1 if (octant & 4) else 0
                sy = 1 if (octant & 2) else 0
                sz = 1 if (octant & 1) else 0

                Tx = transfer_sign_1d[sx]
                Ty = transfer_sign_1d[sy]
                Tz = transfer_sign_1d[sz]

                child_w = weights[child_id]

                # X-mode product: tmp_x[i,b,c] = sum_a Tx[i,a] * child[a,b,c]
                for i in range(order):
                    for b in range(order):
                        base = b * order
                        for cc in range(order):
                            acc = 0.0 + 0.0j
                            for a in range(order):
                                acc += Tx[i, a] * child_w[a * order2 + base + cc]
                            tmp_x[i, b, cc] = acc

                # Y-mode product: tmp_xy[i,j,c] = sum_b Ty[j,b] * tmp_x[i,b,c]
                for i in range(order):
                    for j in range(order):
                        for cc in range(order):
                            acc = 0.0 + 0.0j
                            for b in range(order):
                                acc += Ty[j, b] * tmp_x[i, b, cc]
                            tmp_xy[i, j, cc] = acc

                # Z-mode product and accumulate into flat output.
                out_idx = 0
                for i in range(order):
                    for j in range(order):
                        for k in range(order):
                            acc = 0.0 + 0.0j
                            for cc in range(order):
                                acc += Tz[k, cc] * tmp_xy[i, j, cc]
                            out[out_idx] += acc
                            out_idx += 1

    @_jit
    def _merge_internal_weights_separable_serial(
        node_ids: np.ndarray,
        child_start: np.ndarray,
        child_count: np.ndarray,
        children: np.ndarray,
        child_octant: np.ndarray,
        transfer_sign_1d: np.ndarray,
        weights: np.ndarray,
        order: int,
    ) -> None:
        order2 = order * order
        order3 = order2 * order

        tmp_x = np.empty((order, order, order), dtype=np.complex128)
        tmp_xy = np.empty((order, order, order), dtype=np.complex128)

        for t in range(node_ids.shape[0]):
            node_id = int(node_ids[t])

            out = weights[node_id]
            for p in range(order3):
                out[p] = 0.0 + 0.0j

            start = int(child_start[node_id])
            ccount = int(child_count[node_id])
            for c in range(ccount):
                child_id = int(children[start + c])
                octant = int(child_octant[start + c])

                sx = 1 if (octant & 4) else 0
                sy = 1 if (octant & 2) else 0
                sz = 1 if (octant & 1) else 0

                Tx = transfer_sign_1d[sx]
                Ty = transfer_sign_1d[sy]
                Tz = transfer_sign_1d[sz]

                child_w = weights[child_id]

                for i in range(order):
                    for b in range(order):
                        base = b * order
                        for cc in range(order):
                            acc = 0.0 + 0.0j
                            for a in range(order):
                                acc += Tx[i, a] * child_w[a * order2 + base + cc]
                            tmp_x[i, b, cc] = acc

                for i in range(order):
                    for j in range(order):
                        for cc in range(order):
                            acc = 0.0 + 0.0j
                            for b in range(order):
                                acc += Ty[j, b] * tmp_x[i, b, cc]
                            tmp_xy[i, j, cc] = acc

                out_idx = 0
                for i in range(order):
                    for j in range(order):
                        for k in range(order):
                            acc = 0.0 + 0.0j
                            for cc in range(order):
                                acc += Tz[k, cc] * tmp_xy[i, j, cc]
                            out[out_idx] += acc
                            out_idx += 1

    @_jit
    def _leaf_direct_sum_range_numba(
        k: complex,
        r_tgt: np.ndarray,
        sources_pos: np.ndarray,
        q: np.ndarray,
        leaf_indices: np.ndarray,
        start: int,
        count: int,
        source_index: int,
    ) -> complex:
        acc = 0.0 + 0.0j
        for t in range(count):
            j = int(leaf_indices[start + t])
            if source_index >= 0 and j == source_index:
                continue
            acc += _greens_eval_numba(k, r_tgt, sources_pos[j]) * q[j]
        return acc

    @_jit
    def _equivalent_eval_numba(
        k: complex,
        r_tgt: np.ndarray,
        center: np.ndarray,
        halfwidth: float,
        grid: np.ndarray,
        weights: np.ndarray,
    ) -> complex:
        rx = r_tgt[0]
        ry = r_tgt[1]
        rz = r_tgt[2]

        cx = center[0]
        cy = center[1]
        cz = center[2]

        acc = 0.0 + 0.0j
        for i in range(weights.shape[0]):
            w = weights[i]
            if w != 0.0:
                src0 = cx + halfwidth * grid[i, 0]
                src1 = cy + halfwidth * grid[i, 1]
                src2 = cz + halfwidth * grid[i, 2]
                acc += _greens_eval_numba_xyz(k, rx, ry, rz, src0, src1, src2) * w
        return acc

    @_jit_parallel
    def _apply_treecode_parallel(
        k: complex,
        theta: float,
        max_depth: int,
        centers: np.ndarray,
        halfwidth: np.ndarray,
        child_start: np.ndarray,
        child_count: np.ndarray,
        children: np.ndarray,
        leaf_start: np.ndarray,
        leaf_count: np.ndarray,
        leaf_indices: np.ndarray,
        weights: np.ndarray,
        grid: np.ndarray,
        sources_pos: np.ndarray,
        q: np.ndarray,
        targets_pos: np.ndarray,
        self_targets: bool,
    ) -> np.ndarray:
        n_targets = targets_pos.shape[0]

        out = np.zeros(n_targets, dtype=np.complex128)

        for i in prange(n_targets):
            r_tgt = targets_pos[i]
            source_index = i if self_targets else -1

            depth_bound = literally(max_depth) + 1
            stack_node = np.empty(depth_bound, dtype=np.int32)
            stack_next = np.empty(depth_bound, dtype=np.int32)
            sp = 0

            acc = 0.0 + 0.0j

            stack_node[sp] = 0
            stack_next[sp] = 0
            sp += 1

            while sp > 0:
                node_id = stack_node[sp - 1]

                count = leaf_count[node_id]
                if count > 0:
                    acc += _leaf_direct_sum_range_numba(
                        k,
                        r_tgt,
                        sources_pos,
                        q,
                        leaf_indices,
                        int(leaf_start[node_id]),
                        int(count),
                        int(source_index),
                    )
                    sp -= 1
                    continue

                dx = r_tgt[0] - centers[node_id, 0]
                dy = r_tgt[1] - centers[node_id, 1]
                dz = r_tgt[2] - centers[node_id, 2]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                crit = (_SQRT3 * halfwidth[node_id]) / theta

                if dist > crit:
                    acc += _equivalent_eval_numba(
                        k,
                        r_tgt,
                        centers[node_id],
                        halfwidth[node_id],
                        grid,
                        weights[node_id],
                    )
                    sp -= 1
                    continue

                start = child_start[node_id]
                ccount = child_count[node_id]
                next_child = stack_next[sp - 1]

                if next_child >= ccount:
                    sp -= 1
                    continue

                stack_next[sp - 1] = next_child + 1

                child_id = children[start + next_child]
                stack_node[sp] = child_id
                stack_next[sp] = 0
                sp += 1

            out[i] = acc

        return out

    @_jit
    def _apply_treecode_serial(
        k: complex,
        theta: float,
        max_depth: int,
        centers: np.ndarray,
        halfwidth: np.ndarray,
        child_start: np.ndarray,
        child_count: np.ndarray,
        children: np.ndarray,
        leaf_start: np.ndarray,
        leaf_count: np.ndarray,
        leaf_indices: np.ndarray,
        weights: np.ndarray,
        grid: np.ndarray,
        sources_pos: np.ndarray,
        q: np.ndarray,
        targets_pos: np.ndarray,
        self_targets: bool,
    ) -> np.ndarray:
        n_targets = targets_pos.shape[0]
        out = np.zeros(n_targets, dtype=np.complex128)

        depth_bound = max_depth + 1
        stack_node = np.empty(depth_bound, dtype=np.int32)
        stack_next = np.empty(depth_bound, dtype=np.int32)

        for i in range(n_targets):
            r_tgt = targets_pos[i]
            source_index = i if self_targets else -1

            sp = 0
            acc = 0.0 + 0.0j

            stack_node[sp] = 0
            stack_next[sp] = 0
            sp += 1

            while sp > 0:
                node_id = stack_node[sp - 1]

                count = leaf_count[node_id]
                if count > 0:
                    acc += _leaf_direct_sum_range_numba(
                        k,
                        r_tgt,
                        sources_pos,
                        q,
                        leaf_indices,
                        int(leaf_start[node_id]),
                        int(count),
                        int(source_index),
                    )
                    sp -= 1
                    continue

                dx = r_tgt[0] - centers[node_id, 0]
                dy = r_tgt[1] - centers[node_id, 1]
                dz = r_tgt[2] - centers[node_id, 2]
                dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                crit = (_SQRT3 * halfwidth[node_id]) / theta

                if dist > crit:
                    acc += _equivalent_eval_numba(
                        k,
                        r_tgt,
                        centers[node_id],
                        halfwidth[node_id],
                        grid,
                        weights[node_id],
                    )
                    sp -= 1
                    continue

                start = child_start[node_id]
                ccount = child_count[node_id]
                next_child = stack_next[sp - 1]

                if next_child >= ccount:
                    sp -= 1
                    continue

                stack_next[sp - 1] = next_child + 1

                child_id = children[start + next_child]
                stack_node[sp] = child_id
                stack_next[sp] = 0
                sp += 1

            out[i] = acc

        return out

    _compute_node_weights = _compute_node_weights_numba
    _leaf_direct_sum_range = _leaf_direct_sum_range_numba
    _equivalent_eval = _equivalent_eval_numba


@dataclass(slots=True)
class _Node:
    """Internal octree node.

    Parameters
    ----------
    center:
        Node center ``(3,)``.
    halfwidth:
        Half-width of the node's axis-aligned cube.
    indices:
        Source indices contained in this node.
    children:
        Child nodes (empty for leaves).
    child_octants:
        Octant codes (0..7) for each child.
    depth:
        Depth in the tree.

    Attributes
    ----------
    weights:
        Equivalent-source weights on the Chebyshev grid (set for internal nodes).
    """

    center: np.ndarray  # (3,)
    halfwidth: float
    indices: np.ndarray  # source indices under this node
    children: tuple["_Node", ...]
    child_octants: tuple[int, ...]
    depth: int

    weights: np.ndarray | None = None  # (P^3,) equivalent source strengths

    @property
    def radius(self) -> float:
        """Radius of the circumscribed sphere of the node cube."""
        return float(_SQRT3 * self.halfwidth)

    @property
    def is_leaf(self) -> bool:
        """Whether the node has no children."""
        return len(self.children) == 0


class HelmholtzTreecode:
    """Kernel-independent treecode for scalar Helmholtz interactions.

    This is a *feasibility* prototype for eventually accelerating YASF's coupling
    matvecs with hierarchical methods.

    Model
    -----
    Given sources with complex strengths q_j at positions r_j, evaluate

        u(r_i) = sum_{j != i} G(r_i, r_j) q_j

    using a hierarchical tree traversal, approximating well-separated node
    contributions via Chebyshev interpolation inside the source node.

    Parameters
    ----------
    k:
        Complex wave number.
    order:
        Chebyshev interpolation order per dimension. Equivalent nodes per box are
        order^3.
    leaf_size:
        Max sources per leaf.
    theta:
        Multipole acceptance criterion. A node is considered well-separated from
        target r if dist(center, r) > (radius/theta). Larger theta => more
        aggressive approximation.
    """

    def __init__(
        self,
        *,
        k: complex,
        order: int = 4,
        leaf_size: int = 32,
        theta: float = 0.7,
        max_depth: int = 30,
    ) -> None:
        """Initialize the treecode.

        Parameters
        ----------
        k:
            Complex wave number.
        order:
            Chebyshev interpolation order per dimension.
        leaf_size:
            Maximum number of sources per leaf.
        theta:
            Acceptance parameter controlling well-separatedness.
        max_depth:
            Maximum tree depth.
        """
        if order <= 1:
            raise ValueError("order must be >= 2")
        if leaf_size <= 1:
            raise ValueError("leaf_size must be >= 2")
        if theta <= 0:
            raise ValueError("theta must be > 0")

        self.k = k
        self.order = int(order)
        self.leaf_size = int(leaf_size)
        self.theta = float(theta)
        self.max_depth = int(max_depth)

        self._cheb_nodes_1d = _chebyshev_nodes_1d(self.order)
        self._cheb_w_1d = _barycentric_weights_chebyshev(self.order)
        self._transfer_sign_1d = _build_transfer_sign_matrices(
            self._cheb_nodes_1d, self._cheb_w_1d
        )

        grid = np.stack(
            np.meshgrid(
                self._cheb_nodes_1d,
                self._cheb_nodes_1d,
                self._cheb_nodes_1d,
                indexing="ij",
            ),
            axis=-1,
        )
        self._cheb_grid = grid.reshape((-1, 3))

        self._root: _Node | None = None
        self._sources_pos: np.ndarray | None = None

        # Flattened tree representation for fast threaded traversal.
        self._flat_nodes: list[_Node] | None = None
        self._flat_centers: np.ndarray | None = None
        self._flat_halfwidth: np.ndarray | None = None
        self._flat_child_start: np.ndarray | None = None
        self._flat_child_count: np.ndarray | None = None
        self._flat_children: np.ndarray | None = None
        self._flat_leaf_start: np.ndarray | None = None
        self._flat_leaf_count: np.ndarray | None = None
        self._flat_leaf_indices: np.ndarray | None = None
        self._flat_node_src_start: np.ndarray | None = None
        self._flat_node_src_count: np.ndarray | None = None
        self._flat_node_src_indices: np.ndarray | None = None
        self._flat_child_octant: np.ndarray | None = None
        self._flat_depth: np.ndarray | None = None
        self._flat_internal_nodes_by_depth: list[np.ndarray] | None = None
        self._flat_max_depth: int = 0

        self.last_profile: dict[str, float] | None = None

    def build(self, sources_pos: np.ndarray) -> None:
        """Build the tree over the provided sources.

        Parameters
        ----------
        sources_pos:
            Source positions with shape ``(N, 3)``.

        Notes
        -----
        This populates both the recursive node structure and a flattened
        representation used for faster traversal.
        """
        sources_pos = np.asarray(sources_pos, dtype=float)
        if sources_pos.ndim != 2 or sources_pos.shape[1] != 3:
            raise ValueError("sources_pos must have shape (N, 3)")

        self._sources_pos = sources_pos

        mins = np.min(sources_pos, axis=0)
        maxs = np.max(sources_pos, axis=0)
        center = 0.5 * (mins + maxs)
        halfwidth = 0.5 * float(np.max(maxs - mins))
        halfwidth = max(halfwidth, 1e-12)

        indices = np.arange(sources_pos.shape[0], dtype=np.int64)
        self._root = self._build_node(center, halfwidth, indices, depth=0)
        self._flatten_tree()

    def _build_node(
        self, center: np.ndarray, halfwidth: float, indices: np.ndarray, *, depth: int
    ) -> _Node:
        """Recursively build an octree node."""
        if indices.size <= self.leaf_size or depth >= self.max_depth:
            return _Node(
                center=np.asarray(center, dtype=float),
                halfwidth=float(halfwidth),
                indices=indices,
                children=(),
                child_octants=(),
                depth=depth,
            )

        assert self._sources_pos is not None

        pos = self._sources_pos[indices]
        greater = pos >= center[None, :]
        octant = (
            (greater[:, 0].astype(np.int64) << 2)
            | (greater[:, 1].astype(np.int64) << 1)
            | greater[:, 2].astype(np.int64)
        )

        child_half = halfwidth * 0.5
        children: list[_Node] = []
        child_octants: list[int] = []
        offsets = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=float,
        )

        for child_idx in range(8):
            mask = octant == child_idx
            if not np.any(mask):
                continue
            child_center = center + offsets[child_idx] * child_half
            child_indices = indices[mask]
            child_octants.append(child_idx)
            children.append(
                self._build_node(
                    child_center, child_half, child_indices, depth=depth + 1
                )
            )

        return _Node(
            center=np.asarray(center, dtype=float),
            halfwidth=float(halfwidth),
            indices=indices,
            children=tuple(children),
            child_octants=tuple(child_octants),
            depth=depth,
        )

    def _flatten_tree(self) -> None:
        """Create/refresh a flat representation of the current tree."""
        if self._root is None:
            self._flat_nodes = None
            self._flat_centers = None
            self._flat_halfwidth = None
            self._flat_child_start = None
            self._flat_child_count = None
            self._flat_children = None
            self._flat_leaf_start = None
            self._flat_leaf_count = None
            self._flat_leaf_indices = None
            self._flat_node_src_start = None
            self._flat_node_src_count = None
            self._flat_node_src_indices = None
            self._flat_child_octant = None
            self._flat_depth = None
            self._flat_internal_nodes_by_depth = None
            self._flat_max_depth = 0
            return

        flat_nodes: list[_Node] = []
        centers: list[np.ndarray] = []
        halfwidths: list[float] = []
        child_start: list[int] = []
        child_count: list[int] = []
        children: list[int] = []
        child_octant: list[int] = []
        leaf_start: list[int] = []
        leaf_count: list[int] = []
        leaf_indices: list[int] = []

        node_src_start: list[int] = []
        node_src_count: list[int] = []
        node_src_indices: list[int] = []

        depth: list[int] = []

        max_depth = 0

        def visit(node: _Node) -> int:
            nonlocal max_depth

            node_id = len(flat_nodes)
            flat_nodes.append(node)
            centers.append(np.asarray(node.center, dtype=float))
            halfwidths.append(float(node.halfwidth))
            child_start.append(0)
            child_count.append(0)
            leaf_start.append(0)
            leaf_count.append(0)

            node_src_start.append(0)
            node_src_count.append(0)

            depth.append(int(node.depth))

            if node.depth > max_depth:
                max_depth = node.depth

            start = len(node_src_indices)
            node_src_start[node_id] = start
            node_src_count[node_id] = int(node.indices.size)
            for idx in node.indices:
                node_src_indices.append(int(idx))

            if node.is_leaf:
                start = len(leaf_indices)
                leaf_start[node_id] = start
                leaf_count[node_id] = int(node.indices.size)
                for idx in node.indices:
                    leaf_indices.append(int(idx))
                return node_id

            child_ids: list[int] = []
            for ch in node.children:
                child_ids.append(visit(ch))

            start = len(children)
            child_start[node_id] = start
            child_count[node_id] = len(child_ids)
            for i, child_id in enumerate(child_ids):
                children.append(child_id)
                child_octant.append(int(node.child_octants[i]))

            return node_id

        visit(self._root)

        self._flat_nodes = flat_nodes
        self._flat_centers = np.asarray(centers, dtype=float)
        self._flat_halfwidth = np.asarray(halfwidths, dtype=float)
        self._flat_child_start = np.asarray(child_start, dtype=np.int32)
        self._flat_child_count = np.asarray(child_count, dtype=np.int32)
        self._flat_children = np.asarray(children, dtype=np.int32)
        self._flat_child_octant = np.asarray(child_octant, dtype=np.int8)
        self._flat_leaf_start = np.asarray(leaf_start, dtype=np.int32)
        self._flat_leaf_count = np.asarray(leaf_count, dtype=np.int32)
        self._flat_leaf_indices = np.asarray(leaf_indices, dtype=np.int64)
        self._flat_node_src_start = np.asarray(node_src_start, dtype=np.int32)
        self._flat_node_src_count = np.asarray(node_src_count, dtype=np.int32)
        self._flat_node_src_indices = np.asarray(node_src_indices, dtype=np.int64)
        self._flat_depth = np.asarray(depth, dtype=np.int16)
        self._flat_max_depth = int(max_depth)

        internal_by_depth: list[list[int]] = [[] for _ in range(max_depth + 1)]
        for node_id, node in enumerate(flat_nodes):
            if not node.is_leaf:
                internal_by_depth[int(node.depth)].append(int(node_id))

        self._flat_internal_nodes_by_depth = [
            np.asarray(level, dtype=np.int32) for level in internal_by_depth
        ]

    def _ensure_node_expansion(self, node: _Node, q: np.ndarray) -> None:
        """Compute equivalent-source weights for a node.

        The equivalent-source weights depend on `q` and must be recomputed each
        `apply()` call.
        """

        assert self._sources_pos is not None

        node.weights = _compute_node_weights(
            self._sources_pos,
            q,
            node.indices,
            node.center,
            node.halfwidth,
            self._cheb_nodes_1d,
            self._cheb_w_1d,
            self.order,
        )

    def _is_well_separated(self, node: _Node, r_tgt: np.ndarray) -> bool:
        """Return True if a node is acceptable for approximation."""
        d = r_tgt - node.center
        dist = float(np.sqrt(np.dot(d, d)))
        return dist > (node.radius / self.theta)

    def apply(
        self,
        q: np.ndarray,
        *,
        targets_pos: np.ndarray | None = None,
        parallel_mode: str = "auto",
        profile: bool = False,
    ) -> np.ndarray:
        """Apply the treecode to evaluate potentials at targets.

        Parameters
        ----------
        q:
            Source strengths with shape ``(N,)``.
        targets_pos:
            Target positions with shape ``(M, 3)``. If omitted, targets are the
            sources (self-interaction is skipped).
        parallel_mode:
            Selects the main parallel region when Numba is available.
        profile:
            If True, populate ``last_profile`` with timing information.

        Returns
        -------
        numpy.ndarray
            Complex potentials at the targets.
        """
        if self._root is None or self._sources_pos is None:
            raise RuntimeError("Call build(sources_pos) before apply().")

        q = np.asarray(q)
        if q.shape[0] != self._sources_pos.shape[0]:
            raise ValueError("q must have shape (N,)")

        self_targets = targets_pos is None
        if self_targets:
            targets_pos = self._sources_pos
        else:
            targets_pos = np.asarray(targets_pos, dtype=float)
            if targets_pos.ndim != 2 or targets_pos.shape[1] != 3:
                raise ValueError("targets_pos must have shape (M, 3)")

        if parallel_mode not in {"auto", "traverse", "build", "none"}:
            raise ValueError(
                "parallel_mode must be one of: 'auto', 'traverse', 'build', 'none'"
            )

        if profile:
            self.last_profile = {}
            t_apply_start = time.perf_counter()
        else:
            self.last_profile = None

        if numba is not None and self._flat_nodes is not None:
            assert self._flat_centers is not None
            assert self._flat_halfwidth is not None
            assert self._flat_child_start is not None
            assert self._flat_child_count is not None
            assert self._flat_children is not None
            assert self._flat_leaf_start is not None
            assert self._flat_leaf_count is not None
            assert self._flat_leaf_indices is not None
            assert self._flat_node_src_start is not None
            assert self._flat_node_src_count is not None
            assert self._flat_node_src_indices is not None

            assert self._flat_child_octant is not None
            assert self._flat_internal_nodes_by_depth is not None

            transfer_sign_1d = self._transfer_sign_1d

            # Choose a single dominant parallel region to reduce overhead.
            # - 'build': parallelize weights build/merge, serial traversal
            # - 'traverse': serial build/merge, parallel traversal
            # - 'none': serial for both
            # - 'auto': pick a reasonable default
            mode = parallel_mode
            if mode == "auto":
                mode = "traverse"

            if mode in {"build", "traverse", "none"}:
                build_parallel = mode == "build"
                traverse_parallel = mode == "traverse"
            else:  # pragma: no cover
                build_parallel = True
                traverse_parallel = True

            if profile:
                t_build_start = time.perf_counter()

            # Leaf weights from leaf point sources.
            if build_parallel:
                weights = _build_leaf_weights_parallel(
                    self._sources_pos,
                    q,
                    self._flat_centers,
                    self._flat_halfwidth,
                    self._flat_leaf_count,
                    self._flat_node_src_start,
                    self._flat_node_src_count,
                    self._flat_node_src_indices,
                    self._cheb_nodes_1d,
                    self._cheb_w_1d,
                    self.order,
                )
            else:
                weights = _build_leaf_weights_serial(
                    self._sources_pos,
                    q,
                    self._flat_centers,
                    self._flat_halfwidth,
                    self._flat_leaf_count,
                    self._flat_node_src_start,
                    self._flat_node_src_count,
                    self._flat_node_src_indices,
                    self._cheb_nodes_1d,
                    self._cheb_w_1d,
                    self.order,
                )

            if profile:
                assert self.last_profile is not None
                self.last_profile["weights_build_s"] = (
                    time.perf_counter() - t_build_start
                )
                t_merge_start = time.perf_counter()

            # Bottom-up merge of internal nodes (deep -> shallow).
            for d in range(self._flat_max_depth - 1, -1, -1):
                node_ids = self._flat_internal_nodes_by_depth[d]
                if node_ids.size == 0:
                    continue
                if build_parallel:
                    _merge_internal_weights_separable_parallel(
                        node_ids,
                        self._flat_child_start,
                        self._flat_child_count,
                        self._flat_children,
                        self._flat_child_octant,
                        transfer_sign_1d,
                        weights,
                        self.order,
                    )
                else:
                    _merge_internal_weights_separable_serial(
                        node_ids,
                        self._flat_child_start,
                        self._flat_child_count,
                        self._flat_children,
                        self._flat_child_octant,
                        transfer_sign_1d,
                        weights,
                        self.order,
                    )

            if profile:
                assert self.last_profile is not None
                self.last_profile["merge_s"] = time.perf_counter() - t_merge_start
                t_traverse_start = time.perf_counter()

            if traverse_parallel:
                out = _apply_treecode_parallel(
                    self.k,
                    self.theta,
                    self._flat_max_depth,
                    self._flat_centers,
                    self._flat_halfwidth,
                    self._flat_child_start,
                    self._flat_child_count,
                    self._flat_children,
                    self._flat_leaf_start,
                    self._flat_leaf_count,
                    self._flat_leaf_indices,
                    weights,
                    self._cheb_grid,
                    self._sources_pos,
                    q,
                    targets_pos,
                    self_targets,
                )
            else:
                out = _apply_treecode_serial(
                    self.k,
                    self.theta,
                    self._flat_max_depth,
                    self._flat_centers,
                    self._flat_halfwidth,
                    self._flat_child_start,
                    self._flat_child_count,
                    self._flat_children,
                    self._flat_leaf_start,
                    self._flat_leaf_count,
                    self._flat_leaf_indices,
                    weights,
                    self._cheb_grid,
                    self._sources_pos,
                    q,
                    targets_pos,
                    self_targets,
                )

            if profile:
                assert self.last_profile is not None
                self.last_profile["traverse_s"] = time.perf_counter() - t_traverse_start
                self.last_profile["total_s"] = time.perf_counter() - t_apply_start
                self.last_profile["n_sources"] = float(self._sources_pos.shape[0])
                self.last_profile["n_targets"] = float(targets_pos.shape[0])
                self.last_profile["build_parallel"] = 1.0 if build_parallel else 0.0
                self.last_profile["traverse_parallel"] = (
                    1.0 if traverse_parallel else 0.0
                )

            return out

        out = np.zeros(targets_pos.shape[0], dtype=np.complex128)
        for i, r_tgt in enumerate(targets_pos):
            source_index = i if self_targets else None
            out[i] = self._eval_at_target(
                self._root, r_tgt, q, source_index=source_index
            )

        if profile:
            assert self.last_profile is not None
            self.last_profile["total_s"] = time.perf_counter() - t_apply_start
            self.last_profile["n_sources"] = float(self._sources_pos.shape[0])
            self.last_profile["n_targets"] = float(targets_pos.shape[0])
            self.last_profile["build_parallel"] = float("nan")
            self.last_profile["traverse_parallel"] = float("nan")

        return out

    def _eval_at_target(
        self,
        node: _Node,
        r_tgt: np.ndarray,
        q: np.ndarray,
        *,
        source_index: int | None,
    ) -> complex:
        """Evaluate a single target via recursive traversal."""
        assert self._sources_pos is not None

        if node.is_leaf:
            src_idx = int(source_index) if source_index is not None else -1
            return _leaf_direct_sum_range(
                self.k,
                r_tgt,
                self._sources_pos,
                q,
                node.indices,
                0,
                int(node.indices.size),
                src_idx,
            )

        if self._is_well_separated(node, r_tgt):
            self._ensure_node_expansion(node, q)
            assert node.weights is not None
            return _equivalent_eval(
                self.k,
                r_tgt,
                node.center,
                node.halfwidth,
                self._cheb_grid,
                node.weights,
            )

        acc = 0.0 + 0.0j
        for child in node.children:
            acc += self._eval_at_target(child, r_tgt, q, source_index=source_index)
        return acc
