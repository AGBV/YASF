"""Coupling backend implementations.

A coupling backend provides the action of the inter-particle coupling operator
``W`` (sometimes called the coupling matrix) used in the multiple-scattering
linear system.

Backends differ in how they compute/apply that operator:

- dense: precompute all pairwise lookup tables and apply the exact dense matvec
- tiled dense: compute pairwise lookup tables for particle tiles on the fly
- near/far: prototype approximation applying only near-field interactions

References
----------
The VSWF-based translation/coupling approach is described in classic multiple
scattering formulations :cite:`Waterman-1971-ID50` and in CELES
:cite:`Egel-2017-ID1`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from yasfpy.functions.cpu_numba import (
    particle_interaction_sparse,
    particle_interaction_tiled,
)
from yasfpy.functions.misc import mutual_lookup

if TYPE_CHECKING:
    from yasfpy.simulation import Simulation


class CouplingBackend:
    """Base class for coupling backends.

    Attributes
    ----------
    requires_pairwise_lookups:
        Whether the backend requires precomputed pairwise lookup tables stored on
        a :class:`yasfpy.simulation.Simulation` instance.
    """

    requires_pairwise_lookups: bool = False

    def multiply(
        self, x: np.ndarray, idx: int | None = None
    ) -> np.ndarray:  # pragma: no cover
        """Apply the coupling operator ``W``.

        Parameters
        ----------
        x:
            Input vector/array.
        idx:
            Optional wavelength/channel index.

        Returns
        -------
        numpy.ndarray
            ``W @ x``.
        """

        raise NotImplementedError


class DenseCouplingBackend(CouplingBackend):
    """Exact dense coupling backend.

    This backend assumes that pairwise lookup tables have been precomputed and
    stored on the simulation object (``plm``, ``sph_h``, ``e_j_dm_phi``). The
    actual matvec is delegated to a :class:`yasfpy.coupling.ops.DenseCouplingOps`
    instance, which may be CPU or CUDA.

    Notes
    -----
    The data flow is:

    - lookups computed by :func:`yasfpy.lookups.compute_lookups`
    - backend selected by :class:`yasfpy.simulation.Simulation`
    - matvec executed by :func:`yasfpy.coupling.factory.get_dense_coupling_ops`
    """

    requires_pairwise_lookups: bool = True

    def __init__(self, simulation: "Simulation"):
        """Create a dense coupling backend.

        Parameters
        ----------
        simulation:
            Simulation instance providing pairwise lookup tables and numerics.
        """

        self._sim = simulation

    def multiply(self, x: np.ndarray, idx: int | None = None) -> np.ndarray:
        """Apply the dense coupling operator ``W``.

        Parameters
        ----------
        x:
            Input vector or array.
        idx:
            Optional wavelength/channel index.

        Returns
        -------
        numpy.ndarray
            ``W @ x``.
        """

        sim = self._sim

        # Preserve the user-facing error if lookups are missing.
        if sim.plm is None or sim.sph_h is None or sim.e_j_dm_phi is None:
            raise RuntimeError(
                "Dense coupling backend requires pairwise lookup tables but they are missing."
            )

        from yasfpy.coupling import get_dense_coupling_ops

        return get_dense_coupling_ops(sim).matvec(x, idx)


class TiledDenseCouplingBackend(CouplingBackend):
    """Exact dense coupling without O(N^2) lookup tables.

    This backend computes the same interactions as :class:`DenseCouplingBackend`, but
    constructs the :func:`yasfpy.functions.misc.mutual_lookup` quantities in particle
    tiles and immediately applies them to the matvec. This keeps memory usage bounded
    by tile size.

    Notes
    -----
    This backend is currently CPU-only.
    """

    requires_pairwise_lookups: bool = False

    def __init__(self, simulation: "Simulation"):
        """Create a tiled dense coupling backend.

        Parameters
        ----------
        simulation:
            Simulation instance providing particle geometry and numerics.
        """

        self._sim = simulation

    def multiply(self, x: np.ndarray, idx: int | None = None) -> np.ndarray:
        """Apply tiled dense coupling.

        Parameters
        ----------
        x:
            Input vector/array.
        idx:
            Optional wavelength/channel index.

        Returns
        -------
        numpy.ndarray
            ``W @ x``.

        Raises
        ------
        NotImplementedError
            If GPU execution is requested.
        """

        sim = self._sim
        if sim.numerics.gpu:
            raise NotImplementedError(
                "tiled_dense backend is currently CPU-only; "
                "use numerics.coupling_backend='dense' with gpu=true."
            )

        lmax = sim.numerics.lmax
        particle_number = sim.parameters.particles.number
        nmax = 2 * lmax * (lmax + 2)
        jmax = particle_number * nmax

        translation_table = np.ascontiguousarray(sim.numerics.translation_ab5)
        idx_lookup = np.ascontiguousarray(sim.idx_lookup)
        positions = sim.parameters.particles.position

        if idx is None:
            k_medium = sim.parameters.k_medium
        else:
            k_medium = np.ascontiguousarray(sim.parameters.k_medium[idx : idx + 1])

        tile = int(getattr(sim.numerics, "coupling_tile_size", 64))
        if tile <= 0:
            tile = 64

        wx = np.zeros((jmax, k_medium.shape[0]), dtype=np.complex128)

        for s1_start in range(0, particle_number, tile):
            s1_end = min(particle_number, s1_start + tile)
            pos1 = positions[s1_start:s1_end, :]

            for s2_start in range(0, particle_number, tile):
                s2_end = min(particle_number, s2_start + tile)
                pos2 = positions[s2_start:s2_end, :]

                # mutual_lookup returns: sph_j, sph_h, e_j_dm_phi, plm, ...
                _, sph_h_tile, e_phi_tile, plm_tile = mutual_lookup(
                    lmax,
                    pos1,
                    pos2,
                    k_medium,
                )[:4]

                particle_interaction_tiled(
                    lmax,
                    s1_start,
                    s1_end,
                    s2_start,
                    s2_end,
                    idx_lookup,
                    x,
                    translation_table,
                    np.ascontiguousarray(plm_tile),
                    np.ascontiguousarray(sph_h_tile),
                    np.ascontiguousarray(e_phi_tile),
                    wx,
                )

        if idx is not None:
            wx = np.squeeze(wx)

        return wx


class _NearFarTileCacheEntry(dict):
    """Cache entry for a single source tile.

    The entry stores:

    - ``s1_start``, ``s1_end``: source particle index range
    - ``pos1``: source positions (contiguous)
    - ``s2_indices``: indices of all potentially interacting targets
    - ``pos2``: positions for ``s2_indices`` (contiguous)
    - ``near_mask``: boolean mask selecting near pairs within the tile

    Notes
    -----
    This is intentionally a ``dict`` subclass to keep the cache lightweight and
    avoid extra dependencies.
    """

    pass


class NearFarCouplingBackend(CouplingBackend):
    """Approximate near/far coupling backend (prototype).

    The current implementation applies *near-field* interactions exactly (via
    :func:`yasfpy.functions.misc.mutual_lookup`) for particle pairs within a
    configurable cutoff radius and ignores all remaining (far-field)
    interactions.

    Compared to the initial prototype, this backend caches the near-field
    neighbor structure (per source tile) once, because particle geometry is
    fixed for the lifetime of a :class:`yasfpy.simulation.Simulation`.

    Notes
    -----
    - CPU only (GPU uses the dense GPU kernel today).
    - This is an approximation unless the radius includes all pairs.
    """

    requires_pairwise_lookups: bool = False

    def __init__(self, simulation: "Simulation"):
        """Create a near/far coupling backend.

        Parameters
        ----------
        simulation:
            Simulation instance providing particle geometry and numerics.

        Notes
        -----
        This backend builds and caches a per-tile neighbor structure based on
        ``numerics.coupling_near_field_radius`` and ``numerics.coupling_tile_size``.
        """

        self._sim = simulation
        self._nearfar_cache: list[_NearFarTileCacheEntry] | None = None
        self._nearfar_cache_key: tuple[int, float, int] | None = None

    def _build_cache(self) -> None:
        """Build the near-field neighbor cache.

        Notes
        -----
        The cache is organized per source tile (size ``coupling_tile_size``) and
        stores the set of candidate targets that are within the squared cutoff
        radius ``coupling_near_field_radius**2`` of at least one source particle
        in the tile. Self-coupling is explicitly excluded.
        """

        sim = self._sim
        if sim.numerics.gpu:
            raise NotImplementedError(
                "nearfar backend is currently CPU-only; "
                "use numerics.coupling_backend='dense' with gpu=true."
            )

        cutoff = getattr(sim.numerics, "coupling_near_field_radius", None)
        if cutoff is None:
            raise ValueError(
                "nearfar backend requires numerics.coupling_near_field_radius to be set."
            )

        tile = int(getattr(sim.numerics, "coupling_tile_size", 64))
        if tile <= 0:
            tile = 64

        positions = sim.parameters.particles.position
        particle_number = sim.parameters.particles.number
        cutoff2 = float(cutoff) * float(cutoff)

        cache: list[_NearFarTileCacheEntry] = []

        for s1_start in range(0, particle_number, tile):
            s1_end = min(particle_number, s1_start + tile)
            pos1 = np.ascontiguousarray(positions[s1_start:s1_end, :])

            # Compute squared distances from this tile to all particles.
            dxyz = pos1[:, None, :] - positions[None, :, :]
            dist2 = np.sum(dxyz * dxyz, axis=2)
            near = dist2 <= cutoff2

            # Exclude self-coupling explicitly (it is skipped in the kernel too).
            for s1_local, s1 in enumerate(range(s1_start, s1_end)):
                near[s1_local, s1] = False

            s2_indices = np.nonzero(np.any(near, axis=0))[0].astype(np.int64)
            pos2 = np.ascontiguousarray(positions[s2_indices, :])
            near_mask = np.ascontiguousarray(near[:, s2_indices])

            cache.append(
                _NearFarTileCacheEntry(
                    {
                        "s1_start": int(s1_start),
                        "s1_end": int(s1_end),
                        "pos1": pos1,
                        "s2_indices": np.ascontiguousarray(s2_indices),
                        "pos2": pos2,
                        "near_mask": near_mask,
                    }
                )
            )

        self._nearfar_cache = cache
        self._nearfar_cache_key = (particle_number, float(cutoff), int(tile))

    def multiply(self, x: np.ndarray, idx: int | None = None) -> np.ndarray:
        """Apply the near-field-only coupling approximation.

        Parameters
        ----------
        x:
            Input vector/array of multipole coefficients.
        idx:
            Optional wavelength/channel index.

        Returns
        -------
        numpy.ndarray
            Approximate ``W @ x`` where only near-field pairs contribute.

        Raises
        ------
        ValueError
            If ``numerics.coupling_near_field_radius`` is not set.
        NotImplementedError
            If GPU execution is requested.
        """

        sim = self._sim
        if sim.numerics.gpu:
            raise NotImplementedError(
                "nearfar backend is currently CPU-only; "
                "use numerics.coupling_backend='dense' with gpu=true."
            )

        cutoff = getattr(sim.numerics, "coupling_near_field_radius", None)
        if cutoff is None:
            raise ValueError(
                "nearfar backend requires numerics.coupling_near_field_radius to be set."
            )

        tile = int(getattr(sim.numerics, "coupling_tile_size", 64))
        if tile <= 0:
            tile = 64

        particle_number = sim.parameters.particles.number

        cache_key = (particle_number, float(cutoff), int(tile))
        if self._nearfar_cache is None or self._nearfar_cache_key != cache_key:
            self._build_cache()

        if idx is None:
            k_medium = sim.parameters.k_medium
        else:
            k_medium = np.ascontiguousarray(sim.parameters.k_medium[idx : idx + 1])

        lmax = sim.numerics.lmax
        nmax = 2 * lmax * (lmax + 2)
        jmax = particle_number * nmax

        translation_table = np.ascontiguousarray(sim.numerics.translation_ab5)
        idx_lookup = np.ascontiguousarray(sim.idx_lookup)

        wx = np.zeros((jmax, k_medium.shape[0]), dtype=np.complex128)

        assert self._nearfar_cache is not None
        for entry in self._nearfar_cache:
            s2_indices = entry["s2_indices"]
            if s2_indices.size == 0:
                continue

            pos1 = entry["pos1"]
            pos2 = entry["pos2"]
            near_mask = entry["near_mask"]

            _, sph_h_tile, e_phi_tile, plm_tile = mutual_lookup(
                lmax,
                pos1,
                pos2,
                k_medium,
            )[:4]

            particle_interaction_sparse(
                lmax,
                entry["s1_start"],
                entry["s1_end"],
                s2_indices,
                near_mask,
                idx_lookup,
                x,
                translation_table,
                np.ascontiguousarray(plm_tile),
                np.ascontiguousarray(sph_h_tile),
                np.ascontiguousarray(e_phi_tile),
                wx,
            )

        if idx is not None:
            wx = np.squeeze(wx)

        return wx
