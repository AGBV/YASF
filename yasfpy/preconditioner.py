"""
Block-diagonal preconditioner for iterative solvers.

This module implements a block-diagonal preconditioner based on spatial
partitioning of particles, following the CELES approach. The preconditioner
approximates the full system matrix M = I - TW by block-diagonal components,
where each block corresponds to particles within a spatial cuboid.

Key features:
- Spatial partitioning using configurable cuboid sizes
- LU factorization of each block
- Custom triangular solvers (Numba JIT-compiled)
- Significantly reduces iteration count for large systems
"""

import logging
import numpy as np
from numba import jit
from scipy.linalg import lu
from scipy.special import spherical_jn, hankel1
from time import time
from typing import List, Tuple, Optional

from yasfpy.functions.legendre_normalized_trigon import legendre_normalized_trigon
from yasfpy.functions.misc import multi2single_index


def sph_bessel(nu: int, l: int, z: np.ndarray) -> np.ndarray:
    """
    Spherical Bessel or Hankel function.

    Args:
        nu: Type selector (1=Bessel j_l, 3=Hankel h_l^(1))
        l: Order
        z: Argument(s)

    Returns:
        Spherical Bessel/Hankel function values
    """
    if nu == 1:
        # Spherical Bessel function of first kind
        return spherical_jn(l, z)
    elif nu == 3:
        # Spherical Hankel function of first kind: h_l^(1)(z) = j_l(z) + i*y_l(z)
        # Can be computed from cylindrical Hankel: h_l(z) = sqrt(pi/(2z)) * H_{l+1/2}(z)
        # Avoid divide-by-zero for z=0 (not used in particle interactions)
        result = np.zeros_like(z, dtype=complex)
        nonzero = z != 0
        if np.any(nonzero):
            result[nonzero] = np.sqrt(np.pi / (2 * z[nonzero])) * hankel1(
                l + 0.5, z[nonzero]
            )
        return result
    else:
        raise ValueError(f"nu must be 1 or 3, got {nu}")


class BlockDiagonalPreconditioner:
    """
    Block-diagonal preconditioner for electromagnetic scattering problems.

    The preconditioner divides particles into spatial blocks and pre-computes
    LU factorizations of block-diagonal approximations to the system matrix.
    This dramatically improves convergence for large particle systems.

    Attributes:
        partition_edge_sizes: np.ndarray of shape (3,) containing [dx, dy, dz]
            for cuboid dimensions
        partitioning: List of particle index arrays for each block
        partitioning_idcs: List of SVWF coefficient index arrays for each block
        factorized_matrices: List of dicts containing 'L', 'U', 'P' for each block
    """

    def __init__(self, partition_edge_sizes: np.ndarray):
        """
        Initialize block-diagonal preconditioner.

        Args:
            partition_edge_sizes: Array of shape (3,) with [dx, dy, dz] specifying
                                 cuboid dimensions for spatial partitioning
        """
        self.partition_edge_sizes = np.asarray(partition_edge_sizes, dtype=float)
        if self.partition_edge_sizes.shape != (3,):
            raise ValueError("partition_edge_sizes must be array of shape (3,)")

        self.partitioning: Optional[List[np.ndarray]] = None
        self.partitioning_idcs: Optional[List[np.ndarray]] = None
        self.factorized_matrices: Optional[List[dict]] = None

        self.log = logging.getLogger(self.__class__.__module__)

    def prepare(self, simulation):
        """
        Prepare the preconditioner by computing block partitions and LU factorizations.

        This is a one-time setup step that:
        1. Partitions particles into spatial blocks
        2. Builds master matrix M = I - TW for each block
        3. Computes LU factorization of each block

        Args:
            simulation: Simulation object containing parameters, numerics, and tables

        Returns:
            simulation: Updated simulation object (for chaining)
        """
        self.log.info("Preparing block-diagonal preconditioner...")
        start_time = time()

        # Extract simulation parameters
        lmax = simulation.numerics.lmax
        nmax = simulation.numerics.nmax
        k_medium = simulation.parameters.k_medium[0]  # Assume single wavelength
        particle_positions = simulation.parameters.particles.position
        num_particles = particle_positions.shape[0]

        # Step 1: Spatial partitioning
        self.log.info("  Computing spatial partitioning...")
        self.partitioning = make_particle_partition(
            particle_positions, self.partition_edge_sizes
        )
        self.log.info(f"  Created {len(self.partitioning)} blocks")

        # Step 2: Build and factorize each block
        self.partitioning_idcs = []
        self.factorized_matrices = []

        for jp, particle_indices in enumerate(self.partitioning):
            num_block_particles = len(particle_indices)
            self.log.info(
                f"  Block {jp + 1}/{len(self.partitioning)}: "
                f"{num_block_particles} particles"
            )

            # Build SVWF coefficient indices for this block
            block_idcs = []
            for n in range(nmax):
                block_idcs.extend(particle_indices + num_particles * n)
            block_idcs = np.array(block_idcs, dtype=int)
            self.partitioning_idcs.append(block_idcs)

            # Build master matrix M = I - TW for this block
            self.log.info(
                f"    Building master matrix ({num_block_particles * nmax}x{num_block_particles * nmax})..."
            )

            # Get block positions and parameters
            block_positions = particle_positions[particle_indices]
            block_mie_idcs = simulation.parameters.particles.single_unique_array_idx[
                particle_indices
            ]

            # Pre-compute geometry for all pairs
            geometry = compute_pairwise_geometry(block_positions, k_medium, lmax)

            # Build the matrix
            M = build_master_matrix_from_geometry_wrapper(
                geometry,
                simulation.mie_coefficients[
                    block_mie_idcs, :, 0
                ],  # [particles, nmax, wavelengths] -> squeeze wavelength
                simulation.numerics.translation_ab5,
                lmax,
            )

            # LU factorization
            self.log.info(f"    Computing LU factorization...")
            L, U, P = lu_factorize(M)

            self.factorized_matrices.append({"L": L, "U": U, "P": P})

        elapsed = time() - start_time
        self.log.info(f"Preconditioner preparation complete in {elapsed:.2f}s")

        return simulation

    def apply(self, rhs: np.ndarray) -> np.ndarray:
        """
        Apply the preconditioner to a right-hand side vector.

        Solves M^{-1} @ rhs where M is the block-diagonal approximation.
        For each block: solves P @ L @ U @ x = rhs via forward/back substitution.

        Args:
            rhs: Right-hand side vector (complex array)

        Returns:
            result: Preconditioned vector M^{-1} @ rhs
        """
        if self.factorized_matrices is None or self.partitioning_idcs is None:
            raise RuntimeError("Preconditioner not prepared. Call prepare() first.")

        rhs = np.asarray(rhs).ravel()
        result = np.zeros_like(rhs)

        partitioning_idcs = self.partitioning_idcs
        factorized_matrices = self.factorized_matrices

        for jp, block_data in enumerate(factorized_matrices):
            # Extract block RHS
            block_idcs = partitioning_idcs[jp]
            rhs_block = rhs[block_idcs]

            # Apply permutation
            P = block_data["P"]
            rhs_permuted = rhs_block[P]

            # Solve L @ y = rhs_permuted
            L = block_data["L"]
            y = solve_lower_triangular(L, rhs_permuted)

            # Solve U @ x = y
            U = block_data["U"]
            x = solve_upper_triangular(U, y)

            # Store result
            result[block_idcs] = x

        return result


def make_particle_partition(
    positions: np.ndarray, edge_sizes: np.ndarray
) -> List[np.ndarray]:
    """
    Partition particles into spatial cuboids.

    Creates a grid of cuboids with specified edge sizes and assigns each
    particle to a cuboid based on its position.

    Args:
        positions: Array of shape (N, 3) with particle positions [x, y, z]
        edge_sizes: Array of shape (3,) with cuboid dimensions [dx, dy, dz]

    Returns:
        List of arrays, each containing particle indices in one cuboid
    """
    # Create bin edges for each dimension
    x_min, y_min, z_min = positions.min(axis=0) - 1.0
    x_max, y_max, z_max = positions.max(axis=0) + 1.0

    x_edges = np.arange(x_min, x_max + edge_sizes[0], edge_sizes[0])
    y_edges = np.arange(y_min, y_max + edge_sizes[1], edge_sizes[1])
    z_edges = np.arange(z_min, z_max + edge_sizes[2], edge_sizes[2])

    # Digitize particles into bins
    x_bins = np.digitize(positions[:, 0], x_edges)
    y_bins = np.digitize(positions[:, 1], y_edges)
    z_bins = np.digitize(positions[:, 2], z_edges)

    # Find unique combinations and group particles
    partitioning = []
    bin_combinations = {}

    for idx in range(len(positions)):
        bin_key = (x_bins[idx], y_bins[idx], z_bins[idx])
        if bin_key not in bin_combinations:
            bin_combinations[bin_key] = []
        bin_combinations[bin_key].append(idx)

    # Convert to list of arrays
    for indices in bin_combinations.values():
        partitioning.append(np.array(indices, dtype=int))

    return partitioning


def compute_pairwise_geometry(
    positions: np.ndarray, k_medium: float, lmax: int
) -> dict:
    """
    Pre-compute all pairwise geometric quantities for a block of particles.

    This computes distances, angles, Legendre polynomials, and spherical
    Hankel functions for all particle pairs, which are then used to build
    the master matrix efficiently.

    Args:
        positions: Particle positions, shape (N, 3)
        k_medium: Wave number in medium
        lmax: Maximum multipole order

    Returns:
        Dictionary containing:
            - distances: (N, N) distance matrix
            - cos_theta: (N, N) cosine of polar angle
            - sin_theta: (N, N) sine of polar angle
            - phi: (N, N) azimuthal angle
            - legendre: dict mapping (p, abs(m)) -> (N, N) Legendre polynomial values
            - sph_hankel: dict mapping p -> (N, N) spherical Hankel values
    """
    num_particles = positions.shape[0]

    # Compute relative positions using meshgrid
    x1, x2 = np.meshgrid(positions[:, 0], positions[:, 0])
    y1, y2 = np.meshgrid(positions[:, 1], positions[:, 1])
    z1, z2 = np.meshgrid(positions[:, 2], positions[:, 2])

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2

    # Distances and angles
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    cos_theta = np.divide(dz, distances, where=distances > 0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Handle rounding errors
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = np.arctan2(dy, dx)

    # Compute Legendre polynomials for all pairs
    # Reshape for legendre_normalized_trigon which expects 1D arrays
    ct_flat = cos_theta.ravel()
    st_flat = sin_theta.ravel()

    plm_array = legendre_normalized_trigon(2 * lmax, ct_flat, st_flat)

    # Reshape back to (N, N) and store in dict
    # plm_array has shape (2*lmax+1, 2*lmax+1, num_pairs)
    legendre = {}
    for p in range(2 * lmax + 1):
        for m_abs in range(min(p + 1, 2 * lmax + 1)):  # m_abs can't exceed p
            plm_flat = plm_array[p, m_abs, :]  # 0-indexed
            plm_reshaped = plm_flat.reshape(num_particles, num_particles)
            legendre[(p, m_abs)] = plm_reshaped

    # Compute spherical Hankel functions for all pairs and all p
    sph_hankel = {}
    for p in range(2 * lmax + 1):
        kr = k_medium * distances
        h_p = sph_bessel(3, p, kr)  # Type 3 = Hankel function of first kind
        sph_hankel[p] = h_p

    return {
        "distances": distances,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
        "phi": phi,
        "legendre": legendre,
        "sph_hankel": sph_hankel,
    }


@jit(nopython=True, cache=True)
def build_master_matrix_from_geometry(
    geometry_dict: dict,
    mie_coefficients: np.ndarray,
    translation_table: np.ndarray,
    lmax: int,
) -> np.ndarray:
    """Build master matrix ``M = I - T W`` using pre-computed geometry.

    Notes
    -----
    This function is kept as an API placeholder for a future Numba-friendly
    implementation. Numba does not support ordinary Python ``dict`` objects in
    ``nopython`` mode, so the current implementation lives in
    :func:`build_master_matrix_from_geometry_wrapper`.
    """
    raise NotImplementedError(
        "build_master_matrix_from_geometry is not available in nopython mode. "
        "Use build_master_matrix_from_geometry_wrapper instead."
    )


def build_master_matrix_from_geometry_wrapper(
    geometry: dict,
    mie_coefficients: np.ndarray,
    translation_table: np.ndarray,
    lmax: int,
) -> np.ndarray:
    """
    Build the master matrix M = I - TW for a block of particles.

    This wrapper unpacks pre-computed geometry and builds the matrix.
    Not JIT-compiled due to dict handling.

    Args:
        geometry: Pre-computed geometric quantities from compute_pairwise_geometry
        mie_coefficients: Mie coefficients for particles, shape (N_block, nmax)
        translation_table: Translation table ab5
        lmax: Maximum multipole order

    Returns:
        M: Master matrix of shape (N_block * nmax, N_block * nmax)
    """
    num_particles = mie_coefficients.shape[0]
    nmax = 2 * lmax * (lmax + 2)
    size = num_particles * nmax

    # Initialize M = I
    M = np.eye(size, dtype=np.complex64)

    # Extract geometry
    phi = geometry["phi"]
    legendre = geometry["legendre"]
    sph_hankel = geometry["sph_hankel"]

    # Build the matrix (nested loops over particles and modes)
    for s1 in range(num_particles):
        for s2 in range(num_particles):
            if s1 == s2:
                continue  # Skip diagonal (self-interaction)

            # Loop over all mode combinations
            for tau1 in range(1, 3):
                for l1 in range(1, lmax + 1):
                    for m1 in range(-l1, l1 + 1):
                        n1 = multi2single_index(1, tau1, l1, m1, lmax) - 1  # 0-indexed

                        for tau2 in range(1, 3):
                            for l2 in range(1, lmax + 1):
                                for m2 in range(-l2, l2 + 1):
                                    n2 = multi2single_index(1, tau2, l2, m2, lmax) - 1

                                    # Compute TW contribution
                                    dm = abs(m1 - m2)

                                    tw_sum = 0.0 + 0.0j
                                    for p in range(dm, 2 * lmax + 1):
                                        # Get pre-computed values
                                        plm_val = legendre[(p, dm)][s1, s2]
                                        h_p_val = sph_hankel[p][s1, s2]
                                        trans_coeff = translation_table[n2, n1, p]
                                        mie_coeff = mie_coefficients[s2, n1]
                                        exp_factor = np.exp(
                                            1j * (m2 - m1) * phi[s1, s2]
                                        )

                                        tw_sum += (
                                            mie_coeff
                                            * trans_coeff
                                            * plm_val
                                            * h_p_val
                                            * exp_factor
                                        )

                                    # Update matrix element
                                    row = s1 * nmax + n1
                                    col = s2 * nmax + n2
                                    M[row, col] -= tw_sum

    return M


def lu_factorize(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LU factorization with partial pivoting: P @ M = L @ U

    Uses scipy's LU factorization and extracts L, U, and permutation vector P.

    Args:
        M: Square matrix to factorize

    Returns:
        L: Lower triangular matrix with unit diagonal
        U: Upper triangular matrix
        P: Permutation indices (not matrix!) such that M[P, :] = L @ U
    """
    # scipy.linalg.lu with permute_l=False returns (P, L, U) where P @ M = L @ U
    # P is a permutation matrix
    P_mat, L, U = lu(M, permute_l=False)

    # Extract permutation indices from permutation matrix
    # P_mat @ M = L @ U, so we need to find which rows were permuted
    P = np.argmax(P_mat, axis=1).astype(np.int32)

    return L, U, P


@jit(nopython=True, cache=True)
def solve_lower_triangular(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve L @ x = b for lower triangular matrix L.

    Uses forward substitution algorithm. L must have unit diagonal (from LU).

    Args:
        L: Lower triangular matrix (n x n)
        b: Right-hand side vector (n,)

    Returns:
        x: Solution vector (n,)
    """
    n = L.shape[0]
    x = np.empty(n, dtype=b.dtype)

    for i in range(n):
        sum_val = b[i]
        for j in range(i):
            sum_val -= L[i, j] * x[j]
        x[i] = sum_val  # L has unit diagonal

    return x


@jit(nopython=True, cache=True)
def solve_upper_triangular(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve U @ x = b for upper triangular matrix U.

    Uses backward substitution algorithm.

    Args:
        U: Upper triangular matrix (n x n)
        b: Right-hand side vector (n,)

    Returns:
        x: Solution vector (n,)
    """
    n = U.shape[0]
    x = np.empty(n, dtype=b.dtype)

    for i in range(n - 1, -1, -1):
        sum_val = b[i]
        for j in range(i + 1, n):
            sum_val -= U[i, j] * x[j]
        x[i] = sum_val / U[i, i]

    return x
