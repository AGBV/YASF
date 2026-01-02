# pyright: reportGeneralTypeIssues=false

"""Simulation orchestration.

This module defines :class:`~yasfpy.simulation.Simulation`, which orchestrates
the major steps of a YASF run: lookup-table generation, Mie/incident-field
coefficient computation, linear-system solve for scattering coefficients, and
optional field evaluation.
"""

import logging

# import yasfpy.log as log
from time import time

import numpy as np
import os

# from scipy.spatial.distance import pdist, squareform
# from scipy.special import hankel1
# from scipy.special import lpmv

from yasfpy.parameters import Parameters
from yasfpy.numerics import Numerics

from yasfpy.coupling.backends import (
    DenseCouplingBackend,
    NearFarCouplingBackend,
    TiledDenseCouplingBackend,
)


class Simulation:
    """Run container for the core YASF simulation pipeline.

    Parameters
    ----------
    parameters
        Physical and material parameters for the run.
    numerics
        Numerical configuration (truncation, sampling, coupling backend, etc.).

    Notes
    -----
    The simulation object owns lookup tables and intermediate arrays that are
    produced during preprocessing and solver stages.
    """

    _matvec_detail_enabled: bool
    _matvec_detail: dict[str, float]

    initial_field_coefficients: np.ndarray
    mie_coefficients: np.ndarray
    scatter_to_internal: np.ndarray
    scattered_field_coefficients: np.ndarray
    scattered_field_err_codes: np.ndarray

    sampling_points: np.ndarray
    initial_field_electric: np.ndarray | None
    initial_field_magnetic: np.ndarray | None
    scattered_field: np.ndarray
    total_field_electric: np.ndarray

    lookup_particle_distances: np.ndarray
    h3_table: np.ndarray
    idx_lookup: np.ndarray
    sph_j: np.ndarray | None
    sph_h: np.ndarray | None
    e_j_dm_phi: np.ndarray | None
    plm: np.ndarray | None

    """This class represents the simulation of YASF (Yet Another Scattering Framework).
    It contains methods for initializing the simulation, computing lookup tables, and calculating mie coefficients.
    """

    def __init__(self, parameters: Parameters, numerics: Numerics):
        """
        Initialize the Simulation object.

        Args:
            parameters (Parameters): The parameters for the simulation.
            numerics (Numerics): The numerics for the simulation.
        """
        self.parameters = parameters
        self.numerics = numerics

        # self.log = log.infoing_logger(__name__)
        self.log = logging.getLogger(self.__class__.__module__)

        self._matvec_detail_enabled = bool(
            int(os.environ.get("YASF_MATVEC_DETAIL", "0"))
        )
        self._matvec_detail = {}

        backend_name = str(getattr(self.numerics, "coupling_backend", "dense")).lower()
        if backend_name in {"dense"}:
            self.coupling_backend = DenseCouplingBackend(self)
        elif backend_name in {"tiled_dense", "tiled-dense", "tileddense"}:
            self.coupling_backend = TiledDenseCouplingBackend(self)
        elif backend_name in {"nearfar", "near_far", "near-far"}:
            self.coupling_backend = NearFarCouplingBackend(self)
        else:
            raise ValueError(
                f"Unsupported coupling backend: {backend_name!r}. "
                "Expected one of {'dense', 'tiled_dense', 'nearfar'}."
            )

        self.__setup()

    def legacy_compute_lookup_particle_distances(self):
        """Compute particle distance lookup for legacy tables.

        Delegates to `yasfpy.lookups.legacy_compute_lookup_particle_distances`.
        """

        from yasfpy.lookups import legacy_compute_lookup_particle_distances

        legacy_compute_lookup_particle_distances(self)

    def legacy_compute_h3_table(self):
        """Compute the legacy spherical Hankel lookup table.

        Delegates to `yasfpy.lookups.legacy_compute_h3_table`.
        """

        from yasfpy.lookups import legacy_compute_h3_table

        legacy_compute_h3_table(self)

    def __compute_idx_lookup(self):
        """Create the index lookup table.

        Delegates to `yasfpy.lookups.compute_idx_lookup`.
        """

        from yasfpy.lookups import compute_idx_lookup

        compute_idx_lookup(self)

    def __compute_lookups(self):
        """Compute pairwise lookup tables.

        Delegates to `yasfpy.lookups.compute_lookups`.
        """

        from yasfpy.lookups import compute_lookups

        compute_lookups(self)

    def __setup(self):
        """
        An internal setup function called upon object creation.
        The following functions are called:

        - [__compute_idx_lookups][simulation.Simulation.__compute_idx_lookup]
        - [__compute_lookups][simulation.Simulation.__compute_lookups]
        """
        self.__compute_idx_lookup()

        if self.coupling_backend.requires_pairwise_lookups:
            self.__compute_lookups()
        else:
            # Keep attributes defined for downstream code.
            self.sph_j = None
            self.sph_h = None
            self.e_j_dm_phi = None
            self.plm = None

    def compute_mie_coefficients(self) -> None:
        """Compute Mie coefficients, stored on the simulation.

        Delegates to `yasfpy.mie_coefficients.compute_mie_coefficients`.
        """

        from yasfpy.mie_coefficients import compute_mie_coefficients

        compute_mie_coefficients(self)

    def compute_initial_field_coefficients(self) -> None:
        r"""Computes initial field coefficients $a_{\tau ,l,m}$ and $b_{\tau ,l,m}$.

        Delegates to `yasfpy.initial_field_coefficients.compute_initial_field_coefficients`.
        """

        from yasfpy.initial_field_coefficients import compute_initial_field_coefficients

        compute_initial_field_coefficients(self)

    def compute_right_hand_side(self):
        r"""
        Computes the right hand side $T \\cdot a_I$ of the equation $M \\cdot b = T \\cdot a_I$.

        Attributes
        ----------
        right_hand_side : np.ndarray
            Right hand side of the equation $M \\cdot b = T \\cdot a_I$

        Notes
        -----
        For more information regarding the equation, please refer to the paper by Celes (https://arxiv.org/abs/1706.02145).
        """
        self.right_hand_side = (
            self.mie_coefficients[self.parameters.particles.single_unique_array_idx, :]
            * self.initial_field_coefficients
        )

    def coupling_matrix_multiply(self, x: np.ndarray, idx: int | None = None):
        """Compute Wx = coupling_matrix @ x.

        Delegates to `yasfpy.matrix_apply.coupling_matrix_multiply`.
        """

        from yasfpy.matrix_apply import coupling_matrix_multiply

        return coupling_matrix_multiply(self, x, idx)

    def master_matrix_multiply(self, value: np.ndarray, idx: int):
        """Apply master operator.

        Delegates to `yasfpy.matrix_apply.master_matrix_multiply`.
        """

        from yasfpy.matrix_apply import master_matrix_multiply

        return master_matrix_multiply(self, value, idx)

    def compute_scattered_field_coefficients(
        self, guess: np.ndarray | None = None
    ) -> None:
        """Compute scattered field coefficients.

        Delegates to `yasfpy.scattered_field_coefficients.compute_scattered_field_coefficients`.
        """

        from yasfpy.scattered_field_coefficients import (
            compute_scattered_field_coefficients,
        )

        compute_scattered_field_coefficients(self, guess)

    def compute_fields(self, sampling_points: np.ndarray) -> None:
        """Compute fields at the given sampling points.

        This method delegates to `yasfpy.fields.compute_fields`.
        """

        from yasfpy.fields import compute_fields

        compute_fields(self, sampling_points)
