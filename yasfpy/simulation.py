import yasfpy.log as log
from time import time

import numpy as np
from math import ceil
from numba import cuda
from scipy.sparse.linalg import LinearOperator

# from scipy.spatial.distance import pdist, squareform
from scipy.special import spherical_jn, spherical_yn

# from scipy.special import hankel1
# from scipy.special import lpmv

from yasfpy.parameters import Parameters
from yasfpy.numerics import Numerics
from yasfpy.functions.spherical_functions_trigon import spherical_functions_trigon
from yasfpy.functions.t_entry import t_entry

from yasfpy.functions.misc import transformation_coefficients
from yasfpy.functions.misc import multi2single_index
from yasfpy.functions.misc import mutual_lookup

from yasfpy.functions.cpu_numba import compute_idx_lookups
from yasfpy.functions.cpu_numba import particle_interaction, compute_field
from yasfpy.functions.cuda_numba import particle_interaction_gpu, compute_field_gpu


class Simulation:
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

        self.log = log.scattering_logger(__name__)
        self.__setup()

    def legacy_compute_lookup_particle_distances(self):
        """
        The largest distance between two particles is divided into segments provided by `Numerics.particle_distance_resolution`.
        This array is then used as a lookup for the calculation of the spherical Hankel function.

        Notes
        -----
        This function has been ported from the Matlab Celes framework but is not used by YASF!
        """
        # add two zeros at the beginning to allow interpolation
        # also in the first segment
        step = self.numerics.particle_distance_resolution
        maxdist = (
            self.parameters.particles.max_particle_distance
            + 3 * self.numerics.particle_distance_resolution
        )
        self.lookup_particle_distances = np.concatenate(
            (np.array([0]), np.arange(0, maxdist + np.finfo(float).eps, step))
        )

    def legacy_compute_h3_table(self):
        """
        Computes the spherical hankel function
        at the points calculated in `Simulation.legacy_compute_lookup_particle_distances()`.

        Attributes:
            h3_table (np.ndarray): Lookup table of the spherical hankel function values at `self.lookup_particle_distances`

        Notes:
            This function has been ported from the Matlab Celes framework but is not used by YASF!
        """
        self.h3_table = np.zeros(
            (
                2 * self.numerics.lmax + 1,
                self.lookup_particle_distances.shape[0],
                self.parameters.medium_refractive_index.shape[0],
            ),
            dtype=complex,
        )
        size_param = np.outer(self.lookup_particle_distances, self.parameters.k_medium)

        for p in range(2 * self.numerics.lmax + 1):
            self.h3_table[p, :, :] = spherical_jn(p, size_param) + 1j * spherical_yn(
                p, size_param
            )

    def __compute_idx_lookup(self):
        """
        Creates a lookup table with the indices used in further calculations.
        The lookup table is created using `compute_idx_lookups` function from `yasfpy.functions.cpu_numba`.

        Attributes:
            idx_lookup (np.ndarray): Lookup table of the indices to iterate over large arrays.

        Notes:
            This function utilizes Numba to optimize the computations.
        """
        self.idx_lookup = compute_idx_lookups(
            self.numerics.lmax, self.parameters.particles.number
        )

    def __compute_lookups(self):
        """
        Computes various lookup tables for each particle.

        Attributes:
            sph_j (np.ndarray): Spherical Bessel function lookup table calculated for pair-wise particle distances.
            sph_h (np.ndarray): Spherical Hankel function lookup table calculated for pair-wise particle distances.
            plm (np.ndarray): Associated Legendre polynomial lookup table calculated for the cosine value of the pairwise particle inclination angles.
            e_j_dm_phi (np.ndarray): Exponential function lookup table calculated for the pairwise particle azimuthal angles.

        Notes:
            This function uses numba (https://numba.pydata.org/) under the hood to speed up the computations.
        """
        lookup_computation_time_start = time()
        # TODO: new, could be error prone and is not tested yet!
        self.sph_j, self.sph_h, self.e_j_dm_phi, self.plm = mutual_lookup(
            self.numerics.lmax,
            self.parameters.particles.position,
            self.parameters.particles.position,
            self.parameters.k_medium,
        )[:4]

        # lmax = self.numerics.lmax
        # particle_number = self.parameters.particles.number

        # dists = squareform(pdist(self.parameters.particles.position))
        # ct = np.divide(
        #   np.subtract.outer(
        #     self.parameters.particles.position[:, 2], self.parameters.particles.position[:, 2]),
        #   dists,
        #   out = np.zeros((particle_number, particle_number)),
        #   where = dists != 0)
        # phi = np.arctan2(
        #   np.subtract.outer(
        #     self.parameters.particles.position[:, 1], self.parameters.particles.position[:, 1]),
        #   np.subtract.outer(self.parameters.particles.position[:, 0], self.parameters.particles.position[:, 0]))

        # size_param = np.outer(dists.ravel(), self.parameters.k_medium).reshape(
        #   [particle_number, particle_number, self.parameters.k_medium.shape[0]])

        # self.sph_h = np.zeros((2 * lmax + 1, particle_number, particle_number, self.parameters.k_medium.shape[0]), dtype=complex)
        # self.sph_j = np.zeros_like(self.sph_h)
        # self.e_j_dm_phi = np.zeros((4 * lmax + 1, particle_number, particle_number), dtype=complex)
        # self.plm = np.zeros(((lmax + 1) * (2 * lmax + 1),
        #           particle_number, particle_number))

        # for p in range(2 * lmax + 1):
        #   self.sph_h[p, :, :, :] = np.sqrt(
        #     np.divide(
        #       np.pi / 2,
        #       size_param,
        #       out=np.zeros_like(size_param),
        #       where=size_param != 0)
        #   ) * hankel1(p + 1/2, size_param)
        #   self.sph_j[p, :, :, :] = spherical_jn(p, size_param)
        #   self.e_j_dm_phi[p, :, :] = np.exp(1j * (p - 2 * lmax) * phi)
        #   self.e_j_dm_phi[p + 2 * lmax, :, :] = np.exp(1j * p * phi)
        #   for absdm in range(p + 1):
        #     cml = np.sqrt((2 * p + 1) / 2 /
        #             np.prod(np.arange(p - absdm + 1, p + absdm + 1)))
        #     self.plm[p * (p + 1) // 2 + absdm, :, :] = cml * \
        #       np.power(-1.0, absdm) * lpmv(absdm, p, ct)

        # self.sph_h = np.nan_to_num(
        #   self.sph_h, nan=0) + np.isnan(self.sph_h) * 1

        lookup_computation_time_stop = time()
        self.log.scatter(
            "Computing lookup tables took %f s"
            % (lookup_computation_time_stop - lookup_computation_time_start)
        )

    def __setup(self):
        """
        An internal setup function called upon object creation.
        The following functions are called:

        - [__compute_idx_lookups][simulation.Simulation.__compute_idx_lookup]
        - [__compute_lookups][simulation.Simulation.__compute_lookups]
        """
        self.__compute_idx_lookup()
        self.__compute_lookups()

    def compute_mie_coefficients(self):
        """
        Computes the mie coefficients for the unique pair
        of particle radius and the refractive index of the particle.

        Attributes:
            mie_coefficients (np.ndarray): Mie coefficients table

        See Also:
            [t_entry][functions.t_entry.t_entry] : T-Matrix entry function

        Notes:
            Due to the four nested loops (particles, tau, l, and m),
            it could be rewritten using `numba` to speed the process up.
        """
        self.mie_coefficients = np.zeros(
            (
                self.parameters.particles.num_unique_pairs,
                self.numerics.nmax,
                self.parameters.wavelength.shape[0],
            ),
            dtype=complex,
        )

        self.scatter_to_internal = np.zeros_like(self.mie_coefficients)

        for u_i in range(self.parameters.particles.num_unique_pairs):
            for tau in range(1, 3):
                for l in range(1, self.numerics.lmax + 1):
                    for m in range(-l, l + 1):
                        jmult = multi2single_index(0, tau, l, m, self.numerics.lmax)
                        self.mie_coefficients[u_i, jmult, :] = t_entry(
                            tau=tau,
                            l=l,
                            k_medium=self.parameters.k_medium,
                            k_sphere=self.parameters.omega
                            * self.parameters.particles.unique_radius_index_pairs[
                                u_i, 1:
                            ],
                            radius=np.real(
                                self.parameters.particles.unique_radius_index_pairs[
                                    u_i, 0
                                ]
                            ),
                        )

                        self.scatter_to_internal[u_i, jmult, :] = t_entry(
                            tau=tau,
                            l=l,
                            k_medium=self.parameters.k_medium,
                            k_sphere=self.parameters.omega
                            * self.parameters.particles.unique_radius_index_pairs[
                                u_i, 1:
                            ],
                            radius=np.real(
                                self.parameters.particles.unique_radius_index_pairs[
                                    u_i, 0
                                ]
                            ),
                            field_type="ratio",
                        )

    def compute_initial_field_coefficients(self):
        r"""
        Computes initial field coefficients $a_{\\tau ,l,m}$ and $b_{\\tau ,l,m}$.
        Depending on the `beam_width`, one of two functions is called:

        - [__compute_initial_field_coefficients_wavebundle_normal_incidence][simulation.Simulation.__compute_initial_field_coefficients_wavebundle_normal_incidence], $\\text{beam width} \\in (0, \\infty)$
        - [__compute_initial_field_coefficients_planewave][simulation.Simulation.__compute_initial_field_coefficients_planewave], $\\text{beam width} = 0$ or $\\text{beam width} = \\infty$

        Attributes:
            initial_field_coefficients (np.ndarray): Initial field coefficients
        """
        self.log.scatter("compute initial field coefficients ...")

        if np.isfinite(self.parameters.initial_field.beam_width) and (
            self.parameters.initial_field.beam_width > 0
        ):
            self.log.scatter("\t Gaussian beam ...")
            if self.parameters.initial_field.normal_incidence:
                self.__compute_initial_field_coefficients_wavebundle_normal_incidence()
            else:
                self.log.error("\t this case is not implemented")
        else:
            self.log.scatter("\t plane wave ...")
            self.__compute_initial_field_coefficients_planewave()

        self.log.scatter("done")

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

    def __compute_initial_field_coefficients_planewave(self):
        """The function computes the initial field coefficients for a plane wave based on given parameters
        and spherical coordinates.

        """
        lmax = self.numerics.lmax
        E0 = self.parameters.initial_field.amplitude
        k = self.parameters.k_medium

        beta = self.parameters.initial_field.polar_angle
        cb = np.cos(beta)
        sb = np.sin(beta)
        alpha = self.parameters.initial_field.azimuthal_angle

        # pi and tau symbols for transformation matrix B_dagger
        pilm, taulm = spherical_functions_trigon(lmax, beta)

        # cylindrical coordinates for relative particle positions
        relative_particle_positions = (
            self.parameters.particles.position
            - self.parameters.initial_field.focal_point
        )
        kvec = np.outer(np.array((sb * np.cos(alpha), sb * np.sin(alpha), cb)), k)
        eikr = np.exp(1j * np.matmul(relative_particle_positions, kvec))

        # clean up some memory?
        del (k, beta, cb, sb, kvec, relative_particle_positions)

        self.initial_field_coefficients = np.zeros(
            (
                self.parameters.particles.number,
                self.numerics.nmax,
                self.parameters.k_medium.size,
            ),
            dtype=complex,
        )
        for m in range(-lmax, lmax + 1):
            for tau in range(1, 3):
                for l in range(np.max([1, np.abs(m)]), lmax + 1):
                    n = multi2single_index(0, tau, l, m, lmax)
                    self.initial_field_coefficients[:, n, :] = (
                        4
                        * E0
                        * np.exp(-1j * m * alpha)
                        * eikr
                        * transformation_coefficients(
                            pilm,
                            taulm,
                            tau,
                            l,
                            m,
                            self.parameters.initial_field.pol,
                            dagger=True,
                        )
                    )

    def __compute_initial_field_coefficients_wavebundle_normal_incidence(self):
        """The function initializes the field coefficients for a wave bundle incident at normal incidence.

        TODO:
            Implement this function using the celes function [initial_field_coefficients_wavebundle_normal_incidence.m](https://github.com/disordered-photonics/celes/blob/master/src/initial/initial_field_coefficients_wavebundle_normal_incidence.m)
        """
        self.initial_field_coefficients = (
            np.zeros(
                (
                    self.parameters.particles.number,
                    self.numerics.nmax,
                    self.parameters.k_medium.size,
                ),
                dtype=complex,
            )
            * np.nan
        )

    def coupling_matrix_multiply(self, x: np.ndarray, idx: int = None):
        """Computes the coupling matrix `wx` based on the input parameters.

        Args:
            x (np.ndarray): An input array of shape (n,) or (n, m), where n is the number of particles and m is the number
                of features for each particle. This array represents the input data for which the coupling
                matrix needs to be computed.
            idx (int): An optional integer that specifies the index of a specific spherical harmonic mode. If `idx` is provided,
                the computation will only be performed for that specific mode. If `idx` is not provided or set to `None`,
                the computation will be performed for all spherical harmonic modes.

        Returns:
            wx (np.ndarray): An array of shape (n, m, p), where n is the number of particles, m is the number of features for each
                particle, and p is the number of wavelengths. It represents the coupling matrix `wx`.
        """
        self.log.scatter("prepare particle coupling ... ")
        preparation_time = time()

        lmax = self.numerics.lmax
        particle_number = self.parameters.particles.number
        jmax = particle_number * 2 * lmax * (lmax + 2)
        wavelengths_size = self.parameters.k_medium.shape[0]
        translation_table = self.numerics.translation_ab5
        associated_legendre_lookup = self.plm
        spherical_hankel_lookup = self.sph_h
        e_j_dm_phi_loopup = self.e_j_dm_phi

        idx_lookup = self.idx_lookup

        if idx != None:
            spherical_hankel_lookup = spherical_hankel_lookup[:, :, :, idx]
            spherical_hankel_lookup = np.copy(
                spherical_hankel_lookup[:, :, :, np.newaxis]
            )
            wavelengths_size = 1

        self.log.scatter("\t Starting Wx computation")
        if self.numerics.gpu:
            wx_real = np.zeros(x.shape + (wavelengths_size,), dtype=float)
            wx_imag = np.zeros_like(wx_real)

            idx_device = cuda.to_device(idx_lookup)
            x_device = cuda.to_device(x)
            wx_real_device = cuda.to_device(wx_real)
            wx_imag_device = cuda.to_device(wx_imag)
            translation_device = cuda.to_device(translation_table)
            associated_legendre_device = cuda.to_device(associated_legendre_lookup)
            spherical_hankel_device = cuda.to_device(spherical_hankel_lookup)
            e_j_dm_phi_device = cuda.to_device(e_j_dm_phi_loopup)

            threads_per_block = (16, 16, 2)
            blocks_per_grid_x = ceil(jmax / threads_per_block[0])
            blocks_per_grid_y = ceil(jmax / threads_per_block[1])
            blocks_per_grid_z = ceil(wavelengths_size / threads_per_block[2])
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

            coupling_matrix_time = time()
            particle_interaction_gpu[blocks_per_grid, threads_per_block](
                lmax,
                particle_number,
                idx_device,
                x_device,
                wx_real_device,
                wx_imag_device,
                translation_device,
                associated_legendre_device,
                spherical_hankel_device,
                e_j_dm_phi_device,
            )
            wx_real = wx_real_device.copy_to_host()
            wx_imag = wx_imag_device.copy_to_host()
            wx = wx_real + 1j * wx_imag
            # particle_interaction.parallel_diagnostics(level=4)
            time_end = time()
            self.log.scatter(
                "\t Time taken for preparation: %f"
                % (coupling_matrix_time - preparation_time)
            )
            self.log.scatter(
                "\t Time taken for coupling matrix: %f"
                % (time_end - coupling_matrix_time)
            )
        else:
            # from numba_progress import ProgressBar
            # num_iterations = jmax * jmax * wavelengths
            # progress = ProgressBar(total=num_iterations)
            # progress = None
            wx = particle_interaction(
                lmax,
                particle_number,
                idx_lookup,
                x,
                translation_table,
                associated_legendre_lookup,
                spherical_hankel_lookup,
                e_j_dm_phi_loopup,
            )
            time_end = time()
            self.log.scatter(
                "\t Time taken for coupling matrix: %f" % (time_end - preparation_time)
            )

        if idx != None:
            wx = np.squeeze(wx)

        return wx

    def master_matrix_multiply(self, value: np.ndarray, idx: int):
        """Applies a T-matrix to a given value and returns the result.

        Args:
            value (np.ndarray): The input value for the matrix multiplication operation.
            idx (int): The index of the matrix to be multiplied.

        Returns:
            mx (np.ndarray): The result of the matrix multiplication operation.

        """
        wx = self.coupling_matrix_multiply(value, idx)

        self.log.scatter("apply T-matrix ...")
        t_matrix_start = time()

        twx = (
            self.mie_coefficients[
                self.parameters.particles.single_unique_array_idx, :, idx
            ].ravel(order="C")
            * wx
        )
        mx = value - twx

        t_matrix_stop = time()
        self.log.scatter(f"\t done in {t_matrix_stop - t_matrix_start} seconds.")

        return mx

    def compute_scattered_field_coefficients(self, guess: np.ndarray = None):
        """The function computes the scattered field coefficients using a linear operator and a solver.

        Args:
            guess (np.ndarray): Optional. The initial guess for the solution of the linear system. If no guess is provided,
                the `right_hand_side` variable is used as the initial guess.

        """
        self.log.scatter("compute scattered field coefficients ...")
        jmax = self.parameters.particles.number * self.numerics.nmax
        self.scattered_field_coefficients = np.zeros_like(
            self.initial_field_coefficients
        )
        self.scattered_field_err_codes = np.zeros(self.parameters.wavelengths_number)
        if guess is None:
            guess = self.right_hand_side
        for w in range(self.parameters.wavelengths_number):
            # def mmm(x):
            #     return self.master_matrix_multiply(x, w)

            # A = LinearOperator(shape=(jmax, jmax), matvec=mmm)

            A = LinearOperator(
                shape=(jmax, jmax), matvec=lambda x: self.master_matrix_multiply(x, w)
            )
            b = self.right_hand_side[:, :, w].ravel()
            x0 = guess[:, :, w].ravel()
            self.log.scatter(
                "Solver run %d/%d" % (w + 1, self.parameters.wavelengths_number)
            )
            x, err_code = self.numerics.solver.run(A, b, x0)
            self.scattered_field_coefficients[:, :, w] = x.reshape(
                self.right_hand_side.shape[:2]
            )
            self.scattered_field_err_codes[w] = err_code

    def compute_fields(self, sampling_points: np.ndarray):
        """The function `compute_fields` calculates the field at given sampling points using either CPU or
        GPU computation.

        Args:
            sampling_points (np.ndarray): The numpy array that represents the coordinates of the sampling points.
                It should have a shape of `(n, 3)`, where `n` is the number of sampling points and each row
                represents the `(x, y, z)` coordinates of a point.
        """
        if sampling_points.shape[0] < 1:
            self.log.error("Number of sampling points must be bigger than zero!")
            return
        elif sampling_points.shape[1] != 3:
            self.log.error("The points have to have three coordinates (x,y,z)!")
            return

        # scatter_to_internal_table = np.sum((self.parameters.particles.position[:, np.newaxis, :] - sampling_points[np.newaxis, :, :])**2, axis = 2)
        # scatter_to_internal_table = scatter_to_internal_table < self.parameters.particles.r[:, np.newaxis]**2

        print("Computing mutual lookup")
        (
            _,
            sph_h,
            e_j_dm_phi,
            p_lm,
            e_r,
            e_theta,
            e_phi,
            cosine_theta,
            sine_theta,
            size_parameter,
            sph_h_derivative,
        ) = mutual_lookup(
            self.numerics.lmax,
            self.parameters.particles.position,
            sampling_points,
            self.parameters.k_medium,
            derivatives=True,
            parallel=False,
        )
        pi_lm, tau_lm = spherical_functions_trigon(
            self.numerics.lmax, cosine_theta, sine_theta
        )
        # print(sph_h.size)

        print("Computing field...")
        field_time_start = time()
        self.sampling_points = sampling_points
        if self.numerics.gpu:
            print("\t...using GPU")
            field_real = np.zeros(
                (self.parameters.k_medium.size, sampling_points.shape[0], 3),
                dtype=float,
            )
            field_imag = np.zeros_like(field_real)

            idx_device = cuda.to_device(self.idx_lookup)
            size_parameter_device = cuda.to_device(np.ascontiguousarray(size_parameter))
            sph_h_device = cuda.to_device(np.ascontiguousarray(sph_h))
            sph_h_derivative_device = cuda.to_device(
                np.ascontiguousarray(sph_h_derivative)
            )
            e_j_dm_phi_device = cuda.to_device(np.ascontiguousarray(e_j_dm_phi))
            p_lm_device = cuda.to_device(np.ascontiguousarray(p_lm))
            pi_lm_device = cuda.to_device(np.ascontiguousarray(pi_lm))
            tau_lm_device = cuda.to_device(np.ascontiguousarray(tau_lm))
            e_r_device = cuda.to_device(np.ascontiguousarray(e_r))
            e_theta_device = cuda.to_device(np.ascontiguousarray(e_theta))
            e_phi_device = cuda.to_device(np.ascontiguousarray(e_phi))
            sfc_device = cuda.to_device(
                np.ascontiguousarray(self.scattered_field_coefficients)
            )

            field_real_device = cuda.to_device(field_real)
            field_imag_device = cuda.to_device(field_imag)

            threads_per_block = (16, 16, 2)
            blocks_per_grid = (
                sampling_points.shape[0],
                sph_h.shape[1] * 2 * self.numerics.lmax * (self.numerics.lmax + 2),
                self.parameters.k_medium.size,
            )
            # blocks_per_grid = tuple(
            #     [
            #         ceil(blocks_per_grid[i] / threads_per_block[i])
            #         for i in range(len(threads_per_block))
            #     ]
            # )
            blocks_per_grid = tuple(
                ceil(blocks_per_grid[i] / threads_per_block[i])
                for i in range(len(threads_per_block))
            )

            compute_field_gpu[blocks_per_grid, threads_per_block](
                self.numerics.lmax,
                idx_device,
                size_parameter_device,
                sph_h_device,
                sph_h_derivative_device,
                e_j_dm_phi_device,
                p_lm_device,
                pi_lm_device,
                tau_lm_device,
                e_r_device,
                e_theta_device,
                e_phi_device,
                sfc_device,
                field_real_device,
                field_imag_device,
            )

            field_real = field_real_device.copy_to_host()
            field_imag = field_imag_device.copy_to_host()
            self.scattered_field = field_real + 1j * field_imag

        else:
            print("\t...using CPU")
            self.scattered_field = compute_field(
                self.numerics.lmax,
                self.idx_lookup,
                size_parameter,
                sph_h,
                sph_h_derivative,
                e_j_dm_phi,
                p_lm,
                pi_lm,
                tau_lm,
                e_r,
                e_theta,
                e_phi,
                scattered_field_coefficients=self.scattered_field_coefficients,
            )

        field_time_stop = time()
        self.log.scatter(
            f"\t Time taken for field calculation: {field_time_stop - field_time_start}"
        )
