import logging
from typing import Union
from math import ceil
import pandas as pd
import numpy as np
from numba import cuda
from yasfpy.functions.misc import single_index2multi
from yasfpy.functions.spherical_functions_trigon import spherical_functions_trigon

from yasfpy.simulation import Simulation

from yasfpy.functions.cpu_numba import (
    compute_scattering_cross_section,
    compute_electric_field_angle_components,
    compute_polarization_components,
)
from yasfpy.functions.cuda_numba import (
    compute_scattering_cross_section_gpu,
    compute_electric_field_angle_components_gpu,
    compute_polarization_components_gpu,
)

# from yasfpy.functions.cpu_numba import compute_scattering_cross_section, compute_radial_independent_scattered_field, compute_electric_field_angle_components, compute_polarization_components
# from yasfpy.functions.cuda_numba import compute_scattering_cross_section_gpu, compute_radial_independent_scattered_field_gpu, compute_electric_field_angle_components_gpu, compute_polarization_components_gpu


class Optics:
    """The optics class handles the calculation of the optical scattering properties
    like scattering and extinction cross-sections.
    """

    def __init__(self, simulation: Simulation):
        """
        Initializes the Optics object.

        Args:
            simulation (Simulation): An instance of the Simulation class.

        """
        self.simulation = simulation

        self.c_ext = np.zeros_like(simulation.parameters.wavelength, dtype=complex)
        self.c_sca = np.zeros_like(simulation.parameters.wavelength, dtype=complex)

        self.log = logging.getLogger(__name__)

    def compute_cross_sections(self):
        """
        Compute the scattering and extinction cross sections.

        This method calculates the scattering and extinction cross sections for the simulation.
        It uses the initial field coefficients and scattered field coefficients to perform the calculations.
        """
        a = self.simulation.initial_field_coefficients
        f = self.simulation.scattered_field_coefficients

        self.c_ext = np.zeros(
            self.simulation.parameters.wavelengths_number, dtype=complex
        )
        self.c_ext -= (
            np.sum(np.conjugate(a) * f, axis=(0, 1))
            / np.power(self.simulation.parameters.k_medium, 2)
            * np.pi
        )

        lmax = self.simulation.numerics.lmax
        particle_number = self.simulation.parameters.particles.number
        wavelengths = self.simulation.parameters.k_medium.shape[0]
        translation_table = self.simulation.numerics.translation_ab5
        associated_legendre_lookup = self.simulation.plm
        spherical_bessel_lookup = self.simulation.sph_j
        e_j_dm_phi_loopup = self.simulation.e_j_dm_phi

        idx_lookup = self.simulation.idx_lookup

        if self.simulation.numerics.gpu:
            c_sca_real = np.zeros(wavelengths, dtype=float)
            c_sca_imag = np.zeros_like(c_sca_real)

            idx_device = cuda.to_device(idx_lookup)
            sfc_device = cuda.to_device(f)
            c_sca_real_device = cuda.to_device(c_sca_real)
            c_sca_imag_device = cuda.to_device(c_sca_imag)
            translation_device = cuda.to_device(translation_table)
            associated_legendre_device = cuda.to_device(associated_legendre_lookup)
            spherical_bessel_device = cuda.to_device(spherical_bessel_lookup)
            e_j_dm_phi_device = cuda.to_device(e_j_dm_phi_loopup)

            jmax = particle_number * 2 * lmax * (lmax + 2)
            threads_per_block = (16, 16, 2)
            blocks_per_grid_x = ceil(jmax / threads_per_block[0])
            blocks_per_grid_y = ceil(jmax / threads_per_block[1])
            blocks_per_grid_z = ceil(wavelengths / threads_per_block[2])
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

            compute_scattering_cross_section_gpu[blocks_per_grid, threads_per_block](
                lmax,
                particle_number,
                idx_device,
                sfc_device,
                translation_device,
                associated_legendre_device,
                spherical_bessel_device,
                e_j_dm_phi_device,
                c_sca_real_device,
                c_sca_imag_device,
            )
            c_sca_real = c_sca_real_device.copy_to_host()
            c_sca_imag = c_sca_imag_device.copy_to_host()
            c_sca = c_sca_real + 1j * c_sca_imag

        else:
            # from numba_progress import ProgressBar
            # num_iterations = jmax * jmax * wavelengths
            # progress = ProgressBar(total=num_iterations)
            progress = None
            c_sca = compute_scattering_cross_section(
                lmax,
                particle_number,
                idx_lookup,
                f,
                translation_table,
                associated_legendre_lookup,
                spherical_bessel_lookup,
                e_j_dm_phi_loopup,
            )

        self.c_sca = (
            c_sca / np.power(np.abs(self.simulation.parameters.k_medium), 2) * np.pi
        )

        self.c_ext = np.real(self.c_ext)
        self.c_sca = np.real(self.c_sca)

        self.albedo = self.c_sca / self.c_ext

    def compute_phase_funcition(
        self,
        legendre_coefficients_number: int = 15,
        c_and_b: Union[bool, tuple] = False,
        check_phase_function: bool = False,
    ):
        """The function `compute_phase_function` calculates various polarization components and phase
        function coefficients for a given simulation.

        Args:
            legendre_coefficients_number (int, optional): The number of Legendre coefficients to compute
                for the phase function. These coefficients are used to approximate the phase function
                using Legendre polynomials. The higher the number of coefficients, the more accurate
                the approximation will be.
            c_and_b (Union[bool, tuple], optional): A boolean value or a tuple. If `True`, it indicates
                that the `c` and `b` bounds should be computed. If `False`, the function will return
                without computing the `c` and `b` bounds.
        """
        pilm, taulm = spherical_functions_trigon(
            self.simulation.numerics.lmax, self.simulation.numerics.polar_angles
        )

        if self.simulation.numerics.gpu:
            jmax = (
                self.simulation.parameters.particles.number
                * self.simulation.numerics.nmax
            )
            angles = self.simulation.numerics.azimuthal_angles.size
            wavelengths = self.simulation.parameters.k_medium.size
            e_field_theta_real = np.zeros(
                (
                    self.simulation.numerics.azimuthal_angles.size,
                    self.simulation.parameters.k_medium.size,
                ),
                dtype=float,
            )
            e_field_theta_imag = np.zeros_like(e_field_theta_real)
            e_field_phi_real = np.zeros_like(e_field_theta_real)
            e_field_phi_imag = np.zeros_like(e_field_theta_real)

            particles_position_device = cuda.to_device(
                self.simulation.parameters.particles.position
            )
            idx_device = cuda.to_device(self.simulation.idx_lookup)
            sfc_device = cuda.to_device(self.simulation.scattered_field_coefficients)
            k_medium_device = cuda.to_device(self.simulation.parameters.k_medium)
            azimuthal_angles_device = cuda.to_device(
                self.simulation.numerics.azimuthal_angles
            )
            e_r_device = cuda.to_device(self.simulation.numerics.e_r)
            pilm_device = cuda.to_device(pilm)
            taulm_device = cuda.to_device(taulm)
            e_field_theta_real_device = cuda.to_device(e_field_theta_real)
            e_field_theta_imag_device = cuda.to_device(e_field_theta_imag)
            e_field_phi_real_device = cuda.to_device(e_field_phi_real)
            e_field_phi_imag_device = cuda.to_device(e_field_phi_imag)

            sizes = (jmax, angles, wavelengths)
            threads_per_block = (16, 16, 2)
            # blocks_per_grid = tuple(
            #     [
            #         ceil(sizes[k] / threads_per_block[k])
            #         for k in range(len(threads_per_block))
            #     ]
            # )
            blocks_per_grid = tuple(
                ceil(sizes[k] / threads_per_block[k])
                for k in range(len(threads_per_block))
            )
            # blocks_per_grid = (
            #   ceil(jmax / threads_per_block[0]),
            #   ceil(angles / threads_per_block[1]),
            #   ceil(wavelengths / threads_per_block[2]))

            # print(f"{blocks_per_grid = }")

            compute_electric_field_angle_components_gpu[
                blocks_per_grid, threads_per_block
            ](
                self.simulation.numerics.lmax,
                particles_position_device,
                idx_device,
                sfc_device,
                k_medium_device,
                azimuthal_angles_device,
                e_r_device,
                pilm_device,
                taulm_device,
                e_field_theta_real_device,
                e_field_theta_imag_device,
                e_field_phi_real_device,
                e_field_phi_imag_device,
            )

            e_field_theta_real = e_field_theta_real_device.copy_to_host()
            e_field_theta_imag = e_field_theta_imag_device.copy_to_host()
            e_field_phi_real = e_field_phi_real_device.copy_to_host()
            e_field_phi_imag = e_field_phi_imag_device.copy_to_host()
            e_field_theta = e_field_theta_real + 1j * e_field_theta_imag
            e_field_phi = e_field_phi_real + 1j * e_field_phi_imag

            intensity = np.zeros_like(e_field_theta_real)
            dop = np.zeros_like(e_field_theta_real)
            dolp = np.zeros_like(e_field_theta_real)
            dolq = np.zeros_like(e_field_theta_real)
            dolu = np.zeros_like(e_field_theta_real)
            docp = np.zeros_like(e_field_theta_real)

            intensity_device = cuda.to_device(intensity)
            dop_device = cuda.to_device(dop)
            dolp_device = cuda.to_device(dolp)
            dolq_device = cuda.to_device(dolq)
            dolu_device = cuda.to_device(dolu)
            docp_device = cuda.to_device(docp)

            sizes = (angles, wavelengths)
            threads_per_block = (32, 32)
            # blocks_per_grid = tuple(
            #     [
            #         ceil(sizes[k] / threads_per_block[k])
            #         for k in range(len(threads_per_block))
            #     ]
            # )
            blocks_per_grid = tuple(
                ceil(sizes[k] / threads_per_block[k])
                for k in range(len(threads_per_block))
            )
            compute_polarization_components_gpu[blocks_per_grid, threads_per_block](
                self.simulation.parameters.k_medium.size,
                self.simulation.numerics.azimuthal_angles.size,
                e_field_theta_real_device,
                e_field_theta_imag_device,
                e_field_phi_real_device,
                e_field_phi_imag_device,
                intensity_device,
                dop_device,
                dolp_device,
                dolq_device,
                dolu_device,
                docp_device,
            )

            intensity = intensity_device.copy_to_host()
            dop = dop_device.copy_to_host()
            dolp = dolp_device.copy_to_host()
            dolq = dolq_device.copy_to_host()
            dolu = dolu_device.copy_to_host()
            docp = docp_device.copy_to_host()
        else:
            e_field_theta, e_field_phi = compute_electric_field_angle_components(
                self.simulation.numerics.lmax,
                self.simulation.parameters.particles.position,
                self.simulation.idx_lookup,
                self.simulation.scattered_field_coefficients,
                self.simulation.parameters.k_medium,
                self.simulation.numerics.azimuthal_angles,
                self.simulation.numerics.e_r,
                pilm,
                taulm,
            )

            intensity, dop, dolp, dolq, dolu, docp = compute_polarization_components(
                self.simulation.parameters.k_medium.size,
                self.simulation.numerics.azimuthal_angles.size,
                e_field_theta,
                e_field_phi,
            )

        self.scattering_angles = self.simulation.numerics.polar_angles
        k_medium = self.simulation.parameters.k_medium
        if type(self.simulation.parameters.k_medium) == pd.core.series.Series:
            k_medium = k_medium.to_numpy()
        self.phase_function_3d = (
            intensity
            * 4
            * np.pi
            / np.power(np.abs(k_medium), 2)
            / self.c_sca[np.newaxis, :]
        )
        if check_phase_function:
            res = self.__check_phase_function()
            assert (
                res == True
            ), "The phase function does have the desired precision. Please increase the amount of angles used."

        self.phase_function_legendre_coefficients = np.polynomial.legendre.legfit(
            np.cos(self.scattering_angles),
            self.phase_function_3d,
            legendre_coefficients_number,
        )

        self.degree_of_polarization_3d = dop
        self.degree_of_linear_polarization_3d = dolp
        self.degree_of_linear_polarization_q_3d = dolq
        self.degree_of_linear_polarization_u_3d = dolu
        self.degree_of_circular_polarization_3d = docp

        if (self.simulation.numerics.sampling_points_number is not None) and (
            self.simulation.numerics.sampling_points_number.size == 2
        ):
            self.phase_function = np.mean(
                np.reshape(
                    self.phase_function_3d,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )

            self.degree_of_polarization = np.mean(
                np.reshape(
                    dop,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_linear_polarization = np.mean(
                np.reshape(
                    dolp,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_linear_polarization_q = np.mean(
                np.reshape(
                    dolq,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_linear_polarization_u = np.mean(
                np.reshape(
                    dolu,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_circular_polarization = np.mean(
                np.reshape(
                    docp,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )

            self.scattering_angles = np.reshape(
                self.scattering_angles, self.simulation.numerics.sampling_points_number
            )
            self.scattering_angles = self.scattering_angles[0, :]
        else:
            self.phase_function = self.phase_function_3d

            self.degree_of_polarization = dop
            self.degree_of_linear_polarization = dolp
            self.degree_of_linear_polarization_q = dolq
            self.degree_of_linear_polarization_u = dolu
            self.degree_of_circular_polarization = docp

        self.c_and_b_bounds = c_and_b
        if isinstance(c_and_b, bool):
            if c_and_b:
                self.c_and_b_bounds = ([-1, 0], [1, 1])
            else:
                return

        self.__compute_c_and_b()

    def __check_phase_function(self, precision=1e-4) -> bool:
        integral = 0
        delta_phi = (
            len(self.simulation.numerics.azimuthal_angles)
            / self.simulation.numerics.sampling_points_number[0]
        )
        delta_theta = (
            len(self.simulation.numerics.polar_angles)
            / self.simulation.numerics.sampling_points_number[1]
        )
        for idx in self.simulation.numerics.azimuthal_angles:
            integral += (
                self.phase_function_3d[idx, :]
                * delta_phi
                * np.sin(self.simulation.numerics.azimuthal_angles[idx])
                * delta_theta
            )
        if 1 - integral / (4 * np.pi) > precision:
            return False
        else:
            return True

    def compute_asymmetry(self):
        """Computes the asymmetry parameter by numerical integration over the phase function.
        Therefore depends on the chosen sampling points. Accuracy depends on the number of
        sampling points. **ACCURACY CURRENTLY VERY MUCH IN QUESTION**
        """
        sum = 0
        delta_phi = (2 * np.pi) / self.simulation.numerics.sampling_points_number[0]
        delta_theta = np.pi / self.simulation.numerics.sampling_points_number[1]
        for idx in range(len(self.simulation.numerics.azimuthal_angles)):
            rn = np.cos(self.simulation.numerics.polar_angles[idx])
            a = (
                self.phase_function_3d[idx, :]
                * rn
                * delta_phi
                * np.sin(self.simulation.numerics.polar_angles[idx])
                * delta_theta
            )
            sum += a
        g = sum / (4 * np.pi)
        self.g = g

    def __compute_correction(self) -> float:
        """Naive implementation of a correction term to lessen the underestimation of a spheres surface
        via numerical integration. Does not adequately correct the result, but does improve it.

        Returns:
            correction_term (float): Computed error term
        """
        integral = 0
        delta_phi = (2 * np.pi) / self.simulation.numerics.sampling_points_number[0]
        delta_theta = np.pi / self.simulation.numerics.sampling_points_number[1]
        for idx in range(len(self.simulation.numerics.azimuthal_angles)):
            integral += (
                np.sin(self.simulation.numerics.polar_angles[idx])
                * delta_theta
                * delta_phi
            )
        error = 1 - integral / (4 * np.pi)
        return 12 * error / len(self.simulation.numerics.azimuthal_angles) ** 2

    def compute_asymmetry_corrected(self):
        """Computes asymmetry parameter via numerical integration over the phase function.
        Makes use of a correction term for the surface to compensate for underestimation
        of the spaces surface area. Therefore depends on the chosen sampling points.
        Accuracy depends on the number of sampling points.
        **ACCURACY CURRENTLY VERY MUCH IN QUESTION**
        """
        correction_term = self.__compute_correction()
        print(correction_term)
        sum = 0
        delta_phi = (2 * np.pi) / self.simulation.numerics.sampling_points_number[0]
        delta_theta = np.pi / self.simulation.numerics.sampling_points_number[1]
        for idx in range(len(self.simulation.numerics.azimuthal_angles)):
            rn = np.cos(self.simulation.numerics.polar_angles[idx])
            a = (
                self.phase_function_3d[idx, :]
                * rn
                * delta_phi
                * np.sin(self.simulation.numerics.polar_angles[idx])
                * delta_theta
            )
            correction = self.phase_function_3d[idx, :] * rn * correction_term
            sum += a
            sum += correction
        g = sum / (4 * np.pi)
        self.g_corr = g

    def compute_asymmetry_coeff(self):
        """WIP: Tried using the scattered field coefficients p & q to calculate the asymmetry parameter.
        Does **not** work. Might however be subject to further investigation,
        to achievce more reliable results.
        """
        A = self.simulation.scattered_field_coefficients
        a_i = []
        b_i = []
        for n in range(A.shape[1]):
            _, tau, _, _ = single_index2multi(n, self.simulation.numerics.lmax)
            if tau == 1:
                a_i.append(A[:, n, :])
            else:
                b_i.append(A[:, n, :])

        g = []
        for wv in range(self.simulation.parameters.wavelength.shape[0]):
            sum_term = 0
            for i_n in range(len(a_i) - 1):
                n = i_n + 1
                a_n = np.squeeze(a_i[i_n][:, wv])
                b_n = np.squeeze(b_i[i_n][:, wv])
                a_n2 = np.squeeze(a_i[i_n + 1][:, wv])
                b_n2 = np.squeeze(b_i[i_n + 1][:, wv])
                p1 = (
                    np.dot(a_n, np.conj(a_n2)) + np.dot(b_n, np.conj(b_n2))
                ).real  # *((n*(n+2))/(n+1))
                p2 = (np.dot(a_n, np.conj(b_n))).real  # *((2*n+1)/(n*(n+1)))
                sum_term += p1 + p2
            g_wv = (
                (4 * np.pi)
                / (
                    np.power(np.abs(self.simulation.parameters.k_particle[:, wv]), 2)
                    * self.c_sca[wv]
                )
            ) * sum_term
            g.append(g_wv)
        self.g = g
        return g

    @staticmethod
    def compute_double_henyey_greenstein(theta: np.ndarray, cb: np.ndarray):
        """
        Calculates the scattering phase function using the double Henyey-Greenstein model.

        Args:
            theta (np.ndarray): A numpy array representing the scattering angle. It contains the
                values at which you want to compute the double Henyey-Greenstein phase function.
            cb (np.ndarray): A numpy array that represents the coefficients of the double
                Henyey-Greenstein phase function. It should have a size of either 2 or 3.

        Returns:
            p (np.ndarray): Computed Henyey-Greenstein phase function.
        """
        cb = np.squeeze(cb)
        if cb.size < 2:
            cb = np.array([0, 0.5, 0.5])
        elif cb.size == 2:
            cb = np.append(cb, cb[1])
        elif cb.size > 3:
            cb = cb[:2]

        p1 = (1 - cb[1] ** 2) / np.power(
            1 - 2 * cb[1] * np.cos(theta) + cb[1] ** 2, 3 / 2
        )
        p2 = (1 - cb[2] ** 2) / np.power(
            1 + 2 * cb[2] * np.cos(theta) + cb[2] ** 2, 3 / 2
        )
        return (1 - cb[0]) / 2 * p1 + (1 + cb[0]) / 2 * p2

    def __compute_c_and_b(self):
        """The function computes the values of parameters 'b' and 'c' using the double Henyey-Greenstein
        phase function.
        """
        # double henyey greenstein
        if len(self.c_and_b_bounds[0]) not in [2, 3]:
            self.c_and_b_bounds = ([-1, 0], [1, 1])
            self.log.warning(
                "Number of parameters need to be 2 (b,c) or 3 (b1,b2,c). Reverting to two parameters (b,c) and setting the bounds to standard: b in [0, 1] and c in [-1, 1]"
            )

        from scipy.optimize import least_squares

        if len(self.c_and_b_bounds) == 2:
            bc0 = np.array([0, 0.5])
        else:
            bc0 = np.array([0, 0.5, 0.5])

        self.cb = np.empty((self.phase_function.shape[1], len(self.c_and_b_bounds)))
        for w in range(self.phase_function.shape[1]):
            # def dhg_optimization(bc):
            #     return (
            #         Optics.compute_double_henyey_greenstein(self.scattering_angles, bc)
            #         - self.phase_function[:, w]
            #     )

            # bc = least_squares(
            #     dhg_optimization, bc0, jac="2-point", bounds=self.c_and_b_bounds
            # )

            bc = least_squares(
                lambda bc: Optics.compute_double_henyey_greenstein(
                    self.scattering_angles, bc
                )
                - self.phase_function[:, w],
                bc0,
                jac="2-point",
                bounds=self.c_and_b_bounds,
            )
            self.cb[w, :] = bc.x

    def compute_phase_function_batched(
        self,
        legendre_coefficients_number: int = 15,
        c_and_b: Union[bool, tuple] = False,
        check_phase_function: bool = False,
    ):
        """WIP WIP WIP WIP WIP WIP WIP WIP WIP

        Args:
            legendre_coefficients_number (int, optional): The number of Legendre coefficients to compute
                for the phase function. These coefficients are used to approximate the phase function
                using Legendre polynomials. The higher the number of coefficients, the more accurate
                the approximation will be.
            c_and_b (Union[bool, tuple], optional): A boolean value or a tuple. If `True`, it indicates
                that the `c` and `b` bounds should be computed. If `False`, the function will return
                without computing the `c` and `b` bounds.
        """
        pilm, taulm = spherical_functions_trigon(
            self.simulation.numerics.lmax, self.simulation.numerics.polar_angles
        )

        if self.simulation.numerics.gpu:
            jmax = (
                self.simulation.parameters.particles.number
                * self.simulation.numerics.nmax
            )
            angles = self.simulation.numerics.azimuthal_angles.size
            wavelengths = self.simulation.parameters.k_medium.size
            e_field_theta_real = np.zeros(
                (
                    self.simulation.numerics.azimuthal_angles.size,
                    self.simulation.parameters.k_medium.size,
                ),
                dtype=float,
            )
            e_field_theta_imag = np.zeros_like(e_field_theta_real)
            e_field_phi_real = np.zeros_like(e_field_theta_real)
            e_field_phi_imag = np.zeros_like(e_field_theta_real)

            # stuff we need in full for every iteration
            particles_position_device = cuda.to_device(
                self.simulation.parameters.particles.position
            )
            idx_device = cuda.to_device(self.simulation.idx_lookup)
            sfc_device = cuda.to_device(self.simulation.scattered_field_coefficients)
            k_medium_device = cuda.to_device(self.simulation.parameters.k_medium)

            sizes = (jmax, angles, wavelengths)
            threads_per_block = (16, 16, 2)
            # blocks_per_grid = tuple(
            #     [
            #         ceil(sizes[k] / threads_per_block[k])
            #         for k in range(len(threads_per_block))
            #     ]
            # )
            blocks_per_grid = tuple(
                ceil(sizes[k] / threads_per_block[k])
                for k in range(len(threads_per_block))
            )
            # blocks_per_grid = (
            #   ceil(jmax / threads_per_block[0]),
            #   ceil(angles / threads_per_block[1]),
            #   ceil(wavelengths / threads_per_block[2]))

            to_split = [
                self.simulation.numerics.azimuthal_angles,
                self.simulation.numerics.e_r,
                pilm,
                taulm,
                e_field_theta_real,
                e_field_theta_imag,
                e_field_phi_real,
                e_field_phi_imag,
            ]

            # split data along axis, send maximum amount to GPU, wait for it to be done
            # store results, repeat
            idx_to_split = self.simulation.numerics.azimuthal_angles.shape[0]

            idx_per_array = []
            for array in to_split:
                idx_per_array.append(
                    np.where(np.array(array.shape) == idx_to_split)[0][0]
                )

            threads_per_block = (1, 16 * 16, 2)

            start_idx = 0
            split_idx = 0
            done = False
            while not done:

                if split_idx != 0:
                    to_split[0] = to_split[0][split_idx + 1 :]
                    to_split[1] = to_split[1][split_idx + 1 :, :]
                    to_split[2] = to_split[2][:, :, split_idx + 1 :]
                    to_split[3] = to_split[3][:, :, split_idx + 1 :]
                    to_split[4] = to_split[4][split_idx + 1 :, :]
                    to_split[5] = to_split[5][split_idx + 1 :, :]
                    to_split[6] = to_split[6][split_idx + 1 :, :]
                    to_split[7] = to_split[7][split_idx + 1 :, :]

                    # for i in range(len(to_split)):
                    #     to_split[i] = to_split[i].take(indices=range(start_idx,idx_to_split), axis=idx_per_array[i])

                split_idx = self.__compute_data_split(
                    to_split,
                    idx_list=idx_per_array,
                    threads_per_block=threads_per_block[1],
                )
                if split_idx < 1:
                    break

                azimuthal_split = np.ascontiguousarray(
                    self.simulation.numerics.azimuthal_angles[
                        start_idx : start_idx + split_idx
                    ]
                )
                e_r_split = np.ascontiguousarray(
                    self.simulation.numerics.e_r[start_idx : start_idx + split_idx, :]
                )
                pilm_split = np.ascontiguousarray(
                    pilm[:, :, start_idx : start_idx + split_idx]
                )
                taulm_split = np.ascontiguousarray(
                    taulm[:, :, start_idx : start_idx + split_idx]
                )
                e_theta_imag_split = np.ascontiguousarray(
                    e_field_theta_imag[start_idx : start_idx + split_idx, :]
                )
                e_theta_real_split = np.ascontiguousarray(
                    e_field_theta_real[start_idx : start_idx + split_idx, :]
                )
                e_phi_imag_split = np.ascontiguousarray(
                    e_field_phi_imag[start_idx : start_idx + split_idx, :]
                )
                e_phi_real_split = np.ascontiguousarray(
                    e_field_phi_real[start_idx : start_idx + split_idx, :]
                )

                azimuthal_angles_device = cuda.to_device(azimuthal_split)
                e_r_device = cuda.to_device(e_r_split)
                pilm_device = cuda.to_device(pilm_split)
                taulm_device = cuda.to_device(taulm_split)
                e_field_theta_real_device = cuda.to_device(e_theta_real_split)
                e_field_theta_imag_device = cuda.to_device(e_theta_imag_split)
                e_field_phi_real_device = cuda.to_device(e_phi_real_split)
                e_field_phi_imag_device = cuda.to_device(e_phi_imag_split)

                sizes = (jmax, azimuthal_split.size, wavelengths)

                blocks_per_grid = tuple(
                    ceil(sizes[k] / threads_per_block[k])
                    for k in range(len(threads_per_block))
                )

                compute_electric_field_angle_components_gpu[
                    blocks_per_grid, threads_per_block
                ](
                    self.simulation.numerics.lmax,
                    particles_position_device,
                    idx_device,
                    sfc_device,
                    k_medium_device,
                    azimuthal_angles_device,
                    e_r_device,
                    pilm_device,
                    taulm_device,
                    e_field_theta_real_device,
                    e_field_theta_imag_device,
                    e_field_phi_real_device,
                    e_field_phi_imag_device,
                )

                e_field_theta_real[start_idx : start_idx + split_idx, :] = (
                    e_field_theta_real_device.copy_to_host()
                )
                e_field_theta_imag[start_idx : start_idx + split_idx, :] = (
                    e_field_theta_imag_device.copy_to_host()
                )
                e_field_phi_real[start_idx : start_idx + split_idx, :] = (
                    e_field_phi_real_device.copy_to_host()
                )
                e_field_phi_imag[start_idx : start_idx + split_idx, :] = (
                    e_field_phi_imag_device.copy_to_host()
                )

                # update start_idx
                start_idx += split_idx + 1
                if start_idx >= idx_to_split:
                    done = True

            e_field_theta = e_field_theta_real + 1j * e_field_theta_imag
            e_field_phi = e_field_phi_real + 1j * e_field_phi_imag

            print("Done with e field calculations...")
            # continue with next calculation

            intensity = np.zeros_like(e_field_theta_real)
            dop = np.zeros_like(e_field_theta_real)
            dolp = np.zeros_like(e_field_theta_real)
            dolq = np.zeros_like(e_field_theta_real)
            dolu = np.zeros_like(e_field_theta_real)
            docp = np.zeros_like(e_field_theta_real)

            to_split = [
                e_field_theta_real,
                e_field_theta_imag,
                e_field_phi_real,
                e_field_phi_imag,
                intensity,
                dop,
                dolp,
                dolq,
                dolu,
                docp,
            ]
            threads_per_block = (
                1024,
                1,
            )  # this allows ~65000 wavelengths, limits required batching
            # blocks_per_grid = tuple(
            #     [
            #         ceil(sizes[k] / threads_per_block[k])
            #         for k in range(len(threads_per_block))
            #     ]
            # )

            start_idx = 0
            split_idx = 0
            idx_per_array = [0] * len(to_split)
            done = False
            while not done:

                if split_idx != 0:
                    for i in range(len(to_split)):
                        to_split[i] = to_split[i][split_idx + 1 :, :]

                split_idx = self.__compute_data_split(
                    to_split,
                    idx_list=idx_per_array,
                    threads_per_block=threads_per_block[0],
                )
                if split_idx < 1:
                    break

                intensity_split = np.ascontiguousarray(
                    intensity[start_idx : start_idx + split_idx, :]
                )
                dop_split = np.ascontiguousarray(
                    dop[start_idx : start_idx + split_idx, :]
                )
                dolp_split = np.ascontiguousarray(
                    dolp[start_idx : start_idx + split_idx, :]
                )
                dolq_split = np.ascontiguousarray(
                    dolq[start_idx : start_idx + split_idx, :]
                )
                dolu_split = np.ascontiguousarray(
                    dolu[start_idx : start_idx + split_idx, :]
                )
                docp_split = np.ascontiguousarray(
                    docp[start_idx : start_idx + split_idx, :]
                )

                e_theta_imag_split = np.ascontiguousarray(
                    e_field_theta_imag[start_idx : start_idx + split_idx, :]
                )
                e_theta_real_split = np.ascontiguousarray(
                    e_field_theta_real[start_idx : start_idx + split_idx, :]
                )
                e_phi_imag_split = np.ascontiguousarray(
                    e_field_phi_imag[start_idx : start_idx + split_idx, :]
                )
                e_phi_real_split = np.ascontiguousarray(
                    e_field_phi_real[start_idx : start_idx + split_idx, :]
                )

                sizes = (split_idx, wavelengths)
                blocks_per_grid = tuple(
                    ceil(sizes[k] / threads_per_block[k])
                    for k in range(len(threads_per_block))
                )

                intensity_device = cuda.to_device(intensity_split)
                dop_device = cuda.to_device(dop_split)
                dolp_device = cuda.to_device(dolp_split)
                dolq_device = cuda.to_device(dolq_split)
                dolu_device = cuda.to_device(dolu_split)
                docp_device = cuda.to_device(docp_split)

                e_field_theta_real_device = cuda.to_device(e_theta_real_split)
                e_field_theta_imag_device = cuda.to_device(e_theta_imag_split)
                e_field_phi_real_device = cuda.to_device(e_phi_real_split)
                e_field_phi_imag_device = cuda.to_device(e_phi_imag_split)

                compute_polarization_components_gpu[blocks_per_grid, threads_per_block](
                    self.simulation.parameters.k_medium.size,
                    self.simulation.numerics.azimuthal_angles.size,
                    e_field_theta_real_device,
                    e_field_theta_imag_device,
                    e_field_phi_real_device,
                    e_field_phi_imag_device,
                    intensity_device,
                    dop_device,
                    dolp_device,
                    dolq_device,
                    dolu_device,
                    docp_device,
                )

                intensity[start_idx : start_idx + split_idx, :] = (
                    intensity_device.copy_to_host()
                )
                dop[start_idx : start_idx + split_idx, :] = dop_device.copy_to_host()
                dolq[start_idx : start_idx + split_idx, :] = dolp_device.copy_to_host()
                dolq[start_idx : start_idx + split_idx, :] = dolq_device.copy_to_host()
                dolu[start_idx : start_idx + split_idx, :] = dolu_device.copy_to_host()
                docp[start_idx : start_idx + split_idx, :] = docp_device.copy_to_host()

                # update start_idx
                start_idx += split_idx + 1
                if start_idx >= idx_to_split:
                    done = True

        self.scattering_angles = self.simulation.numerics.polar_angles
        k_medium = self.simulation.parameters.k_medium
        if type(self.simulation.parameters.k_medium) == pd.core.series.Series:
            k_medium = k_medium.to_numpy()
        self.phase_function_3d = (
            intensity
            * 4
            * np.pi
            / np.power(np.abs(k_medium), 2)
            / self.c_sca[np.newaxis, :]
        )
        if check_phase_function:
            res = self.__check_phase_function()
            assert (
                res == True
            ), "The phase function does have the desired precision. Please increase the amount of angles used."

        self.phase_function_legendre_coefficients = np.polynomial.legendre.legfit(
            np.cos(self.scattering_angles),
            self.phase_function_3d,
            legendre_coefficients_number,
        )

        self.degree_of_polarization_3d = dop
        self.degree_of_linear_polarization_3d = dolp
        self.degree_of_linear_polarization_q_3d = dolq
        self.degree_of_linear_polarization_u_3d = dolu
        self.degree_of_circular_polarization_3d = docp

        if (self.simulation.numerics.sampling_points_number is not None) and (
            self.simulation.numerics.sampling_points_number.size == 2
        ):
            self.phase_function = np.mean(
                np.reshape(
                    self.phase_function_3d,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )

            self.degree_of_polarization = np.mean(
                np.reshape(
                    dop,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_linear_polarization = np.mean(
                np.reshape(
                    dolp,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_linear_polarization_q = np.mean(
                np.reshape(
                    dolq,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_linear_polarization_u = np.mean(
                np.reshape(
                    dolu,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )
            self.degree_of_circular_polarization = np.mean(
                np.reshape(
                    docp,
                    np.append(
                        self.simulation.numerics.sampling_points_number,
                        self.simulation.parameters.k_medium.size,
                    ),
                ),
                axis=0,
            )

            self.scattering_angles = np.reshape(
                self.scattering_angles, self.simulation.numerics.sampling_points_number
            )
            self.scattering_angles = self.scattering_angles[0, :]
        else:
            self.phase_function = self.phase_function_3d

            self.degree_of_polarization = dop
            self.degree_of_linear_polarization = dolp
            self.degree_of_linear_polarization_q = dolq
            self.degree_of_linear_polarization_u = dolu
            self.degree_of_circular_polarization = docp

        self.c_and_b_bounds = c_and_b
        if isinstance(c_and_b, bool):
            if c_and_b:
                self.c_and_b_bounds = ([-1, 0], [1, 1])
            else:
                return

        self.__compute_c_and_b()

    def __compute_data_split(
        self, data: list[np.ndarray], idx_list: list, threads_per_block: int
    ) -> int:

        # buffer = 10000 # buffer to accomodate for varying GPU mem usage
        device = cuda.select_device(0)
        handle = cuda.cudadrv.devices.get_context()
        mem_info = cuda.cudadrv.driver.Context(device, handle).get_memory_info()
        free_bytes = mem_info.free
        total_data_bytes = 0
        for array in data:
            total_data_bytes += array.size * array.itemsize

        print("---------------------------------------------------")
        print(f"{total_data_bytes*1e-9} GB of data remaining")
        idx = data[0].shape[idx_list[0]]
        num = idx
        while total_data_bytes > free_bytes:
            new_data_bytes = 0
            num -= 1000
            for i in range(len(data)):
                temp_shape = data[i].shape
                temp_size = 1
                for s in temp_shape:
                    if s == idx:
                        temp_size *= num
                    else:
                        temp_size *= s
                new_data_bytes += temp_size * data[i].itemsize
            total_data_bytes = new_data_bytes

        print(f"{free_bytes*1e-9} GB of data available on GPU")
        print(f"Unused GPU memory: {(free_bytes-total_data_bytes)*1e-6} MB")

        print(f"{num//threads_per_block} > {2**16-1}?")
        if num // threads_per_block > 2**16 - 1:
            num = (2**16 - 1) * threads_per_block
            print("need to limit number of blocks")

        print("---------------------------------------------------")
        return num
