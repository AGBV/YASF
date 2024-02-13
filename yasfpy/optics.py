import logging
from typing import Union
from math import ceil

import numpy as np
from numba import cuda
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
    """
    Class representing the optics of a simulation.

    Args:
        simulation (Simulation): The simulation object.

    Attributes:
        c_ext (ndarray): Array of complex numbers representing the extinction cross sections.
        c_sca (ndarray): Array of complex numbers representing the scattering cross sections.
        log (Logger): Logger object for logging messages.
        albedo (ndarray): Array of real numbers representing the albedo.
        scattering_angles (ndarray): Array of scattering angles.
        phase_function_3d (ndarray): Array of phase function values in 3D.
        phase_function_legendre_coefficients (ndarray): Array of Legendre coefficients of the phase function.
        degree_of_polarization_3d (ndarray): Array of degree of polarization values in 3D.
        degree_of_linear_polarization_3d (ndarray): Array of degree of linear polarization values in 3D.
        degree_of_linear_polarization_q_3d (ndarray): Array of degree of linear polarization (Q) values in 3D.
        degree_of_linear_polarization_u_3d (ndarray): Array of degree of linear polarization (U) values in 3D.
        degree_of_circular_polarization_3d (ndarray): Array of degree of circular polarization values in 3D.
    """

    def __init__(self, simulation: Simulation):
        self.simulation = simulation

        self.c_ext = np.zeros_like(simulation.parameters.wavelength, dtype=complex)
        self.c_sca = np.zeros_like(simulation.parameters.wavelength, dtype=complex)

        self.log = logging.getLogger(__name__)

    def compute_cross_sections(self):
        """
        Compute the cross sections.
        """
        # Code for computing cross sections...

    def compute_phase_funcition(
        self,
        legendre_coefficients_number: int = 15,
        c_and_b: Union[bool, tuple] = False,
    ):
        """
        Compute the phase function.

        Args:
            legendre_coefficients_number (int): Number of Legendre coefficients to compute.
            c_and_b (Union[bool, tuple]): Whether to compute the C and B matrices.

        Returns:
            None
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
            blocks_per_grid = tuple(
                [
                    ceil(sizes[k] / threads_per_block[k])
                    for k in range(len(threads_per_block))
                ]
            )
            # blocks_per_grid = (
            #   ceil(jmax / threads_per_block[0]),
            #   ceil(angles / threads_per_block[1]),
            #   ceil(wavelengths / threads_per_block[2]))

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
            blocks_per_grid = tuple(
                [
                    ceil(sizes[k] / threads_per_block[k])
                    for k in range(len(threads_per_block))
                ]
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

        self.phase_function_3d = (
            intensity
            * 4
            * np.pi
            / np.power(np.abs(self.simulation.parameters.k_medium), 2)
            / self.c_sca[np.newaxis, :]
        )
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

    @staticmethod
    def compute_double_henyey_greenstein(theta: np.ndarray, cb: np.ndarray):
        """
        Compute the double Henyey-Greenstein phase function.

        Parameters:
        theta (np.ndarray): Array of angles at which to compute the phase function.
        cb (np.ndarray): Array of phase function coefficients.

        Returns:
        np.ndarray: Array of phase function values corresponding to the given angles.
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
        """
        Compute the values of c and b parameters for the double Henyey-Greenstein phase function.

        If the number of parameters is not 2 (b,c) or 3 (b1,b2,c), the function reverts to two parameters (b,c)
        and sets the bounds to standard values: b in [0, 1] and c in [-1, 1].

        Uses scipy's least_squares optimization method to find the optimal values of c and b.

        Returns:
            None
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

            def dhg_optimization(bc):
                return (
                    Optics.compute_double_henyey_greenstein(self.scattering_angles, bc)
                    - self.phase_function[:, w]
                )

            bc = least_squares(
                dhg_optimization, bc0, jac="2-point", bounds=self.c_and_b_bounds
            )
            self.cb[w, :] = bc.x
