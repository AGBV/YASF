from abc import ABC, abstractmethod
import numpy as np


class Computer(ABC):
    @abstractmethod
    @staticmethod
    def particle_interaction(
        lmax: int,
        particle_number: int,
        idx: np.ndarray,
        x: np.ndarray,
        wx_real: np.ndarray,
        wx_imag: np.ndarray,
        translation_table: np.ndarray,
        plm: np.ndarray,
        sph_h: np.ndarray,
        e_j_dm_phi,
    ):
        """
        Perform particle interaction calculations on the GPU.

        Args:
            lmax (int): Maximum angular momentum quantum number.
            particle_number (int): Number of particles.
            idx (np.ndarray): Array containing particle indices.
            x (np.ndarray): Array of particle positions.
            wx_real (np.ndarray): Array to store the real part of the result.
            wx_imag (np.ndarray): Array to store the imaginary part of the result.
            translation_table (np.ndarray): Array containing translation table.
            plm (np.ndarray): Array containing associated Legendre polynomials.
            sph_h (np.ndarray): Array containing spherical harmonics.
            e_j_dm_phi (np.ndarray): Additional parameter for the calculation.

        Todo:
            Implement data batching for GPUs with smaller memory
        """
        return

    @abstractmethod
    @staticmethod
    def compute_scattering_cross_section(
        lmax: int,
        particle_number: int,
        idx: np.ndarray,
        sfc: np.ndarray,
        translation_table: np.ndarray,
        plm: np.ndarray,
        sph_h: np.ndarray,
        e_j_dm_phi: np.ndarray,
        c_sca_real: np.ndarray,
        c_sca_imag: np.ndarray,
    ):
        """
        Compute the scattering cross section on the GPU using CUDA.

        Args:
            lmax (int): The maximum degree of the spherical harmonics expansion.
            particle_number (int): The number of particles.
            idx (np.ndarray): The index array.
            sfc (np.ndarray): The scattering form factor array.
            translation_table (np.ndarray): The translation table array.
            plm (np.ndarray): The associated Legendre polynomials array.
            sph_h (np.ndarray): The spherical harmonics array.
            e_j_dm_phi (np.ndarray): The phase factor array.
            c_sca_real (np.ndarray): The real part of the scattering cross section array.
            c_sca_imag (np.ndarray): The imaginary part of the scattering cross section array.
        """
        return

    @abstractmethod
    @staticmethod
    def compute_radial_independent_scattered_field(
        lmax: int,
        particles_position: np.ndarray,
        idx: np.ndarray,
        sfc: np.ndarray,
        k_medium: np.ndarray,
        azimuthal_angles: np.ndarray,
        e_r: np.ndarray,
        e_phi: np.ndarray,
        e_theta: np.ndarray,
        pilm: np.ndarray,
        taulm: np.ndarray,
        e_1_sca_real: np.ndarray,
        e_1_sca_imag: np.ndarray,
    ):
        """
        Compute the radial independent scattered field using GPU acceleration.

        Args:
            lmax (int): The maximum degree of the spherical harmonics expansion.
            particles_position (np.ndarray): Array of particle positions.
            idx (np.ndarray): Array of indices for particle properties.
            sfc (np.ndarray): Array of scattering form factors.
            k_medium (np.ndarray): Array of wave numbers in the medium.
            azimuthal_angles (np.ndarray): Array of azimuthal angles.
            e_r (np.ndarray): Array of radial electric field components.
            e_phi (np.ndarray): Array of azimuthal electric field components.
            e_theta (np.ndarray): Array of polar electric field components.
            pilm (np.ndarray): Array of associated Legendre polynomials.
            taulm (np.ndarray): Array of tau coefficients.
            e_1_sca_real (np.ndarray): Array of real parts of the scattered electric field.
            e_1_sca_imag (np.ndarray): Array of imaginary parts of the scattered electric field.
        """
        return

    @abstractmethod
    @staticmethod
    def compute_electric_field_angle_components(
        lmax: int,
        particles_position: np.ndarray,
        idx: np.ndarray,
        sfc: np.ndarray,
        k_medium: np.ndarray,
        azimuthal_angles: np.ndarray,
        e_r: np.ndarray,
        pilm: np.ndarray,
        taulm: np.ndarray,
        e_field_theta_real: np.ndarray,
        e_field_theta_imag: np.ndarray,
        e_field_phi_real: np.ndarray,
        e_field_phi_imag: np.ndarray,
    ):
        """
        Compute the electric field angle components on the GPU.

        Args:
            lmax (int): The maximum angular momentum quantum number.
            particles_position (np.ndarray): Array of particle positions.
            idx (np.ndarray): Array of indices.
            sfc (np.ndarray): Array of scattering form factors.
            k_medium (np.ndarray): Array of medium wavevectors.
            azimuthal_angles (np.ndarray): Array of azimuthal angles.
            e_r (np.ndarray): Array of radial unit vectors.
            pilm (np.ndarray): Array of associated Legendre polynomials.
            taulm (np.ndarray): Array of tau coefficients.
            e_field_theta_real (np.ndarray): Array of real parts of electric field theta component.
            e_field_theta_imag (np.ndarray): Array of imaginary parts of electric field theta component.
            e_field_phi_real (np.ndarray): Array of real parts of electric field phi component.
            e_field_phi_imag (np.ndarray): Array of imaginary parts of electric field phi component.
        """
        return

    @abstractmethod
    @staticmethod
    def compute_polarization_components(
        number_of_wavelengths: int,
        number_of_angles: int,
        e_field_theta_real: np.ndarray,
        e_field_theta_imag: np.ndarray,
        e_field_phi_real: np.ndarray,
        e_field_phi_imag: np.ndarray,
        intensity: np.ndarray,
        degree_of_polarization: np.ndarray,
        degree_of_linear_polarization: np.ndarray,
        degree_of_linear_polarization_q: np.ndarray,
        degree_of_linear_polarization_u: np.ndarray,
        degree_of_circular_polarization: np.ndarray,
    ):
        """
        Compute the polarization components using GPU acceleration.

        Args:
            number_of_wavelengths (int): Number of wavelengths.
            number_of_angles (int): Number of angles.
            e_field_theta_real (np.ndarray): Real part of the electric field in the theta direction.
            e_field_theta_imag (np.ndarray): Imaginary part of the electric field in the theta direction.
            e_field_phi_real (np.ndarray): Real part of the electric field in the phi direction.
            e_field_phi_imag (np.ndarray): Imaginary part of the electric field in the phi direction.
            intensity (np.ndarray): Array to store the intensity component.
            degree_of_polarization (np.ndarray): Array to store the degree of polarization component.
            degree_of_linear_polarization (np.ndarray): Array to store the degree of linear polarization component.
            degree_of_linear_polarization_q (np.ndarray): Array to store the degree of linear polarization (Q) component.
            degree_of_linear_polarization_u (np.ndarray): Array to store the degree of linear polarization (U) component.
            degree_of_circular_polarization (np.ndarray): Array to store the degree of circular polarization component.
        """
        return

    @abstractmethod
    @staticmethod
    def compute_field(
        lmax: int,
        idx: np.ndarray,
        size_parameter: np.ndarray,
        sph_h: np.ndarray,
        derivative: np.ndarray,
        e_j_dm_phi: np.ndarray,
        p_lm: np.ndarray,
        pi_lm: np.ndarray,
        tau_lm: np.ndarray,
        e_r: np.ndarray,
        e_theta: np.ndarray,
        e_phi: np.ndarray,
        scattered_field_coefficients: np.ndarray,
        field_real: np.ndarray,
        field_imag: np.ndarray,
    ):  # , initial_field_coefficients: np.ndarray, scatter_to_internal: np.ndarray):
        """
        Compute the field on the GPU using CUDA.

        Args:
            lmax (int): Maximum degree of the spherical harmonics.
            idx (np.ndarray): Array of indices.
            size_parameter (np.ndarray): Array of size parameters.
            sph_h (np.ndarray): Array of spherical harmonics.
            derivative (np.ndarray): Array of derivatives.
            e_j_dm_phi (np.ndarray): Array of phi-dependent terms.
            p_lm (np.ndarray): Array of Legendre polynomials.
            pi_lm (np.ndarray): Array of pi-dependent terms.
            tau_lm (np.ndarray): Array of tau-dependent terms.
            e_r (np.ndarray): Array of r-dependent terms.
            e_theta (np.ndarray): Array of theta-dependent terms.
            e_phi (np.ndarray): Array of phi-dependent terms.
            scattered_field_coefficients (np.ndarray): Array of scattered field coefficients.
            field_real (np.ndarray): Array to store the real part of the field.
            field_imag (np.ndarray): Array to store the imaginary part of the field.
        """
        return
