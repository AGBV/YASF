import numpy as np

from yasfpy.particles import Particles
from yasfpy.initial_field import InitialField


import numpy as np


class Parameters:
    """
    The Parameters class represents the parameters for a simulation, including wavelength, refractive
    indices, scattering particles, and initial field, and provides methods for computing angular
    frequency and wave vectors.
    """

    def __init__(
        self,
        wavelength: np.array,
        medium_refractive_index: np.array,
        particles: Particles,
        initial_field: InitialField,
    ):
        """Initializes the class with the given parameters and sets up the necessary variables.

        Args:
            wavelength (np.array): An array that represents the wavelengths of the light being used.
                It contains the values of the wavelengths at which the simulation will be performed.
            medium_refractive_index (np.array): An array that represents the refractive index of the
                medium in which the particles are located. It contains the refractive index values at different
                wavelengths.
            particles (Particles): An instance of the "Particles" class. It represents the particles
                present in the medium.
            initial_field (InitialField): An object of the `InitialField` class. It represents the
                initial field configuration for the simulation.
        """
        self.wavelength = wavelength
        self.medium_refractive_index = medium_refractive_index
        self.wavelengths_number = wavelength.size
        self.particles = particles
        self.initial_field = initial_field

        self.__setup()

    def __setup(self):
        """The function sets up the necessary computations for omega and ks."""
        self.__compute_omega()
        self.__compute_ks()

    def __compute_omega(self):
        """The function calculates the value of omega using the wavelength."""
        self.omega = 2 * np.pi / self.wavelength

    def __interpolate_refractive_index_from_table(self):
        """Interpolates the refractive index values from a table for different wavelengths.

        Returns:
            refractive_index_interpolated (np.array): An array that contains the interpolated refractive index values for the particles
                at different wavelengths.
        """
        refractive_index_interpolated = np.zeros(
            (self.particles.num_unique_refractive_indices, self.wavelength.size),
            dtype=complex,
        )
        for idx, data in enumerate(self.particles.refractive_index_table):
            table = data["ref_idx"].to_numpy().astype(float)
            n = np.interp(
                self.wavelength / 1e3,
                table[:, 0],
                table[:, 1],
                left=table[0, 1],
                right=table[-1, 1],
            )
            k = np.interp(
                self.wavelength / 1e3,
                table[:, 0],
                table[:, 2],
                left=table[0, 2],
                right=table[-1, 2],
            )
            refractive_index_interpolated[idx, :] = n + 1j * k
        return refractive_index_interpolated

    def __index_to_table(self):
        """
        Todo:
            do all the idx to value conversion here
        """
        pass

    def __compute_ks(self):
        """Computes the values of k_medium and k_particle based on the refractive index of the
        medium and particles.
        """
        self.k_medium = self.omega * self.medium_refractive_index
        if self.particles.refractive_index_table is None:
            self.k_particle = np.outer(self.particles.refractive_index, self.omega)
        else:
            table = self.__interpolate_refractive_index_from_table()
            self.k_particle = (
                np.take(table, self.particles.refractive_index, axis=0)
                * self.omega[np.newaxis, :]
            )

            unique_radius_index_pairs = np.zeros(
                (
                    self.particles.unique_radius_index_pairs.shape[0],
                    self.wavelength.size + 1,
                ),
                dtype=complex,
            )
            unique_radius_index_pairs[:, 0] = self.particles.unique_radius_index_pairs[
                :, 0
            ]
            unique_radius_index_pairs[:, 1:] = np.take(
                table,
                self.particles.unique_radius_index_pairs[:, 1].astype(int),
                axis=0,
            )

            self.particles.unique_radius_index_pairs = unique_radius_index_pairs
