import numpy as np

from yasfpy.particles import Particles
from yasfpy.initial_field import InitialField


import numpy as np


class Parameters:
    """
    Class representing the parameters for a simulation.

    Args:
        wavelength (np.array): Array of wavelengths.
        medium_refractive_index (np.array): Array of refractive indices for the medium.
        particles (Particles): Instance of the Particles class.
        initial_field (InitialField): Instance of the InitialField class.
    """

    def __init__(
        self,
        wavelength: np.array,
        medium_refractive_index: np.array,
        particles: Particles,
        initial_field: InitialField,
    ):
        """
        Initialize the Parameters object.

        Args:
            wavelength (np.array): Array of wavelengths.
            medium_refractive_index (np.array): Array of refractive indices for the medium.
            particles (Particles): Particles object representing the scattering particles.
            initial_field (InitialField): InitialField object representing the initial field.

        Returns:
            None
        """
        self.wavelength = wavelength
        self.medium_refractive_index = medium_refractive_index
        self.wavelengths_number = wavelength.size
        self.particles = particles
        self.initial_field = initial_field

        self.__setup()

    def __setup(self):
        """
        Performs the setup operations for the object.
        This method computes the omega and ks values.
        """
        self.__compute_omega()
        self.__compute_ks()

    def __compute_omega(self):
        """
        Compute the angular frequency (omega) based on the wavelength.

        The angular frequency is calculated as 2 * pi divided by the wavelength.

        Parameters:
            None

        Returns:
            None
        """
        self.omega = 2 * np.pi / self.wavelength

    def __interpolate_refractive_index_from_table(self):
        """
        Interpolates the refractive index from a table.

        Returns:
            numpy.ndarray: An array of interpolated refractive indices.
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
        # TODO: do all the idx to value conversion here
        pass

    def __compute_ks(self):
        """
        Compute the wave vectors for the medium and particles.

        This method calculates the wave vectors for the medium and particles based on their refractive indices and the angular frequency.

        Returns:
            None
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
