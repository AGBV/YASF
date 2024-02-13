import yasfpy.log as log

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from yasfpy.functions.material_handler import material_handler


class Particles:
    """
    Class representing a collection of particles.

    Args:
        position (np.array): Array of particle positions.
        r (np.array): Array of particle radii.
        refractive_index (np.array): Array of particle refractive indices.
        refractive_index_table (list, optional): List of refractive index tables. Defaults to None.
        shape_type (str, optional): Type of particle shape. Defaults to "sphere".

    Attributes:
        position (np.array): Array of particle positions.
        r (np.array): Array of particle radii.
        refractive_index (np.array): Array of particle refractive indices.
        type (str): Type of particle shape.
        refractive_index_table (list): List of refractive index tables.
        number (int): Number of particles.
        log: Logging object.
        unique_refractive_indices (np.array): Array of unique refractive indices.
        refractive_index_array_idx (np.array): Array of indices mapping refractive indices to unique indices.
        num_unique_refractive_indices (int): Number of unique refractive indices.
        unqiue_radii (np.array): Array of unique radii.
        radius_array_idx (np.array): Array of indices mapping radii to unique indices.
        num_unique_radii (int): Number of unique radii.
        unique_radius_index_pairs (np.array): Array of unique radius-index pairs.
        single_unique_array_idx (np.array): Array of indices mapping unique radius-index pairs to unique indices.
        unique_single_radius_index_pairs (np.array): Array of unique single radius-index pairs.
        single_unique_idx (np.array): Array of single unique indices.
        num_unique_pairs (int): Number of unique radius-index pairs.
        max_particle_distance (float): Maximum distance between particles.
        geometric_projection (float): Geometric projection of the particles.

    Methods:
        generate_refractive_index_table(urls): Generates a refractive index table from a list of URLs.
        compute_unique_refractive_indices(): Computes the unique refractive indices and their indices.
        compute_unique_radii(): Computes the unique radii and their indices.
        compute_unique_radii_index_pairs(): Computes the unique radius-index pairs.
        compute_single_unique_idx(): Computes the single unique indices.
        compute_maximal_particle_distance(): Computes the maximum distance between particles.
        compute_volume_equivalent_area(): Computes the volume equivalent area.
        __setup_impl(): Helper method to set up the class.

    """

    def __init__(
        self,
        position: np.array,
        r: np.array,
        refractive_index: np.array,
        refractive_index_table: list = None,
        shape_type: str = "sphere",
    ):
        """
        Initialize a Particle object.

        Args:
            position (np.array): The position of the particle.
            r (np.array): The radius of the particle.
            refractive_index (np.array): The refractive index of the particle.
            refractive_index_table (list, optional): A table of refractive indices for different wavelengths. Defaults to None.
            shape_type (str, optional): The shape type of the particle. Defaults to "sphere".
        """
        self.position = position
        self.r = r
        self.refractive_index = refractive_index
        self.type = shape_type

        self.log = log.scattering_logger(__name__)

        # TODO: Keep it for now, remove later...
        self.refractive_index_table = refractive_index_table

        if refractive_index_table is None:
            if self.refractive_index.shape[1] == 2:
                self.refractive_index = (
                    self.refractive_index[:, 0] + 1j * self.refractive_index[:, 1]
                )
        elif self.refractive_index.shape[1] > 2:
            self.log.error(
                "Refractive index should be either complex or a two column matrix!"
            )
        else:
            self.refractive_index = refractive_index.astype(int)
            self.refractive_index_table = refractive_index_table

        self.number = r.shape[0]
        self.__setup_impl()

    @staticmethod
    def generate_refractive_index_table(urls: list):
        """
        Generates a refractive index table from a list of URLs.

        Args:
            urls (list): List of URLs.

        Returns:
            list: List of refractive index tables.

        """
        data = [None] * len(urls)
        for k, url in enumerate(urls):
            data[k] = material_handler(url)

        return data

    def compute_unique_refractive_indices(self):
        """
        Computes the unique refractive indices and their indices.

        """
        self.unique_refractive_indices, self.refractive_index_array_idx = np.unique(
            self.refractive_index, return_inverse=True, axis=0
        )
        self.num_unique_refractive_indices = self.unique_refractive_indices.shape[0]

    def compute_unique_radii(self):
        """
        Computes the unique radii and their indices.

        """
        self.unqiue_radii, self.radius_array_idx = np.unique(
            self.r, return_inverse=True, axis=0
        )
        self.num_unique_radii = self.unqiue_radii.shape[0]

    def compute_unique_radii_index_pairs(self):
        """
        Computes the unique radius-index pairs.

        """
        self.unique_radius_index_pairs, self.single_unique_array_idx = np.unique(
            np.column_stack((self.r, self.refractive_index)),
            return_inverse=True,
            axis=0,
        )
        self.unique_single_radius_index_pairs = np.unique(
            np.column_stack((self.radius_array_idx, self.refractive_index_array_idx)),
            axis=0,
        )

    def compute_single_unique_idx(self):
        """
        Computes the single unique indices.

        """
        self.single_unique_idx = (
            np.sum(self.unique_single_radius_index_pairs, axis=1)
            * (np.sum(self.unique_single_radius_index_pairs, axis=1) + 1)
        ) // 2 + self.unique_single_radius_index_pairs[:, 1]

        # pairedArray = (
        #   self.radius_array_idx + self.refractive_index_array_idx *
        #   (self.radius_array_idx + self.refractive_index_array_idx + 1)
        # ) // 2 + self.refractive_index_array_idx

        # self.single_unique_idx, self.single_unique_array_idx = np.unique(
        #   pairedArray,
        #   return_inverse=True,
        #   axis=0)

        self.num_unique_pairs = self.unique_radius_index_pairs.shape[0]

    def compute_maximal_particle_distance(self):
        """
        Computes the maximum distance between particles.

        """
        hull = ConvexHull(self.position)
        vert = self.position[hull.vertices, :]
        self.max_particle_distance = max(pdist(vert))

    def compute_volume_equivalent_area(self):
        """
        Computes the volume equivalent area.

        """
        r3 = np.power(self.r, 3)
        self.geometric_projection = np.pi * np.power(np.sum(r3), 2 / 3)

    def __setup_impl(self):
        """
        Helper method to set up the class.

        """
        self.compute_unique_refractive_indices()
        self.compute_unique_radii()
        self.compute_unique_radii_index_pairs()
        self.compute_single_unique_idx()
        self.compute_maximal_particle_distance()
        self.compute_volume_equivalent_area()
