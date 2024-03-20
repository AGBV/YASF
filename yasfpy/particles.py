import yasfpy.log as log

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from yasfpy.functions.material_handler import material_handler


class Particles:
    """The `Particles` class represents a collection of particles with various properties such as position,
    radius, and refractive index, and provides methods for computing unique properties and
    characteristics of the particles.
    """

    def __init__(
        self,
        position: np.array,
        r: np.array,
        refractive_index: np.array,
        refractive_index_table: list = None,
        shape_type: str = "sphere",
    ):
        """Initializes an object with position, radius, refractive index, refractive index table, and shape type attributes.

        Args:
            position (np.array): A numpy array representing the position of the shape.
            r (np.array): A numpy array containing the radius values for each shape in the system.
            refractive_index (np.array): A numpy array representing the refractive index of the shape.
                It can be either a complex number or a two-column matrix.
            refractive_index_table (list): A list containing the refractive index values for different materials.
                Each element in the list represents a material, and the refractive index values for that material are stored as a complex number.
            shape_type (str, optional): A string specifying the type of shape for the object.
                Defaults to "sphere" or any other supported shape type.

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
        """The function `generate_refractive_index_table` takes a list of URLs, retrieves data from each
        URL using the `material_handler` function, and returns a list of the retrieved data.

        Args:
            urls (list): A list of URLs representing different materials.

        Returns:
            data (list): A list of data. Each element in the list corresponds to a URL in the input list,
                and the data is obtained by calling the `material_handler` function on each URL.

        """
        data = [None] * len(urls)
        for k, url in enumerate(urls):
            data[k] = material_handler(url)

        return data

    def compute_unique_refractive_indices(self):
        """Computes the unique refractive indices and their indices."""
        self.unique_refractive_indices, self.refractive_index_array_idx = np.unique(
            self.refractive_index, return_inverse=True, axis=0
        )
        self.num_unique_refractive_indices = self.unique_refractive_indices.shape[0]

    def compute_unique_radii(self):
        """The function computes the unique radii from an array and stores them in a variable."""
        self.unqiue_radii, self.radius_array_idx = np.unique(
            self.r, return_inverse=True, axis=0
        )
        self.num_unique_radii = self.unqiue_radii.shape[0]

    def compute_unique_radii_index_pairs(self):
        """The function computes unique pairs of radii and refractive indices and stores them in different
        arrays.

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
        """The function computes a single unique index based on the sum of pairs of values and their
        corresponding indices.

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
        """The function computes the maximum distance between particles using the ConvexHull algorithm."""
        if len(self.position) < 4:
            if len(self.position) == 1:
                self.max_particle_distance = 0
            else:
                assert len(self.position) == 1
                "SCIPY CONVEX HULL NEEDS 4 POINT MINIMUM; not currently accounted for"

        else:
            hull = ConvexHull(self.position)
            vert = self.position[hull.vertices, :]
            self.max_particle_distance = max(pdist(vert))

    def compute_volume_equivalent_area(self):
        """The function computes the volume equivalent area by calculating the geometric projection."""
        r3 = np.power(self.r, 3)
        self.geometric_projection = np.pi * np.power(np.sum(r3), 2 / 3)

    def __setup_impl(self):
        """The function sets up various computations related to refractive indices, radii, and particle
        distances.

        """
        self.compute_unique_refractive_indices()
        self.compute_unique_radii()
        self.compute_unique_radii_index_pairs()
        self.compute_single_unique_idx()
        self.compute_maximal_particle_distance()
        self.compute_volume_equivalent_area()
