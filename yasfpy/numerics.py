import yasfpy.log as log

from typing import Union, Callable
import numpy as np
import pywigxjpf as wig

from yasfpy.functions.misc import jmult_max
from yasfpy.functions.misc import multi2single_index
from yasfpy.functions.legendre_normalized_trigon import legendre_normalized_trigon


class Numerics:
    """
    The `Numerics` class is used for numerical computations in the YASF (Yet Another Scattering
    Framework) library, providing methods for computing associated Legendre polynomials, translation
    tables, Fibonacci sphere points, and spherical unity vectors.
    """

    def __init__(
        self,
        lmax: int,
        sampling_points_number: Union[int, np.ndarray] = 100,
        polar_angles: np.ndarray = None,
        polar_weight_func: Callable = lambda x: x,
        azimuthal_angles: np.ndarray = None,
        gpu: bool = False,
        particle_distance_resolution=10.0,
        solver=None,
    ):
        """The `__init__` function initializes the Numerics class with various parameters and sets up the
        necessary attributes.

        Args:
            lmax (int): The maximum degree of the spherical harmonics expansion.
            sampling_points_number (Union[int, np.ndarray], optional): The `sampling_points_number` parameter specifies the number of sampling points on the unit
                sphere. It can be either an integer or a numpy array. If it is an integer, it represents the
                total number of sampling points. If it is a numpy array, it can have one or two dimensions. If
            polar_angles (np.ndarray): An array containing the polar angles of the sampling points on the unit sphere.
            polar_weight_func (Callable): The `polar_weight_func` parameter is a callable function that takes a single argument `x` and
                returns a value. This function is used as a weight function for the polar angles of the sampling
                points on the unit sphere. By default, it is set to `lambda x: x`, which
            azimuthal_angles (np.ndarray): An array containing the azimuthal angles of the sampling points on the unit sphere.
            gpu (bool, optional): A flag indicating whether to use GPU acceleration. If set to True, the computations will be
                performed on a GPU if available. If set to False, the computations will be performed on the CPU.
            particle_distance_resolution (float): The parameter "particle_distance_resolution" represents the resolution of the particle
                distance. It determines the accuracy of the numerical computations related to particle distances
                in the code. The value of this parameter is set to 10.0 by default.
            solver (Solver): The `solver` parameter is an optional argument that specifies the solver to use for the
                numerical computations. It is used to solve the scattering problem and obtain the scattering
                amplitudes. If no solver is provided, the default solver will be used.

        """
        self.log = log.scattering_logger(__name__)
        self.lmax = lmax

        self.sampling_points_number = np.squeeze(sampling_points_number)

        if (polar_angles is None) or (azimuthal_angles is None):
            if self.sampling_points_number.size == 0:
                self.sampling_points_number = np.array([100])
                self.log.warning(
                    "Number of sampling points cant be an empty array. Reverting to 100 points (Fibonacci sphere)."
                )
            elif self.sampling_points_number.size > 2:
                self.sampling_points_number = np.array([sampling_points_number[0]])
                self.log.warning(
                    "Number of sampling points with more than two dimensions is not supported. Reverting to the first element in the provided array (Fibonacci sphere)."
                )

            if self.sampling_points_number.size == 1:
                (
                    _,
                    polar_angles,
                    azimuthal_angles,
                ) = Numerics.compute_fibonacci_sphere_points(sampling_points_number[0])
            elif self.sampling_points_number.size == 2:
                # if polar_weight_func is None:
                #   polar_weight_func = lambda x: x
                self.polar_angles_linspace = np.pi * polar_weight_func(
                    np.linspace(0, 1, sampling_points_number[1])
                )
                self.azimuthal_angles_linspace = (
                    2 * np.pi * np.linspace(0, 1, sampling_points_number[0] + 1)[:-1]
                )

                polar_angles, azimuthal_angles = np.meshgrid(
                    self.polar_angles_linspace,
                    self.azimuthal_angles_linspace,
                    indexing="xy",
                )

                polar_angles = polar_angles.ravel()
                azimuthal_angles = azimuthal_angles.ravel()

        else:
            self.sampling_points_number = None

        self.polar_angles = polar_angles
        self.azimuthal_angles = azimuthal_angles
        self.gpu = gpu
        self.particle_distance_resolution = particle_distance_resolution
        self.solver = solver

        if self.gpu:
            from numba import cuda

            if not cuda.is_available():
                self.log.warning(
                    "No supported GPU in numba detected! Falling back to the CPU implementation."
                )
                self.gpu = False

        self.__setup()

    def __compute_nmax(self):
        """
        The function computes the maximum number of coefficients based on the values of lmax.
        """
        self.nmax = 2 * self.lmax * (self.lmax + 2)

    def __plm_coefficients(self):
        """
        The function computes the coefficients for the associated Legendre polynomials using the sympy
        library.
        """
        import sympy as sym

        self.plm_coeff_table = np.zeros(
            (2 * self.lmax + 1, 2 * self.lmax + 1, self.lmax + 1)
        )

        ct = sym.Symbol("ct")
        st = sym.Symbol("st")
        plm = legendre_normalized_trigon(2 * self.lmax, ct, y=st)

        for l in range(2 * self.lmax + 1):
            for m in range(l + 1):
                cf = sym.poly(plm[l, m], ct, st).coeffs()
                self.plm_coeff_table[l, m, 0 : len(cf)] = cf

    def __setup(self):
        """The function performs the setup for numerical computations."""
        self.__compute_nmax()
        # self.compute_translation_table()
        # self.__plm_coefficients()

    def compute_plm_coefficients(self):
        """
        The function computes the coefficients for the associated Legendre polynomials.
        """
        self.__plm_coefficients()

    def compute_translation_table(self):
        """
        The function computes a translation table using Wigner 3j symbols and stores the results in a
        numpy array.
        """
        self.log.scatter("Computing the translation table")
        jmax = jmult_max(1, self.lmax)
        self.translation_ab5 = np.zeros((jmax, jmax, 2 * self.lmax + 1), dtype=complex)

        # No idea why or how this value for max_two_j works,
        # but got it through trial and error.
        # If you get any Wigner errors, change this value (e.g. 3*lmax)
        max_two_j = 3 * self.lmax
        wig.wig_table_init(max_two_j, 3)
        wig.wig_temp_init(max_two_j)

        # Needs to be paralilized or the loop needs to be shortened!
        # Probably using one/two loop(s) and index using the lookup table.
        for tau1 in range(1, 3):
            for l1 in range(1, self.lmax + 1):
                for m1 in range(-l1, l1 + 1):
                    j1 = multi2single_index(0, tau1, l1, m1, self.lmax)
                    for tau2 in range(1, 3):
                        for l2 in range(1, self.lmax + 1):
                            for m2 in range(-l2, l2 + 1):
                                j2 = multi2single_index(0, tau2, l2, m2, self.lmax)
                                for p in range(0, 2 * self.lmax + 1):
                                    if tau1 == tau2:
                                        self.translation_ab5[j1, j2, p] = (
                                            np.power(
                                                1j,
                                                abs(m1 - m2)
                                                - abs(m1)
                                                - abs(m2)
                                                + l2
                                                - l1
                                                + p,
                                            )
                                            * np.power(-1.0, m1 - m2)
                                            * np.sqrt(
                                                (2 * l1 + 1)
                                                * (2 * l2 + 1)
                                                / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))
                                            )
                                            * (
                                                l1 * (l1 + 1)
                                                + l2 * (l2 + 1)
                                                - p * (p + 1)
                                            )
                                            * np.sqrt(2 * p + 1)
                                            * wig.wig3jj_array(
                                                2
                                                * np.array(
                                                    [l1, l2, p, m1, -m2, -m1 + m2]
                                                )
                                            )
                                            * wig.wig3jj_array(
                                                2 * np.array([l1, l2, p, 0, 0, 0])
                                            )
                                        )
                                    elif p > 0:
                                        self.translation_ab5[j1, j2, p] = (
                                            np.power(
                                                1j,
                                                abs(m1 - m2)
                                                - abs(m1)
                                                - abs(m2)
                                                + l2
                                                - l1
                                                + p,
                                            )
                                            * np.power(-1.0, m1 - m2)
                                            * np.sqrt(
                                                (2 * l1 + 1)
                                                * (2 * l2 + 1)
                                                / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))
                                            )
                                            * np.lib.scimath.sqrt(
                                                (l1 + l2 + 1 + p)
                                                * (l1 + l2 + 1 - p)
                                                * (p + l1 - l2)
                                                * (p - l1 + l2)
                                                * (2 * p + 1)
                                            )
                                            * wig.wig3jj_array(
                                                2
                                                * np.array(
                                                    [l1, l2, p, m1, -m2, -m1 + m2]
                                                )
                                            )
                                            * wig.wig3jj_array(
                                                2 * np.array([l1, l2, p - 1, 0, 0, 0])
                                            )
                                        )

        wig.wig_table_free()
        wig.wig_temp_free()

    @staticmethod
    def compute_fibonacci_sphere_points(n: int = 100):
        """Computes the points on a Fibonacci sphere using the given number of points.

        Args:
            n (int, optional): The number of points to be computed on the Fibonacci sphere.
                Defaults to 100.

        Returns:
            tuple (np.ndarray): A tuple containing:
                - points (np.ndarray): The Cartesian points of the Fibonacci sphere.
                - theta (np.ndarray): The polar angles of the points on the Fibonacci sphere.
                - phi (np.ndarray): The azimuthal angles of the points on the Fibonacci sphere.
        """
        golden_ratio = (1 + 5**0.5) / 2
        i = np.arange(0, n)
        phi = 2 * np.pi * (i / golden_ratio % 1)
        theta = np.arccos(1 - 2 * i / n)

        return (
            np.stack(
                (
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ),
                axis=1,
            ),
            theta,
            phi,
        )

    def compute_spherical_unity_vectors(self):
        """
        The function computes the spherical unity vectors e_r, e_theta, and e_phi based on the given
        polar and azimuthal angles.
        """
        self.e_r = np.stack(
            (
                np.sin(self.polar_angles) * np.cos(self.azimuthal_angles),
                np.sin(self.polar_angles) * np.sin(self.azimuthal_angles),
                np.cos(self.polar_angles),
            ),
            axis=1,
        )

        self.e_theta = np.stack(
            (
                np.cos(self.polar_angles) * np.cos(self.azimuthal_angles),
                np.cos(self.polar_angles) * np.sin(self.azimuthal_angles),
                -np.sin(self.polar_angles),
            ),
            axis=1,
        )

        self.e_phi = np.stack(
            (
                -np.sin(self.azimuthal_angles),
                np.cos(self.azimuthal_angles),
                np.zeros_like(self.azimuthal_angles),
            ),
            axis=1,
        )
