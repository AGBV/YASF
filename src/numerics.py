import src.log as log

from typing import Union, Callable
import numpy as np
import pywigxjpf as wig

from src.functions.misc import jmult_max
from src.functions.misc import multi2single_index
from src.functions.legendre_normalized_trigon import legendre_normalized_trigon


class Numerics:
  def __init__(self, lmax: int, sampling_points_number: Union[int, np.ndarray] = 100, polar_angles: np.ndarray = None, polar_weight_func: Callable = lambda x: x, azimuthal_angles: np.ndarray = None, gpu: bool = False, particle_distance_resolution=10.0, solver=None):
    self.log = log.scattering_logger(__name__)
    self.lmax = lmax

    self.sampling_points_number = np.squeeze(sampling_points_number)

    if (polar_angles is None) or (azimuthal_angles is None):
      if self.sampling_points_number.size == 0:
        self.sampling_points_number = np.array([100])
        self.log.warning(
          'Number of sampling points cant be an empty array. Reverting to 100 points (Fibonacci sphere).')
      elif self.sampling_points_number.size > 2:
        self.sampling_points_number = np.array(
          [sampling_points_number[0]])
        self.log.warning(
          'Number of sampling points with more than two dimensions is not supported. Reverting to the first element in the provided array (Fibonacci sphere).')

      if self.sampling_points_number.size == 1:
        _, polar_angles, azimuthal_angles = Numerics.compute_fibonacci_sphere_points(
          sampling_points_number[0])
      elif self.sampling_points_number.size == 2:
        # if polar_weight_func is None:
        #   polar_weight_func = lambda x: x
        self.polar_angles_linspace = np.pi * polar_weight_func(np.linspace(0, 1, sampling_points_number[1]))
        self.azimuthal_angles_linspace =           2 * np.pi * np.linspace(0, 1, sampling_points_number[0] + 1)[:-1]

        polar_angles, azimuthal_angles = np.meshgrid(self.polar_angles_linspace, self.azimuthal_angles_linspace, indexing='xy')

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
          'No supported GPU in numba detected! Falling back to the CPU implementation.')
        self.gpu = False

    self.__setup()

  def __compute_nmax(self):
    self.nmax = 2 * self.lmax * (self.lmax + 2)

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmn.html
  def __plm_coefficients(self):
    import sympy as sym
    self.plm_coeff_table = np.zeros((
      2 * self.lmax + 1,
      2 * self.lmax + 1,
      self.lmax+1))

    ct = sym.Symbol('ct')
    st = sym.Symbol('st')
    plm = legendre_normalized_trigon(2*self.lmax, ct, y=st)

    for l in range(2*self.lmax+1):
      for m in range(l+1):
        cf = sym.poly(plm[l, m], ct, st).coeffs()
        self.plm_coeff_table[l, m, 0:len(cf)] = cf

  def __setup(self):
    self.__compute_nmax()
    # self.compute_translation_table()
    # self.__plm_coefficients()

  def compute_plm_coefficients(self):
    self.__plm_coefficients()

  def compute_translation_table(self):
    self.log.scatter('Computing the translation table')
    jmax = jmult_max(1, self.lmax)
    self.translation_ab5 = np.zeros(
      (jmax, jmax, 2 * self.lmax + 1), dtype=complex)

    # No idea why or how this value for max_two_j works,
    # but got it through trial and error.
    # If you get any Wigner errors, change this value (e.g. 3*lmax)
    max_two_j = 3 * self.lmax
    wig.wig_table_init(max_two_j, 3)
    wig.wig_temp_init(max_two_j)

    # Needs to be paralilized or the loop needs to be shortened!
    # Probably using one/two loop(s) and index using the lookup table.
    for tau1 in range(1, 3):
      for l1 in range(1, self.lmax+1):
        for m1 in range(-l1, l1+1):
          j1 = multi2single_index(0, tau1, l1, m1, self.lmax)
          for tau2 in range(1, 3):
            for l2 in range(1, self.lmax+1):
              for m2 in range(-l2, l2+1):
                j2 = multi2single_index(
                  0, tau2, l2, m2, self.lmax)
                for p in range(0, 2*self.lmax+1):
                  if tau1 == tau2:
                    self.translation_ab5[j1, j2, p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * np.sqrt(2 * p + 1) * \
                      wig.wig3jj_array(2 * np.array([l1, l2, p, m1, -m2, -m1+m2])) * wig.wig3jj_array(
                      2 * np.array([l1, l2, p, 0, 0, 0]))
                  elif p > 0:
                    self.translation_ab5[j1, j2, p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      np.lib.scimath.sqrt((l1 + l2 + 1 + p) * (l1 + l2 + 1 - p) * (p + l1 - l2) * (p - l1 + l2) * (2 * p + 1)) * \
                      wig.wig3jj_array(2 * np.array([l1, l2, p, m1, -m2, -m1+m2])) * wig.wig3jj_array(
                      2 * np.array([l1, l2, p-1, 0, 0, 0]))

    wig.wig_table_free()
    wig.wig_temp_free()

  @staticmethod
  def compute_fibonacci_sphere_points(n=100):
    golden_ratio = (1 + 5**0.5) / 2
    i = np.arange(0, n)
    phi = 2 * np.pi * (i / golden_ratio % 1)
    theta = np.arccos(1 - 2 * i / n)

    return np.stack((
      np.sin(theta) * np.cos(phi),
      np.sin(theta) * np.sin(phi),
      np.cos(theta)), axis=1), theta, phi

  def compute_spherical_unity_vectors(self):
    self.e_r = np.stack((
      np.sin(self.polar_angles) * np.cos(self.azimuthal_angles),
      np.sin(self.polar_angles) * np.sin(self.azimuthal_angles),
      np.cos(self.polar_angles)), axis=1)

    self.e_theta = np.stack((
      np.cos(self.polar_angles) * np.cos(self.azimuthal_angles),
      np.cos(self.polar_angles) * np.sin(self.azimuthal_angles),
      -np.sin(self.polar_angles)), axis=1)

    self.e_phi = np.stack((
      -np.sin(self.azimuthal_angles),
      np.cos(self.azimuthal_angles),
      np.zeros_like(self.azimuthal_angles)), axis=1)
