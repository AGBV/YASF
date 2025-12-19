import math
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy.io import loadmat
from scipy.special import lpmv, spherical_jn, spherical_yn

# Add the project root to the path to allow imports from yasfpy
sys.path.append(str(Path(__file__).parents[1]))

from yasfpy.functions.legendre_normalized_trigon import legendre_normalized_trigon
from yasfpy.functions.misc import (
    multi2single_index,
    mutual_lookup,
    transformation_coefficients,
)
from yasfpy.functions.spherical_functions_trigon import spherical_functions_trigon
from yasfpy.functions.t_entry import t_entry
from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation

# --- Constants and Helpers ---
DATA_PATH = Path(__file__).parent / "data"
RTOL = 1e-5


def _reconstruct_matlab_cell_array(cell_array, lmax, num_angles):
    """Helper to convert Matlab cell array to a 3D Numpy array."""
    if (
        num_angles == 1
        and isinstance(cell_array[0, 0], np.ndarray)
        and cell_array[0, 0].size == 1
    ):
        reconstructed_array = np.zeros((lmax + 1, lmax + 1, num_angles))
        for l in range(lmax + 1):
            for m in range(l + 1):
                reconstructed_array[l, m, 0] = (
                    cell_array[l, m][0, 0] if cell_array[l, m].shape else 0.0
                )
        return reconstructed_array

    reconstructed_array = np.zeros((lmax + 1, lmax + 1, num_angles))
    for l in range(lmax + 1):
        for m in range(l + 1):
            if cell_array[l, m].shape:
                reconstructed_array[l, m, :] = cell_array[l, m].flatten()
    return reconstructed_array


# --- Tests ---


def test_multi2single_index():
    data = loadmat(DATA_PATH / "multi2single_index_data.mat")
    lmax = int(data["lmax_msi"][0, 0])
    inputs = data["inputs_msi"]
    expected_outputs = data["outputs_msi"].flatten()
    for i in range(inputs.shape[0]):
        jS, tau, l, m = inputs[i]
        output = multi2single_index(jS - 1, tau, l, m, lmax)
        assert output == expected_outputs[i] - 1


def test_legendre_normalized_trigon():
    data = loadmat(DATA_PATH / "legendre_normalized_trigon_data.mat")
    lmax = int(data["lmax_lnt"][0, 0])
    theta = data["theta_lnt"].flatten()
    matlab_plm = _reconstruct_matlab_cell_array(data["plm_lnt"], lmax, len(theta))
    python_plm = legendre_normalized_trigon(lmax, theta)
    npt.assert_allclose(python_plm, matlab_plm, rtol=RTOL)


def test_spherical_functions_trigon():
    data = loadmat(DATA_PATH / "spherical_functions_trigon_data.mat")
    lmax = int(data["lmax_sft"][0, 0])
    theta = data["theta_sft"].flatten()
    ct = np.cos(theta)
    st = np.sin(theta)
    matlab_pilm = _reconstruct_matlab_cell_array(data["pilm_sft"], lmax, len(theta))
    matlab_taulm = _reconstruct_matlab_cell_array(data["taulm_sft"], lmax, len(theta))
    python_pilm, python_taulm = spherical_functions_trigon(lmax, ct, st)
    npt.assert_allclose(python_pilm, matlab_pilm, rtol=RTOL)
    npt.assert_allclose(python_taulm, matlab_taulm, rtol=RTOL)


@pytest.mark.parametrize(
    "tau_key, field_type, expected_key",
    [
        ("tau_te_1", "scattered", "output_te_1_scattered"),
        ("tau_te_2", "scattered", "output_te_2_scattered"),
        ("tau_te_1", "internal", "output_te_1_internal"),
        ("tau_te_2", "internal", "output_te_2_internal"),
    ],
)
def test_t_entry(tau_key, field_type, expected_key):
    data = loadmat(DATA_PATH / "t_entry_data.mat")
    l = int(data["l_te"][0, 0])
    kM = float(data["kM_te"][0, 0])
    kS = complex(data["kS_te"][0, 0])
    R = float(data["R_te"][0, 0])
    tau = int(data[tau_key][0, 0])
    expected = data[expected_key].flatten()[0]
    output = t_entry(tau, l, kM, kS, R, field_type)
    npt.assert_allclose(output, expected, rtol=RTOL)


def test_translation_table_ab():
    data = loadmat(DATA_PATH / "translation_table_ab_data.mat")
    lmax = int(data["lmax_tta"][0, 0])
    expected_table = data["translation_table_tta"]["ab5"][0, 0]
    numerics = Numerics(lmax=lmax)
    numerics.compute_translation_table()
    python_table = numerics.translation_ab5
    npt.assert_allclose(python_table, expected_table, rtol=RTOL)


def test_sph_bessel():
    data = loadmat(DATA_PATH / "sph_bessel_data.mat")
    l = int(data["l_sb"][0, 0])
    Z = data["Z_sb"].flatten()
    expected_bessel = data["output_sb_nu1"].flatten()
    python_bessel = spherical_jn(l, Z)
    npt.assert_allclose(python_bessel, expected_bessel, rtol=RTOL)
    expected_hankel = data["output_sb_nu3"].flatten()
    python_hankel = spherical_jn(l, Z) + 1j * spherical_yn(l, Z)
    npt.assert_allclose(python_hankel, expected_hankel, rtol=RTOL)


def test_dx_xz():
    data = loadmat(DATA_PATH / "dx_xz_data.mat")
    l = int(data["l_dx"][0, 0])
    Z = data["Z_dx"].flatten()

    def dx_xz_bessel(l_in, Z_in):
        return Z_in * spherical_jn(l_in - 1, Z_in) - l_in * spherical_jn(l_in, Z_in)

    expected_bessel_deriv = data["output_dx_nu1"].flatten()
    python_bessel_deriv = dx_xz_bessel(l, Z)
    npt.assert_allclose(python_bessel_deriv, expected_bessel_deriv, rtol=RTOL)

    def dx_xz_hankel(l_in, Z_in):
        h_l = spherical_jn(l_in, Z_in) + 1j * spherical_yn(l_in, Z_in)
        h_l_minus_1 = spherical_jn(l_in - 1, Z_in) + 1j * spherical_yn(l_in - 1, Z_in)
        return Z_in * h_l_minus_1 - l_in * h_l

    expected_hankel_deriv = data["output_dx_nu3"].flatten()
    python_hankel_deriv = dx_xz_hankel(l, Z)
    npt.assert_allclose(python_hankel_deriv, expected_hankel_deriv, rtol=RTOL)


@pytest.mark.parametrize(
    "tau_key, pol_key, dagger, expected_key",
    [
        ("tau_tc_1", "pol_tc_1", False, "B_11"),
        ("tau_tc_1", "pol_tc_2", False, "B_12"),
        ("tau_tc_2", "pol_tc_1", False, "B_21"),
        ("tau_tc_2", "pol_tc_2", False, "B_22"),
        ("tau_tc_1", "pol_tc_1", True, "Bd_11"),
        ("tau_tc_1", "pol_tc_2", True, "Bd_12"),
        ("tau_tc_2", "pol_tc_1", True, "Bd_21"),
        ("tau_tc_2", "pol_tc_2", True, "Bd_22"),
    ],
)
def test_transformation_coefficients(tau_key, pol_key, dagger, expected_key):
    data = loadmat(DATA_PATH / "transformation_coefficients_data.mat")
    lmax = int(data["lmax_tc"][0, 0])
    theta = float(data["theta_tc"][0, 0])
    l = int(data["l_tc"][0, 0])
    m = int(data["m_tc"][0, 0])
    tau = int(data[tau_key][0, 0])
    pol = int(data[pol_key][0, 0])
    py_pilm, py_taulm = spherical_functions_trigon(lmax, np.cos(theta), np.sin(theta))
    py_pilm = np.squeeze(py_pilm)
    py_taulm = np.squeeze(py_taulm)
    expected = np.squeeze(data[expected_key])
    output = transformation_coefficients(
        py_pilm, py_taulm, tau, l, m, pol, dagger=dagger
    )
    npt.assert_allclose(output, expected, rtol=RTOL)


def test_coupling_matrix_multiply():
    data = loadmat(DATA_PATH / "coupling_matrix_multiply_data.mat")
    lmax = int(data["lmax"][0, 0])
    wavelength = np.array([float(data["wavelength"][0, 0])])
    medium_refractive_index = np.array([complex(data["mediumRefractiveIndex"][0, 0])])
    particles_position = data["particles_position"].astype(float)
    particles_radius = data["particles_radius"].flatten()
    particles_refractive_index = data["particles_refractiveIndex"].flatten()
    input_x_matlab = data["input_x"].flatten()
    expected_Wx_matlab = data["coupling_matrix_output_Wx"].flatten()

    # MATLAB CUDA stores data in [n, s] order (coefficient outer, particle inner)
    # Python stores data in [s, n] order (particle outer, coefficient inner)
    # We need to transpose the data to match Python's layout
    num_particles = particles_position.shape[0]
    nmax = 2 * lmax * (lmax + 2)

    # Reshape from MATLAB order [nmax, num_particles] to Python order [num_particles, nmax]
    input_x = input_x_matlab.reshape((nmax, num_particles)).T.flatten()
    expected_Wx = expected_Wx_matlab.reshape((nmax, num_particles)).T.flatten()

    particles = Particles(
        position=particles_position,
        r=particles_radius,
        refractive_index=particles_refractive_index,
    )
    initial_field = InitialField(beam_width=np.inf, focal_point=np.array([0, 0, 0]))
    parameters = Parameters(
        wavelength, medium_refractive_index, particles, initial_field
    )
    numerics = Numerics(lmax=lmax, gpu=False)
    numerics.compute_translation_table()
    simulation = Simulation(parameters, numerics)
    python_Wx = simulation.coupling_matrix_multiply(input_x, idx=0)

    # Use relaxed tolerance because MATLAB CUDA uses single precision (float32)
    # while Python uses double precision (float64). For float32, machine epsilon
    # is ~1.2e-7, but accumulated errors in matrix-vector products result in
    # max relative differences of ~1.5e-4. We also need atol for values near zero
    # since the formula is: |actual - expected| <= atol + rtol * |expected|
    npt.assert_allclose(python_Wx, expected_Wx, rtol=2e-4, atol=1e-5)


def test_cpu_gpu_consistency():
    try:
        from numba import cuda

        if not cuda.is_available():
            pytest.skip("GPU not available or not configured correctly with Numba.")
    except ImportError:
        pytest.skip("Numba or CUDA components not installed.")
    lmax = 2
    wavelength = np.array([632.8])
    medium_refractive_index = np.array([1.33])
    particles_position = np.array(
        [[0, 0, 0], [200, 50, -100], [-50, 200, 100]], dtype=float
    )
    particles_radius = np.array([60.0, 65.0, 70.0])
    particles_refractive_index = np.array([1.5 + 0.01j, 1.6 + 0.02j, 1.7 + 0.03j])
    np.random.seed(42)
    num_particles = particles_position.shape[0]
    jmax = 2 * num_particles * lmax * (lmax + 2)
    input_x = np.random.rand(jmax) + 1j * np.random.rand(jmax)
    particles = Particles(
        position=particles_position,
        r=particles_radius,
        refractive_index=particles_refractive_index,
    )
    initial_field = InitialField(beam_width=np.inf, focal_point=np.array([0, 0, 0]))
    parameters = Parameters(
        wavelength, medium_refractive_index, particles, initial_field
    )
    numerics_cpu = Numerics(lmax=lmax, gpu=False)
    numerics_cpu.compute_translation_table()
    simulation_cpu = Simulation(parameters, numerics_cpu)
    output_cpu = simulation_cpu.coupling_matrix_multiply(input_x, idx=0)
    numerics_gpu = Numerics(lmax=lmax, gpu=True)
    numerics_gpu.compute_translation_table()
    simulation_gpu = Simulation(parameters, numerics_gpu)
    output_gpu = simulation_gpu.coupling_matrix_multiply(input_x, idx=0)
    (
        npt.assert_allclose(output_cpu, output_gpu, rtol=RTOL),
        ("CPU and GPU outputs do not match!"),
    )


def test_legendre_implementations_consistency():
    """
    Directly compares the two different implementations of the
    normalized associated Legendre polynomials. This test now asserts
    that they are numerically equivalent for a single angle.
    """
    lmax = 3
    # Use a single, consistent angle for comparison
    theta = np.array([np.pi / 4])

    # Method 1: The yasfpy port of the celes recurrence relation
    plm_recurrence = legendre_normalized_trigon(lmax, theta)

    # Method 2: The implementation from mutual_lookup
    # We need to create dummy inputs to call mutual_lookup that match the theta.
    # mutual_lookup computes: differences = positions_1 - positions_2
    # To get a vector with polar angle theta, we set up:
    #   pos1 = unit vector at angle theta = [sin(theta), 0, cos(theta)]
    #   pos2 = origin = [0, 0, 0]
    # So differences = pos1 - pos2 = [sin(theta), 0, cos(theta)]
    # which has cos(polar_angle) = cos(theta) as desired.
    pos1 = np.array([[np.sin(theta[0]), 0.0, np.cos(theta[0])]])
    pos2 = np.array([[0.0, 0.0, 0.0]])

    # The 'p' in mutual_lookup's plm goes up to 2*lmax, while the 'l' in
    # legendre_normalized_trigon goes up to lmax. We need to tell mutual_lookup
    # to compute up to lmax for the 'p' part of plm to compare apples to apples.
    # However, mutual_lookup is hardcoded to compute up to 2*lmax for translations.
    # The test is fundamentally comparing a subset of the translation table's plm
    # with a field expansion's plm. They should be identical for the same (l,m,theta).

    _, _, _, plm_mutual_lookup, _, _, _, _, _, _, _ = mutual_lookup(
        lmax, pos1, pos2, np.array([1.0])
    )

    # The plm from mutual_lookup is flattened. We need to reshape it for comparison.
    # It has P_p^m for p up to 2*lmax. We only need up to lmax.
    plm_mutual_lookup_reshaped = np.zeros_like(plm_recurrence)
    for l in range(lmax + 1):
        for m in range(l + 1):
            # The index into the flattened plm from mutual_lookup
            idx = l * (l + 1) // 2 + m
            if idx < plm_mutual_lookup.shape[0]:
                plm_mutual_lookup_reshaped[l, m, 0] = plm_mutual_lookup[idx, 0, 0]

    npt.assert_allclose(plm_recurrence, plm_mutual_lookup_reshaped, rtol=RTOL)
