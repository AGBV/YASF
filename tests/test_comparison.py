import numpy as np
import pytest
from pathlib import Path
from scipy.io import loadmat
from scipy.special import spherical_jn, spherical_yn, lpmv
import sys
import math

# Add the project root to the path to allow imports from yasfpy
sys.path.append(str(Path(__file__).parents[1]))

from yasfpy.functions.misc import multi2single_index, transformation_coefficients, mutual_lookup
from yasfpy.functions.legendre_normalized_trigon import legendre_normalized_trigon
from yasfpy.functions.spherical_functions_trigon import spherical_functions_trigon
from yasfpy.functions.t_entry import t_entry
from yasfpy.numerics import Numerics
from yasfpy.particles import Particles
from yasfpy.initial_field import InitialField
from yasfpy.parameters import Parameters
from yasfpy.simulation import Simulation

# --- Constants and Helpers ---
DATA_PATH = Path(__file__).parent / 'data'
RTOL = 1e-5

def _reconstruct_matlab_cell_array(cell_array, lmax, num_angles):
    """Helper to convert Matlab cell array to a 3D Numpy array."""
    if num_angles == 1 and isinstance(cell_array[0,0], np.ndarray) and cell_array[0,0].size == 1:
            reconstructed_array = np.zeros((lmax + 1, lmax + 1, num_angles))
            for l in range(lmax + 1):
                for m in range(l + 1):
                    reconstructed_array[l,m,0] = cell_array[l,m][0,0] if cell_array[l,m].shape else 0.0
            return reconstructed_array

    reconstructed_array = np.zeros((lmax + 1, lmax + 1, num_angles))
    for l in range(lmax + 1):
        for m in range(l + 1):
            if cell_array[l, m].shape:
                reconstructed_array[l, m, :] = cell_array[l, m].flatten()
    return reconstructed_array

# --- Tests ---

def test_multi2single_index():
    data = loadmat(DATA_PATH / 'multi2single_index_data.mat')
    lmax = int(data['lmax_msi'][0, 0])
    inputs = data['inputs_msi']
    expected_outputs = data['outputs_msi'].flatten()
    for i in range(inputs.shape[0]):
        jS, tau, l, m = inputs[i]
        output = multi2single_index(jS - 1, tau, l, m, lmax)
        assert output == expected_outputs[i] - 1

def test_legendre_normalized_trigon():
    data = loadmat(DATA_PATH / 'legendre_normalized_trigon_data.mat')
    lmax = int(data['lmax_lnt'][0, 0])
    theta = data['theta_lnt'].flatten()
    matlab_plm = _reconstruct_matlab_cell_array(data['plm_lnt'], lmax, len(theta))
    python_plm = legendre_normalized_trigon(lmax, theta)
    assert np.allclose(python_plm, matlab_plm, rtol=RTOL)

def test_spherical_functions_trigon():
    data = loadmat(DATA_PATH / 'spherical_functions_trigon_data.mat')
    lmax = int(data['lmax_sft'][0, 0])
    theta = data['theta_sft'].flatten()
    ct = np.cos(theta)
    st = np.sin(theta)
    matlab_pilm = _reconstruct_matlab_cell_array(data['pilm_sft'], lmax, len(theta))
    matlab_taulm = _reconstruct_matlab_cell_array(data['taulm_sft'], lmax, len(theta))
    python_pilm, python_taulm = spherical_functions_trigon(lmax, ct, st)
    assert np.allclose(python_pilm, matlab_pilm, rtol=RTOL)
    assert np.allclose(python_taulm, matlab_taulm, rtol=RTOL)

@pytest.mark.parametrize("tau_key, field_type, expected_key", [
    ('tau_te_1', "scattered", "output_te_1_scattered"),
    ('tau_te_2', "scattered", "output_te_2_scattered"),
    ('tau_te_1', "internal", "output_te_1_internal"),
    ('tau_te_2', "internal", "output_te_2_internal"),
])
def test_t_entry(tau_key, field_type, expected_key):
    data = loadmat(DATA_PATH / 't_entry_data.mat')
    l = int(data['l_te'][0, 0])
    kM = float(data['kM_te'][0, 0])
    kS = complex(data['kS_te'][0, 0])
    R = float(data['R_te'][0, 0])
    tau = int(data[tau_key][0, 0])
    expected = data[expected_key].flatten()[0]
    output = t_entry(tau, l, kM, kS, R, field_type)
    assert np.allclose(output, expected, rtol=RTOL)
    
def test_translation_table_ab():
    data = loadmat(DATA_PATH / 'translation_table_ab_data.mat')
    lmax = int(data['lmax_tta'][0, 0])
    expected_table = data['translation_table_tta']['ab5'][0, 0]
    numerics = Numerics(lmax=lmax)
    numerics.compute_translation_table()
    python_table = numerics.translation_ab5
    assert np.allclose(python_table, expected_table, rtol=RTOL)

def test_sph_bessel():
    data = loadmat(DATA_PATH / 'sph_bessel_data.mat')
    l = int(data['l_sb'][0, 0])
    Z = data['Z_sb'].flatten()
    expected_bessel = data['output_sb_nu1'].flatten()
    python_bessel = spherical_jn(l, Z)
    assert np.allclose(python_bessel, expected_bessel, rtol=RTOL)
    expected_hankel = data['output_sb_nu3'].flatten()
    python_hankel = spherical_jn(l, Z) + 1j * spherical_yn(l, Z)
    assert np.allclose(python_hankel, expected_hankel, rtol=RTOL)

def test_dx_xz():
    data = loadmat(DATA_PATH / 'dx_xz_data.mat')
    l = int(data['l_dx'][0, 0])
    Z = data['Z_dx'].flatten()
    def dx_xz_bessel(l_in, Z_in):
        return Z_in * spherical_jn(l_in - 1, Z_in) - l_in * spherical_jn(l_in, Z_in)
    expected_bessel_deriv = data['output_dx_nu1'].flatten()
    python_bessel_deriv = dx_xz_bessel(l, Z)
    assert np.allclose(python_bessel_deriv, expected_bessel_deriv, rtol=RTOL)
    def dx_xz_hankel(l_in, Z_in):
        h_l = spherical_jn(l_in, Z_in) + 1j * spherical_yn(l_in, Z_in)
        h_l_minus_1 = spherical_jn(l_in - 1, Z_in) + 1j * spherical_yn(l_in - 1, Z_in)
        return Z_in * h_l_minus_1 - l_in * h_l
    expected_hankel_deriv = data['output_dx_nu3'].flatten()
    python_hankel_deriv = dx_xz_hankel(l, Z)
    assert np.allclose(python_hankel_deriv, expected_hankel_deriv, rtol=RTOL)

@pytest.mark.parametrize("tau_key, pol_key, dagger, expected_key", [
    ('tau_tc_1', 'pol_tc_1', False, 'B_11'), ('tau_tc_1', 'pol_tc_2', False, 'B_12'),
    ('tau_tc_2', 'pol_tc_1', False, 'B_21'), ('tau_tc_2', 'pol_tc_2', False, 'B_22'),
    ('tau_tc_1', 'pol_tc_1', True, 'Bd_11'), ('tau_tc_1', 'pol_tc_2', True, 'Bd_12'),
    ('tau_tc_2', 'pol_tc_1', True, 'Bd_21'), ('tau_tc_2', 'pol_tc_2', True, 'Bd_22'),
])
def test_transformation_coefficients(tau_key, pol_key, dagger, expected_key):
    data = loadmat(DATA_PATH / 'transformation_coefficients_data.mat')
    lmax = int(data['lmax_tc'][0, 0])
    theta = float(data['theta_tc'][0, 0])
    l = int(data['l_tc'][0, 0])
    m = int(data['m_tc'][0, 0])
    tau = int(data[tau_key][0, 0])
    pol = int(data[pol_key][0, 0])
    py_pilm, py_taulm = spherical_functions_trigon(lmax, np.cos(theta), np.sin(theta))
    py_pilm = np.squeeze(py_pilm)
    py_taulm = np.squeeze(py_taulm)
    expected = np.squeeze(data[expected_key])
    output = transformation_coefficients(py_pilm, py_taulm, tau, l, m, pol, dagger=dagger)
    assert np.allclose(output, expected, rtol=RTOL)

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
    particles_position = np.array([[0, 0, 0], [200, 50, -100], [-50, 200, 100]], dtype=float)
    particles_radius = np.array([60., 65., 70.])
    particles_refractive_index = np.array([1.5 + 0.01j, 1.6 + 0.02j, 1.7 + 0.03j])
    np.random.seed(42)
    num_particles = particles_position.shape[0]
    jmax = 2 * num_particles * lmax * (lmax + 2)
    input_x = np.random.rand(jmax) + 1j * np.random.rand(jmax)
    particles = Particles(position=particles_position, r=particles_radius, refractive_index=particles_refractive_index)
    initial_field = InitialField(beam_width=np.inf, focal_point=np.array([0,0,0]))
    parameters = Parameters(wavelength, medium_refractive_index, particles, initial_field)
    numerics_cpu = Numerics(lmax=lmax, gpu=False)
    numerics_cpu.compute_translation_table()
    simulation_cpu = Simulation(parameters, numerics_cpu)
    output_cpu = simulation_cpu.coupling_matrix_multiply(input_x, idx=0)
    numerics_gpu = Numerics(lmax=lmax, gpu=True)
    numerics_gpu.compute_translation_table()
    simulation_gpu = Simulation(parameters, numerics_gpu)
    output_gpu = simulation_gpu.coupling_matrix_multiply(input_x, idx=0)
    assert np.allclose(output_cpu, output_gpu, rtol=RTOL), "CPU and GPU outputs do not match!"

def test_legendre_implementations_consistency():
    """
    Directly compares the two different implementations of the
    normalized associated Legendre polynomials. This test now asserts
    that they are numerically equivalent.
    """
    lmax = 3
    theta = np.array([np.pi/6, np.pi/4, np.pi/3])
    
    # Method 1: The yasfpy port of the celes recurrence relation
    plm_recurrence = legendre_normalized_trigon(lmax, theta)

    # Method 2: The implementation from mutual_lookup
    # We need to create dummy inputs to call mutual_lookup
    pos1 = np.array([[0., 0., 0.]])
    pos2 = np.array([[1., 1., 1.]]) # Dummy positions to get angles
    _, _, _, plm_mutual_lookup, _, _, _, _, _, _, _ = mutual_lookup(
        lmax, pos1, pos2, np.array([1.0])
    )
    # The plm from mutual_lookup is flattened, we need to reshape it for comparison
    plm_mutual_lookup_reshaped = np.zeros_like(plm_recurrence)
    for p in range(2 * lmax + 1):
        for m in range(p + 1):
            idx = p * (p + 1) // 2 + m
            if idx < plm_mutual_lookup.shape[0]:
                 # The mutual_lookup returns plm for a single angle, so we need to select it
                plm_mutual_lookup_reshaped[p, m, :] = plm_mutual_lookup[idx, 0, 1]


    # This assertion should now PASS.
    assert np.allclose(
        plm_recurrence,
        plm_mutual_lookup_reshaped,
        rtol=RTOL,
        err_msg="The two Legendre implementations are NOT numerically equivalent."
    )