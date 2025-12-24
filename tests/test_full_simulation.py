"""
End-to-end tests for full light scattering simulation.

This module tests the complete simulation workflow from initial field coefficients
through scattered field coefficients to final electric and magnetic field computation.
Tests compare Python implementation against MATLAB reference data.
"""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.io as sio

from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.optics import Optics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver


@pytest.fixture
def matlab_data():
    """Load MATLAB reference data for full simulation."""
    return sio.loadmat("tests/data/full_simulation_data.mat")


@pytest.fixture
def config_data():
    """Load test configuration."""
    return sio.loadmat("tests/data/test_config.mat")


def setup_simulation(matlab_data):
    """Set up Python simulation from MATLAB test data."""
    # Extract parameters
    lmax = int(matlab_data["lmax"][0, 0])
    wavelength = np.array([float(matlab_data["wavelength"][0, 0])])
    medium_refractive_index = np.array([float(matlab_data["mediumRefractiveIndex"][0, 0])])

    # Particle data
    positions = matlab_data["particles_position"]
    radii = matlab_data["particles_radius"].flatten()
    refractive_indices = matlab_data["particles_refractiveIndex"].flatten()

    # Create particle object
    particles = Particles(
        position=positions,
        r=radii,
        refractive_index=refractive_indices,
    )

    # Create initial field (plane wave, TE polarization, normal incidence)
    initial_field = InitialField(
        beam_width=0,  # Plane wave
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="TE",
    )

    # Create parameters
    parameters = Parameters(
        wavelength=wavelength,
        medium_refractive_index=medium_refractive_index,
        particles=particles,
        initial_field=initial_field,
    )

    # Create solver (for scattered field coefficient computation)
    solver = Solver(
        solver_type="gmres",
        tolerance=1e-4,
        max_iter=1000,
        restart=1000,
    )

    # Create numerics
    numerics = Numerics(
        lmax=lmax,
        sampling_points_number=np.array([100]),  # Default value
        particle_distance_resolution=1,
        gpu=False,
        solver=solver,
    )

    # Compute translation table
    numerics.compute_translation_table()

    # Create simulation
    simulation = Simulation(parameters, numerics)

    return simulation


def test_initial_field_coefficients(matlab_data):
    """Test initial field coefficient computation."""
    simulation = setup_simulation(matlab_data)

    # Compute initial field coefficients
    simulation.compute_initial_field_coefficients()

    # Get Python results (squeeze out wavelength dimension since we only have one wavelength)
    python_coeffs = simulation.initial_field_coefficients.squeeze()

    # Get MATLAB reference
    # MATLAB stores as [particles, coefficients] in row-major
    matlab_coeffs = matlab_data["initial_field_coefficients"]

    # Account for potential memory layout differences
    # Python uses [particles, coefficients], MATLAB might too
    # Check shapes first
    assert python_coeffs.shape == matlab_coeffs.shape, (
        f"Shape mismatch: Python {python_coeffs.shape} vs MATLAB {matlab_coeffs.shape}"
    )

    # Compare with appropriate tolerances for complex numbers
    npt.assert_allclose(
        python_coeffs,
        matlab_coeffs,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Initial field coefficients mismatch",
    )


def test_mie_coefficients(matlab_data):
    """Test Mie coefficient (T-matrix) computation."""
    simulation = setup_simulation(matlab_data)

    # Compute Mie coefficients
    simulation.compute_mie_coefficients()

    # Get Python results (squeeze out wavelength dimension)
    python_mie = simulation.mie_coefficients.squeeze()

    # Get MATLAB reference
    matlab_mie = matlab_data["mie_coefficients"]

    # Compare
    npt.assert_allclose(
        python_mie,
        matlab_mie,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Mie coefficients mismatch",
    )


def test_right_hand_side(matlab_data):
    """Test right-hand side computation (T * initial_field)."""
    simulation = setup_simulation(matlab_data)

    # Compute prerequisites
    simulation.compute_initial_field_coefficients()
    simulation.compute_mie_coefficients()
    simulation.compute_right_hand_side()

    # Get Python results (squeeze out wavelength dimension)
    python_rhs = simulation.right_hand_side.squeeze()

    # Compute MATLAB reference from components
    # rhs = T * aI (element-wise multiplication)
    matlab_initial = matlab_data["initial_field_coefficients"]
    matlab_mie = matlab_data["mie_coefficients"]
    matlab_single_unique = matlab_data["single_unique_array_index"].flatten() - 1  # 0-indexed

    # Apply Mie coefficients to initial field
    matlab_rhs = matlab_mie[matlab_single_unique, :] * matlab_initial

    # Compare
    npt.assert_allclose(
        python_rhs,
        matlab_rhs,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Right-hand side mismatch",
    )


def test_scattered_field_coefficients(matlab_data):
    """Test scattered field coefficient computation (solution of linear system)."""
    simulation = setup_simulation(matlab_data)

    # Compute full workflow up to scattered field coefficients
    simulation.compute_mie_coefficients()
    simulation.compute_initial_field_coefficients()
    simulation.compute_right_hand_side()
    simulation.compute_scattered_field_coefficients()

    # Get Python results (squeeze out wavelength dimension)
    python_scattered = simulation.scattered_field_coefficients.squeeze()

    # Get MATLAB reference
    matlab_scattered = matlab_data["scattered_field_coefficients"]

    # The iterative solver may produce slightly different results
    # Use relaxed tolerance
    npt.assert_allclose(
        python_scattered,
        matlab_scattered,
        rtol=1e-3,  # Relaxed tolerance for iterative solver
        atol=1e-5,
        err_msg="Scattered field coefficients mismatch",
    )


def test_electric_field_scattered(matlab_data):
    """Test scattered electric field computation at points."""
    simulation = setup_simulation(matlab_data)

    # Run full simulation
    simulation.compute_mie_coefficients()
    simulation.compute_initial_field_coefficients()
    simulation.compute_right_hand_side()
    simulation.compute_scattered_field_coefficients()

    # Get field points
    field_points = matlab_data["field_points"]

    # Compute fields
    simulation.compute_fields(field_points)

    # Get Python results (squeeze out wavelength dimension)
    python_E_scattered = simulation.scattered_field.squeeze()

    # Get MATLAB reference
    matlab_E_scattered = matlab_data["E_scattered"]

    # Compare (may need memory layout adjustments)
    npt.assert_allclose(
        python_E_scattered,
        matlab_E_scattered,
        rtol=1e-3,
        atol=1e-5,
        err_msg="Scattered electric field mismatch",
    )


def test_electric_field_initial(matlab_data):
    """Test initial electric field computation at points."""
    simulation = setup_simulation(matlab_data)

    # Run full simulation first (needed for compute_fields to work)
    simulation.compute_mie_coefficients()
    simulation.compute_initial_field_coefficients()
    simulation.compute_right_hand_side()
    simulation.compute_scattered_field_coefficients()

    # Get field points
    field_points = matlab_data["field_points"]

    # Compute fields (this computes both initial and scattered)
    simulation.compute_fields(field_points)

    # Get Python results (squeeze out wavelength dimension)
    python_E_initial = simulation.initial_field_electric.squeeze()

    # Get MATLAB reference
    matlab_E_initial = matlab_data["E_initial"]

    # Compare
    npt.assert_allclose(
        python_E_initial,
        matlab_E_initial,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Initial electric field mismatch",
    )


def test_simulation_workflow(matlab_data, config_data):
    """Integration test for complete simulation workflow."""
    # This test ensures all components work together
    simulation = setup_simulation(matlab_data)

    # Run the full workflow
    simulation.compute_mie_coefficients()
    simulation.compute_initial_field_coefficients()
    simulation.compute_right_hand_side()
    simulation.compute_scattered_field_coefficients()

    # Verify we got results
    assert simulation.initial_field_coefficients is not None
    assert simulation.mie_coefficients is not None
    assert simulation.right_hand_side is not None
    assert simulation.scattered_field_coefficients is not None

    # Verify shapes are consistent
    num_particles = matlab_data["particles_position"].shape[0]
    lmax = int(matlab_data["lmax"][0, 0])
    nmax = 2 * lmax * (lmax + 2)

    # Python has shape (particles, nmax, wavelengths) with wavelengths=1
    assert simulation.initial_field_coefficients.shape == (num_particles, nmax, 1)
    assert simulation.scattered_field_coefficients.shape == (num_particles, nmax, 1)

    # Get test size info
    test_size = str(config_data["test_size"][0])
    print(f"\nSuccessfully ran {test_size} test case:")
    print(f"  - {num_particles} particles")
    print(f"  - lmax = {lmax}")
    print(f"  - {nmax} coefficients per particle")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
