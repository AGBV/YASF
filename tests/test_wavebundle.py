"""
Test wavebundle (Gaussian beam) initial field computation.

This test validates the implementation of Gaussian beam expansion in
spherical vector wave functions for normal incidence.
"""

import numpy as np
import numpy.testing as npt
import pytest

from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation


def test_wavebundle_normal_incidence_basic():
    """
    Test that wavebundle computation runs and produces non-NaN results.

    This is a basic smoke test to ensure the implementation doesn't crash
    and produces valid output.
    """
    # Create a simple particle system
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 100.0],
        ]
    )
    radii = np.array([50.0, 50.0, 50.0])
    refractive_indices = np.array([1.5 + 0.01j, 1.5 + 0.01j, 1.5 + 0.01j])

    particles = Particles(
        position=positions,
        r=radii,
        refractive_index=refractive_indices,
    )

    # Create Gaussian beam at normal incidence
    # beam_width > 0 triggers Gaussian beam computation
    initial_field = InitialField(
        beam_width=500.0,  # Gaussian beam with 500nm width
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,  # Normal incidence
        azimuthal_angle=0.0,
        polarization="TE",
    )

    # Create parameters
    wavelength = np.array([550.0])  # 550nm
    medium_refractive_index = np.array([1.0])

    parameters = Parameters(
        wavelength=wavelength,
        medium_refractive_index=medium_refractive_index,
        particles=particles,
        initial_field=initial_field,
    )

    # Create numerics with polar angle grid
    # Use 2D sampling grid: [azimuthal_points, polar_points]
    numerics = Numerics(
        lmax=3,
        sampling_points_number=np.array([10, 20]),  # 10 azimuthal, 20 polar
        particle_distance_resolution=1,
        gpu=False,
    )

    # Create simulation
    simulation = Simulation(parameters, numerics)

    # Compute Mie coefficients
    simulation.compute_mie_coefficients()

    # Compute initial field coefficients (Gaussian beam)
    simulation.compute_initial_field_coefficients()

    # Check that coefficients are computed
    assert simulation.initial_field_coefficients is not None
    assert simulation.initial_field_coefficients.shape == (3, numerics.nmax, 1)

    # Check no NaN values
    assert not np.any(np.isnan(simulation.initial_field_coefficients))

    # Check coefficients are non-zero
    assert np.any(np.abs(simulation.initial_field_coefficients) > 0)

    print(f"✓ Gaussian beam coefficients computed successfully")
    print(
        f"  Max coefficient magnitude: {np.max(np.abs(simulation.initial_field_coefficients)):.3e}"
    )


def test_wavebundle_beam_focusing():
    """
    Test that Gaussian beam shows expected focusing behavior.

    A focused beam should have stronger coefficients for particles near
    the focal point compared to those far from it.
    """
    # Create particles at different positions relative to focal point
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # At focal point
            [0.0, 0.0, 500.0],  # Far from focal point
        ]
    )
    radii = np.array([50.0, 50.0])
    refractive_indices = np.array([1.5 + 0.01j, 1.5 + 0.01j])

    particles = Particles(
        position=positions,
        r=radii,
        refractive_index=refractive_indices,
    )

    wavelength = np.array([550.0])
    medium_refractive_index = np.array([1.0])

    # Numerics with polar angle grid
    numerics = Numerics(
        lmax=2,
        sampling_points_number=np.array([10, 30]),
        particle_distance_resolution=1,
        gpu=False,
    )

    # Focused Gaussian beam
    initial_field = InitialField(
        beam_width=300.0,  # Moderately focused beam
        focal_point=np.array([0.0, 0.0, 0.0]),  # Focus at origin
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="TE",
    )

    parameters = Parameters(
        wavelength=wavelength,
        medium_refractive_index=medium_refractive_index,
        particles=particles,
        initial_field=initial_field,
    )

    simulation = Simulation(parameters, numerics)
    simulation.compute_mie_coefficients()
    simulation.compute_initial_field_coefficients()

    # Coefficients for particle at focal point vs far away
    coeffs_at_focus = simulation.initial_field_coefficients[0, :, 0]
    coeffs_far = simulation.initial_field_coefficients[1, :, 0]

    mag_at_focus = np.linalg.norm(coeffs_at_focus)
    mag_far = np.linalg.norm(coeffs_far)

    print(f"✓ Gaussian beam focusing:")
    print(f"  Magnitude at focal point: {mag_at_focus:.3e}")
    print(f"  Magnitude far from focus: {mag_far:.3e}")
    print(f"  Ratio: {mag_at_focus / mag_far:.2f}x stronger at focus")

    # Beam should be stronger at focal point
    assert mag_at_focus > mag_far, "Beam should be stronger at focal point"


def test_wavebundle_polarization():
    """
    Test that TE and TM polarizations produce different results.
    """
    positions = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([50.0])
    refractive_indices = np.array([1.5 + 0.01j])

    particles = Particles(
        position=positions,
        r=radii,
        refractive_index=refractive_indices,
    )

    wavelength = np.array([550.0])
    medium_refractive_index = np.array([1.0])

    numerics = Numerics(
        lmax=2,
        sampling_points_number=np.array([10, 20]),
        particle_distance_resolution=1,
        gpu=False,
    )

    # TE polarization
    initial_field_te = InitialField(
        beam_width=500.0,
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="TE",
    )

    parameters_te = Parameters(
        wavelength=wavelength,
        medium_refractive_index=medium_refractive_index,
        particles=particles,
        initial_field=initial_field_te,
    )

    simulation_te = Simulation(parameters_te, numerics)
    simulation_te.compute_mie_coefficients()
    simulation_te.compute_initial_field_coefficients()

    # TM polarization
    initial_field_tm = InitialField(
        beam_width=500.0,
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="TM",
    )

    parameters_tm = Parameters(
        wavelength=wavelength,
        medium_refractive_index=medium_refractive_index,
        particles=particles,
        initial_field=initial_field_tm,
    )

    simulation_tm = Simulation(parameters_tm, numerics)
    simulation_tm.compute_mie_coefficients()
    simulation_tm.compute_initial_field_coefficients()

    # Coefficients should be different
    coeffs_te = simulation_te.initial_field_coefficients.squeeze()
    coeffs_tm = simulation_tm.initial_field_coefficients.squeeze()

    diff = np.linalg.norm(coeffs_te - coeffs_tm)
    print(f"✓ TE vs TM polarization difference: {diff:.3e}")

    assert diff > 1e-6, "TE and TM should produce different coefficients"


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WAVEBUNDLE (GAUSSIAN BEAM) TESTS")
    print("=" * 70)

    test_wavebundle_normal_incidence_basic()
    print()
    test_wavebundle_beam_focusing()
    print()
    test_wavebundle_polarization()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
