"""
Example: Gaussian Beam Scattering

This example demonstrates how to simulate electromagnetic scattering
from particles illuminated by a focused Gaussian beam (wavebundle).

Compared to plane wave illumination, Gaussian beams:
- Have a finite beam width and focal point
- Show focusing effects
- More accurately represent realistic laser beams
"""

import numpy as np
from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver


def run_gaussian_beam_simulation():
    """
    Simulate scattering from particles illuminated by a Gaussian beam.
    """
    print("\n" + "="*70)
    print("GAUSSIAN BEAM SCATTERING EXAMPLE")
    print("="*70)

    # =========================================================================
    # 1. Define particles
    # =========================================================================
    print("\n1. Setting up particle configuration...")

    # Create a simple cluster of particles
    positions = np.array([
        [0.0, 0.0, 0.0],       # Particle 1: at beam focus
        [100.0, 0.0, 0.0],     # Particle 2: off-axis
        [0.0, 100.0, 0.0],     # Particle 3: off-axis
        [0.0, 0.0, 200.0],     # Particle 4: along beam axis
    ])

    # All particles have same size and material
    radii = np.array([50.0, 50.0, 50.0, 50.0])  # 50 nm radius
    refractive_indices = np.array([1.5 + 0.01j, 1.5 + 0.01j, 1.5 + 0.01j, 1.5 + 0.01j])

    particles = Particles(
        position=positions,
        r=radii,
        refractive_index=refractive_indices,
    )

    print(f"   Created {particles.number} particles")

    # =========================================================================
    # 2. Define Gaussian beam
    # =========================================================================
    print("\n2. Configuring Gaussian beam...")

    beam_width = 400.0  # Beam waist: 400 nm
    focal_point = np.array([0.0, 0.0, 0.0])  # Focus at origin
    polarization = "TE"  # Transverse electric polarization

    initial_field = InitialField(
        beam_width=beam_width,
        focal_point=focal_point,
        polar_angle=0.0,  # Normal incidence (beam along +z)
        azimuthal_angle=0.0,
        polarization=polarization,
    )

    print(f"   Beam width: {beam_width} nm")
    print(f"   Focal point: {focal_point}")
    print(f"   Polarization: {polarization}")

    # =========================================================================
    # 3. Set up simulation parameters
    # =========================================================================
    print("\n3. Setting up simulation...")

    wavelength = np.array([550.0])  # 550 nm (green light)
    medium_refractive_index = np.array([1.0])  # Vacuum

    parameters = Parameters(
        wavelength=wavelength,
        medium_refractive_index=medium_refractive_index,
        particles=particles,
        initial_field=initial_field,
    )

    # Create solver
    solver = Solver(
        solver_type="gmres",
        tolerance=1e-4,
        max_iter=1000,
        restart=1000,
    )

    # Create numerics with angular grid for integration
    # sampling_points_number = [azimuthal, polar]
    numerics = Numerics(
        lmax=3,  # Multipole expansion order
        sampling_points_number=np.array([15, 40]),  # Angular grid for wavebundle integration
        particle_distance_resolution=1,
        gpu=False,
        solver=solver,
    )

    print(f"   Wavelength: {wavelength[0]} nm")
    print(f"   Multipole order (lmax): {numerics.lmax}")
    print(f"   Angular grid: {15} azimuthal × {40} polar points")

    # Compute translation table
    numerics.compute_translation_table()
    print("   ✓ Translation table computed")

    # =========================================================================
    # 4. Run simulation
    # =========================================================================
    print("\n4. Running simulation...")

    simulation = Simulation(parameters, numerics)

    # Compute Mie coefficients
    simulation.compute_mie_coefficients()
    print("   ✓ Mie coefficients computed")

    # Compute initial field (Gaussian beam expansion)
    simulation.compute_initial_field_coefficients()
    print("   ✓ Gaussian beam expansion computed")

    # Compute right-hand side
    simulation.compute_right_hand_side()
    print("   ✓ Right-hand side computed")

    # Solve for scattered field coefficients
    simulation.compute_scattered_field_coefficients()
    print("   ✓ Scattered field computed")

    # =========================================================================
    # 5. Analyze results
    # =========================================================================
    print("\n5. Analyzing results...")

    # Initial field coefficients for each particle
    initial_coeffs = simulation.initial_field_coefficients[:, :, 0]  # Shape: (particles, nmax)

    print("\n   Initial field strength at each particle:")
    for i in range(particles.number):
        magnitude = np.linalg.norm(initial_coeffs[i, :])
        print(f"   Particle {i+1}: {magnitude:.3e}")

    # Scattered field coefficients
    scattered_coeffs = simulation.scattered_field_coefficients[:, :, 0]

    print("\n   Scattered field strength from each particle:")
    for i in range(particles.number):
        magnitude = np.linalg.norm(scattered_coeffs[i, :])
        print(f"   Particle {i+1}: {magnitude:.3e}")

    # =========================================================================
    # 6. Compare with plane wave
    # =========================================================================
    print("\n6. Comparing with plane wave illumination...")

    # Create plane wave with same parameters
    initial_field_pw = InitialField(
        beam_width=0.0,  # beam_width=0 triggers plane wave
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization=polarization,
    )

    parameters_pw = Parameters(
        wavelength=wavelength,
        medium_refractive_index=medium_refractive_index,
        particles=particles,
        initial_field=initial_field_pw,
    )

    simulation_pw = Simulation(parameters_pw, numerics)
    simulation_pw.compute_mie_coefficients()
    simulation_pw.compute_initial_field_coefficients()
    simulation_pw.compute_right_hand_side()
    simulation_pw.compute_scattered_field_coefficients()

    # Compare scattered field magnitudes
    scattered_gauss = np.linalg.norm(simulation.scattered_field_coefficients)
    scattered_plane = np.linalg.norm(simulation_pw.scattered_field_coefficients)

    print(f"\n   Total scattered field magnitude:")
    print(f"   Gaussian beam: {scattered_gauss:.3e}")
    print(f"   Plane wave:    {scattered_plane:.3e}")
    print(f"   Ratio (Gauss/Plane): {scattered_gauss/scattered_plane:.3f}")

    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)

    return simulation, simulation_pw


if __name__ == "__main__":
    simulation, simulation_pw = run_gaussian_beam_simulation()

    # Optional: Add visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        print("\n[Info] Visualization could be added here using matplotlib")
        print("       (e.g., plot field intensity, scattering patterns, etc.)")
    except ImportError:
        pass
