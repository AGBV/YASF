import numpy as np
import pytest

from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver


def _make_minimal_simulation(*, coupling_backend: str) -> Simulation:
    particles = Particles(
        position=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        r=np.array([0.1, 0.1]),
        refractive_index=np.array([1.5 + 0.0j, 1.5 + 0.0j]),
    )
    initial_field = InitialField(
        beam_width=0,
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="TE",
    )
    parameters = Parameters(
        wavelength=np.array([1.0]),
        medium_refractive_index=np.array([1.0 + 0.0j]),
        particles=particles,
        initial_field=initial_field,
    )

    solver = Solver(
        solver_type="gmres",
        tolerance=1e-6,
        max_iter=50,
        restart=50,
    )
    numerics = Numerics(
        lmax=1,
        sampling_points_number=np.array([10]),
        particle_distance_resolution=1.0,
        gpu=False,
        solver=solver,
        coupling_backend=coupling_backend,
    )
    numerics.compute_translation_table()
    return Simulation(parameters, numerics)


def test_nearfar_backend_skips_dense_lookups():
    sim = _make_minimal_simulation(coupling_backend="nearfar")

    assert sim.sph_h is None
    assert sim.plm is None
    assert sim.e_j_dm_phi is None

    with pytest.raises(ValueError):
        sim.coupling_matrix_multiply(
            np.zeros(2 * sim.parameters.particles.number * sim.numerics.nmax)
        )


def test_dense_backend_still_builds_lookups():
    sim = _make_minimal_simulation(coupling_backend="dense")

    assert sim.sph_h is not None
    assert sim.plm is not None
    assert sim.e_j_dm_phi is not None


def test_tiled_dense_backend_skips_dense_lookups():
    sim = _make_minimal_simulation(coupling_backend="tiled_dense")

    assert sim.sph_h is None
    assert sim.plm is None
    assert sim.e_j_dm_phi is None
