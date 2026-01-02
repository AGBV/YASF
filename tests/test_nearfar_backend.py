import numpy as np

from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver


def _make_simulation(*, coupling_backend: str, near_radius: float | None) -> Simulation:
    particles = Particles(
        position=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.25, 0.5, 0.0],
            ]
        ),
        r=np.array([0.1, 0.1, 0.1]),
        refractive_index=np.array([1.5 + 0.0j, 1.5 + 0.0j, 1.5 + 0.0j]),
    )
    initial_field = InitialField(
        beam_width=0,
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="TE",
    )
    parameters = Parameters(
        wavelength=np.array([1.0, 1.3]),
        medium_refractive_index=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
        particles=particles,
        initial_field=initial_field,
    )

    solver = Solver(
        solver_type="gmres",
        tolerance=1e-6,
        max_iter=20,
        restart=20,
    )
    numerics = Numerics(
        lmax=1,
        sampling_points_number=np.array([10]),
        particle_distance_resolution=1.0,
        gpu=False,
        solver=solver,
        coupling_backend=coupling_backend,
        coupling_tile_size=2,
        coupling_near_field_radius=near_radius,
    )
    numerics.compute_translation_table()

    return Simulation(parameters, numerics)


def test_nearfar_large_radius_matches_dense():
    dense = _make_simulation(coupling_backend="dense", near_radius=None)
    nearfar = _make_simulation(coupling_backend="nearfar", near_radius=10.0)

    jmax = dense.parameters.particles.number * dense.numerics.nmax
    rng = np.random.default_rng(0)
    x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)

    wx_dense = dense.coupling_matrix_multiply(x)
    wx_nf = nearfar.coupling_matrix_multiply(x)

    np.testing.assert_allclose(wx_nf, wx_dense, rtol=1e-12, atol=1e-12)


def test_nearfar_zero_radius_removes_coupling():
    sim = _make_simulation(coupling_backend="nearfar", near_radius=0.0)

    jmax = sim.parameters.particles.number * sim.numerics.nmax
    rng = np.random.default_rng(1)
    x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)

    wx = sim.coupling_matrix_multiply(x)
    np.testing.assert_allclose(wx, 0.0, rtol=0.0, atol=0.0)


def test_nearfar_supports_single_wavelength_slice():
    dense = _make_simulation(coupling_backend="dense", near_radius=None)
    nearfar = _make_simulation(coupling_backend="nearfar", near_radius=10.0)

    jmax = dense.parameters.particles.number * dense.numerics.nmax
    rng = np.random.default_rng(2)
    x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)

    wx_dense = dense.coupling_matrix_multiply(x, idx=1)
    wx_nf = nearfar.coupling_matrix_multiply(x, idx=1)

    np.testing.assert_allclose(wx_nf, wx_dense, rtol=1e-12, atol=1e-12)
