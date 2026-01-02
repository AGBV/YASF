import numpy as np

from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver


def _make_simulation(*, coupling_backend: str, tile_size: int = 2) -> Simulation:
    # 3 particles so the tiled backend exercises multiple tiles.
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
        coupling_tile_size=tile_size,
    )
    numerics.compute_translation_table()

    return Simulation(parameters, numerics)


def test_tiled_dense_matches_dense_matvec():
    dense = _make_simulation(coupling_backend="dense")
    tiled = _make_simulation(coupling_backend="tiled_dense", tile_size=2)

    jmax = dense.parameters.particles.number * dense.numerics.nmax
    rng = np.random.default_rng(0)
    x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)

    wx_dense = dense.coupling_matrix_multiply(x)
    wx_tiled = tiled.coupling_matrix_multiply(x)

    assert wx_dense.shape == wx_tiled.shape
    np.testing.assert_allclose(wx_tiled, wx_dense, rtol=1e-12, atol=1e-12)


def test_tiled_dense_matches_single_wavelength_slice():
    dense = _make_simulation(coupling_backend="dense")
    tiled = _make_simulation(coupling_backend="tiled_dense", tile_size=2)

    jmax = dense.parameters.particles.number * dense.numerics.nmax
    rng = np.random.default_rng(1)
    x = rng.normal(size=jmax) + 1j * rng.normal(size=jmax)

    # Compare idx-sliced matvecs as used by the solver loop.
    wx_dense = dense.coupling_matrix_multiply(x, idx=1)
    wx_tiled = tiled.coupling_matrix_multiply(x, idx=1)

    assert wx_dense.shape == wx_tiled.shape
    np.testing.assert_allclose(wx_tiled, wx_dense, rtol=1e-12, atol=1e-12)
