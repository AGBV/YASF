import miepython as mie
import numpy as np
import numpy.testing as npt
import pytest

from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.optics import Optics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver


@pytest.fixture(scope="module")
def numerics_lmax8() -> Numerics:
    solver = Solver(
        solver_type="gmres",
        tolerance=1e-10,
        max_iter=1200,
        restart=1200,
    )

    numerics = Numerics(
        lmax=8,
        sampling_points_number=np.array([30]),
        particle_distance_resolution=1.0,
        gpu=False,
        solver=solver,
    )

    numerics.compute_spherical_unity_vectors()
    numerics.compute_translation_table()
    return numerics


def _yasf_single_sphere_qext_qsca(
    *,
    wavelength: float,
    radius: float,
    refractive_index: complex,
    numerics: Numerics,
    medium_refractive_index: complex = 1.0 + 0j,
) -> tuple[float, float]:
    particles = Particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        r=np.array([radius]),
        refractive_index=np.array([refractive_index], dtype=complex),
    )

    initial_field = InitialField(
        beam_width=0,
        focal_point=np.array([0.0, 0.0, 0.0]),
        polar_angle=0.0,
        azimuthal_angle=0.0,
        polarization="UNP",
    )

    parameters = Parameters(
        wavelength=np.array([wavelength], dtype=float),
        medium_refractive_index=np.array([medium_refractive_index], dtype=complex),
        particles=particles,
        initial_field=initial_field,
    )

    particles.compute_volume_equivalent_area()

    simulation = Simulation(parameters, numerics)
    optics = Optics(simulation)

    simulation.compute_mie_coefficients()
    simulation.compute_initial_field_coefficients()
    simulation.compute_right_hand_side()
    simulation.compute_scattered_field_coefficients()

    optics.compute_cross_sections()
    optics.compute_efficiencies()

    return float(np.real(optics.q_ext[0])), float(np.real(optics.q_sca[0]))


@pytest.mark.smoke
def test_smoke_single_sphere_efficiencies_match_miepython(
    numerics_lmax8: Numerics,
):
    """Fast end-to-end check against miepython.

    This uses a single parameter set (instead of the full parametrized matrix)
    so it can serve as a quick release/install sanity test.
    """

    # Use n_medium=1 to avoid convention differences (YASF multiplies by |n_medium|).
    n_medium = 1.0 + 0j
    m_sphere = 1.5 + 0j
    wavelength = 1.0
    radius = 0.10

    qext_yasf, qsca_yasf = _yasf_single_sphere_qext_qsca(
        wavelength=wavelength,
        radius=radius,
        refractive_index=m_sphere,
        medium_refractive_index=n_medium,
        numerics=numerics_lmax8,
    )

    # miepython takes relative refractive index m (sphere/medium) and size x = 2π r n_medium / λ.
    m_rel = m_sphere / n_medium
    x = 2 * np.pi * radius * np.real(n_medium) / wavelength
    qext_mie, qsca_mie, _, _ = mie.efficiencies_mx(m_rel, x)

    npt.assert_allclose(qext_yasf, qext_mie)
    npt.assert_allclose(qsca_yasf, qsca_mie)


@pytest.mark.parametrize(
    ("m_sphere", "wavelength", "radius"),
    [
        (1.5 + 0j, 1.0, 0.10),
        (1.5 + 0.02j, 1.0, 0.08),
        (1.5 + 0j, 1.0, 0.15),
    ],
)
def test_single_sphere_matches_miepython_efficiencies(
    numerics_lmax8: Numerics,
    m_sphere: complex,
    wavelength: float,
    radius: float,
):
    # Use n_medium=1 to avoid convention differences (YASF multiplies by |n_medium|).
    n_medium = 1.0 + 0j

    qext_yasf, qsca_yasf = _yasf_single_sphere_qext_qsca(
        wavelength=wavelength,
        radius=radius,
        refractive_index=m_sphere,
        medium_refractive_index=n_medium,
        numerics=numerics_lmax8,
    )

    # miepython takes relative refractive index m (sphere/medium) and size x = 2π r n_medium / λ.
    m_rel = m_sphere / n_medium
    x = 2 * np.pi * radius * np.real(n_medium) / wavelength
    qext_mie, qsca_mie, _, _ = mie.efficiencies_mx(m_rel, x)

    npt.assert_allclose(qext_yasf, qext_mie)
    npt.assert_allclose(qsca_yasf, qsca_mie)
