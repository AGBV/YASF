import shutil

import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip(
    "refidxdb",
    reason="refidxdb (yasfpy[explore]) is required for compare_mstm4 tests",
)

from yasfpy.benchmark.compare_mstm4 import compare_single_sphere


@pytest.mark.skipif(shutil.which("mstm") is None, reason="mstm binary not available")
def test_compare_single_sphere_qext_qsca_and_phase_and_dolp(tmp_path):
    # Use micrometer wavelengths/radii (the helper config uses scale=1e-6).
    result = compare_single_sphere(
        tmp_dir=tmp_path,
        wavelength=np.array([1.0]),
        radius=0.2,
        sphere_refractive_index=1.5 + 0.0j,
        medium_refractive_index=1.0 + 0.0j,
        lmax=8,
        solver_tolerance=1e-10,
        solver_max_iter=1200,
        solver_restart=1200,
        mstm_parallel=4,
    )

    # YASF vs MSTM4 should match closely for a single sphere.
    # Keep a small tolerance to avoid CPU/solver/platform differences.
    npt.assert_allclose(result.yasf_q_ext, result.mstm_q_ext_unp, rtol=5e-4, atol=0)
    npt.assert_allclose(result.yasf_q_sca, result.mstm_q_sca_unp, rtol=5e-4, atol=0)

    # Phase function: compare normalized angular shape.
    # Both arrays are normalized such that ∫ p(θ) dΩ = 4π, with dΩ = 2π sin(θ) dθ.
    weights = 2 * np.pi * np.sin(result.phase_theta)[:, np.newaxis]
    yasf_integral = np.trapz(
        result.yasf_phase_function * weights, x=result.phase_theta, axis=0
    )
    mstm_integral = np.trapz(
        result.mstm_phase_function * weights, x=result.phase_theta, axis=0
    )
    npt.assert_allclose(yasf_integral, 4 * np.pi, rtol=5e-3, atol=0)
    npt.assert_allclose(mstm_integral, 4 * np.pi, rtol=5e-3, atol=0)

    # Compare the full curve (on MSTM's theta grid).
    # Normalization differences are removed; this checks angular distribution.
    npt.assert_allclose(
        result.yasf_phase_function, result.mstm_phase_function, rtol=2e-3, atol=0
    )

    # Degree of linear polarization for unpolarized incidence.
    # This compare uses MSTM4 with `normalize_s11`, so the relevant quantity is `-S12`.
    npt.assert_allclose(result.yasf_dolp, result.mstm_dolp, rtol=5e-3, atol=3.2e-4)
