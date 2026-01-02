"""Comparison utilities for MSTM4 reference results.

Contains helpers/data structures to compare YASF output to MSTM4 and summarize
error metrics.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np

from yasfpy.benchmark.mstm4 import MSTM4Manager
import yasfpy.mueller as mueller
from yasfpy.yasf import YASF


def _as_radians(theta: np.ndarray) -> np.ndarray:
    """Return angles in radians.

    Heuristic: MSTM's scattering angles are commonly in degrees (0..180).
    """

    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return theta
    if np.nanmax(theta) > 2 * np.pi + 1e-6:
        return np.deg2rad(theta)
    return theta


def _as_2d(x: np.ndarray) -> np.ndarray:
    """Return an array with at least two dimensions.

    Parameters
    ----------
    x
        Input array.

    Returns
    -------
    numpy.ndarray
        If ``x`` is 1D, returns ``x[:, np.newaxis]``. Otherwise returns ``x``.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, np.newaxis]
    return x


def _resample_phase_function(
    *,
    theta_target: np.ndarray,
    theta_source: np.ndarray,
    phase_source: np.ndarray,
) -> np.ndarray:
    """Interpolate `phase_source(theta_source)` onto `theta_target`.

    Both theta arrays must be in radians.
    """

    theta_target = np.asarray(theta_target, dtype=float)
    theta_source = np.asarray(theta_source, dtype=float)
    phase_source = _as_2d(phase_source)

    order = np.argsort(theta_source)
    theta_sorted = theta_source[order]
    phase_sorted = phase_source[order, :]

    if theta_sorted.size < 2:
        raise ValueError("phase function requires at least 2 angles")

    phase_target = np.empty((theta_target.size, phase_sorted.shape[1]), dtype=float)
    for wl_idx in range(phase_sorted.shape[1]):
        phase_target[:, wl_idx] = np.interp(
            theta_target, theta_sorted, phase_sorted[:, wl_idx]
        )
    return phase_target


def _phase_integral(theta: np.ndarray, phase_function: np.ndarray) -> np.ndarray:
    """Compute ∫ p(θ) dΩ assuming azimuthal symmetry.

    Uses dΩ = 2π sin(θ) dθ.

    Returns a 1D array of shape (n_wavelengths,).
    """

    theta = np.asarray(theta, dtype=float)
    phase_function = _as_2d(phase_function)
    weights = 2 * np.pi * np.sin(theta)[:, np.newaxis]

    # Integrate along theta for each wavelength.
    return np.trapz(phase_function * weights, x=theta, axis=0)


def _normalize_phase_function_to_4pi(
    theta: np.ndarray, phase_function: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scale phase function so integral over dΩ equals 4π."""

    phase_function = _as_2d(phase_function)
    integral = _phase_integral(theta, phase_function)
    if np.any(integral == 0):
        raise ZeroDivisionError("phase function integral is zero")

    scale = (4 * np.pi) / integral
    return phase_function * scale[np.newaxis, :], integral


@dataclass(frozen=True)
class CompareResult:
    """Container for comparison outputs.

    This data structure groups the key arrays extracted from YASF and MSTM4
    runs (efficiencies, phase function, and polarization diagnostics).
    """

    wavelength: np.ndarray

    yasf_q_ext: np.ndarray
    yasf_q_sca: np.ndarray
    mstm_q_ext_unp: np.ndarray
    mstm_q_sca_unp: np.ndarray

    # Phase function comparison
    # `phase_theta` is in radians and is a common grid.
    # Phase functions are normalized to integrate to 4π over dΩ.
    phase_theta: np.ndarray
    yasf_phase_function: np.ndarray
    mstm_phase_function: np.ndarray

    yasf_phase_integral: np.ndarray
    mstm_phase_integral: np.ndarray

    # Polarization comparison
    # With MSTM4's `normalize_s11` enabled, the exported scattering-matrix column "12"
    # is already normalized by S11. Therefore DoLP for unpolarized incidence is `-S12`.
    yasf_dolp: np.ndarray
    mstm_dolp: np.ndarray


def _write_constant_nk_csv(path: Path, wavelength: np.ndarray, n: complex) -> None:
    """Write a minimal `w,n,k` CSV suitable for `refidxdb`.

    Important: `refidxdb`'s CSV reader applies an internal default scale of `1e-6`
    when `w_column` is wavelength ("wl"). Therefore, the `w` values in this file
    are expected to be expressed in micrometers.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    n_real = np.full_like(wavelength, float(np.real(n)), dtype=float)
    n_imag = np.full_like(wavelength, float(np.imag(n)), dtype=float)

    header = "w,n,k\n"
    lines = [
        f"{w:.16e},{nr:.16e},{ni:.16e}\n"
        for w, nr, ni in zip(wavelength, n_real, n_imag, strict=True)
    ]
    path.write_text(header + "".join(lines))


def _write_single_sphere_geometry(path: Path, radius: float) -> None:
    """Write a single-sphere CSV geometry file.

    Parameters
    ----------
    path
        Output CSV path.
    radius
        Sphere radius (in the same units as the config particle scale).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # x,y,z,r (material index inferred via distribution)
    path.write_text(f"0,0,0,{radius:.16e}\n")


def compare_single_sphere(
    *,
    tmp_dir: str | Path,
    wavelength: np.ndarray,
    radius: float,
    sphere_refractive_index: complex,
    medium_refractive_index: complex = 1.0 + 0.0j,
    lmax: int = 10,
    solver_type: str = "gmres",
    solver_tolerance: float = 1e-8,
    solver_max_iter: int = 2000,
    solver_restart: int = 2000,
    mstm_parallel: int = 4,
    mstm_binary: str = "mstm",
) -> CompareResult:
    """Run YASF and MSTM4 for a single sphere and compare efficiencies.

    This helper is designed for tests/benchmarks and intentionally avoids network
    refractive-index sources by creating local constant `w,n,k` CSV tables that
    `refidxdb.Handler` can load.

    Notes
    -----
    YASF ``"UNP"`` corresponds to a single combined polarization state (not a
    TE/TM average). MSTM provides explicit unpolarized-incidence efficiencies.
    For an apples-to-apples comparison, use MSTM's ``q_*_unp`` outputs.
    """

    tmp_dir = Path(tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    config_path = (tmp_dir / "config.json").resolve()
    geometry_path = (tmp_dir / "sphere.csv").resolve()
    material_path = (tmp_dir / "material.csv").resolve()

    wavelength = np.asarray(wavelength, dtype=float)
    if wavelength.ndim != 1 or wavelength.size == 0:
        raise ValueError("wavelength must be a 1D non-empty array")

    if isinstance(medium_refractive_index, (float, int)):
        medium_value: float | dict[str, float] = float(medium_refractive_index)
    else:
        # Config only supports float or url handler for medium. Use float for now.
        if np.imag(medium_refractive_index) != 0:
            raise ValueError(
                "Config currently only supports real-valued medium refractive index (float)"
            )
        medium_value = float(np.real(medium_refractive_index))

    # radius is expressed in micrometers, consistent with `particles_scale`.
    _write_single_sphere_geometry(geometry_path, radius)
    _write_constant_nk_csv(
        material_path, wavelength=wavelength, n=sphere_refractive_index
    )

    # Use micrometers for wavelength/material tables so that refidxdb's default
    # CSV scaling (1e-6) produces meter units internally.
    wavelength_um = wavelength
    wavelength_scale = 1e-6
    particles_scale = 1e-6

    config: dict[str, Any] = {
        "particles": {
            "geometry": {
                "file": geometry_path.name,
                "delimiter": ",",
                "scale": particles_scale,
            },
            "material": [
                {
                    "path": material_path.name,
                    "w_column": "wl",
                    "scale": wavelength_scale,
                }
            ],
            "distribution": [1.0],
        },
        "initial_field": {
            "beam_width": 0,
            "focal_point": [0, 0, 0],
            "polar_angle": 0,
            "azimuthal_angle": 0,
            "polarization": "UNP",
        },
        "parameters": {
            "wavelength": {"data": wavelength_um.tolist(), "scale": wavelength_scale},
            "medium": medium_value,
        },
        "solver": {
            "type": solver_type,
            "tolerance": solver_tolerance,
            "max_iter": solver_max_iter,
            "restart": solver_restart,
        },
        "numerics": {
            "lmax": lmax,
            "sampling_points": [20, 40],
            "particle_distance_resolution": 1,
            "gpu": False,
        },
        "optics": True,
        "output": {"folder": str((tmp_dir / "out").resolve()), "extension": "bz2"},
    }
    config_path.write_text(json.dumps(config))

    cwd = Path.cwd()
    try:
        # Ensure relative geometry/material paths resolve against config directory
        # (Config joins relative paths with `path_config.parent`).
        # We also keep MSTM input/output local to tmp_dir by chdir.
        # MSTM4Manager uses plain filenames for `input_file`/`output_file`.
        # noqa: SIM115
        import os

        os.chdir(tmp_dir)

        yasf = YASF(path_config=str(config_path), preprocess=True)
        yasf.run()

        if shutil.which(mstm_binary) is None:
            raise FileNotFoundError(f"MSTM binary '{mstm_binary}' not found in PATH")

        mstm = MSTM4Manager(
            path_config=str(config_path),
            binary=mstm_binary,
            parallel=mstm_parallel,
            nix=True,
            random_orientation=False,
            incidence_average=False,
            azimuthal_average=False,
        )
        mstm.run(cleanup=True)

        yasf_theta = _as_radians(np.asarray(yasf.optics.scattering_angles))
        yasf_phase = np.asarray(yasf.optics.phase_function, dtype=float)

        mstm_theta = _as_radians(np.asarray(mstm.output["scattering_matrix"]["theta"]))
        mstm_s11 = np.asarray(mstm.output["scattering_matrix"]["11"], dtype=float)

        # Use MSTM's angle grid as reference and ensure it is sorted.
        mstm_order = np.argsort(mstm_theta)
        phase_theta = mstm_theta[mstm_order]
        mstm_s11_sorted = _as_2d(mstm_s11)[mstm_order, :]

        yasf_phase_resampled = _resample_phase_function(
            theta_target=phase_theta,
            theta_source=yasf_theta,
            phase_source=yasf_phase,
        )

        yasf_phase_norm, yasf_phase_integral = _normalize_phase_function_to_4pi(
            phase_theta, yasf_phase_resampled
        )
        mstm_phase_norm, mstm_phase_integral = _normalize_phase_function_to_4pi(
            phase_theta, mstm_s11_sorted
        )

        # For polarization quantities MSTM uses a dense 0..180° grid (181 points).
        # Using the same θ-grid in YASF avoids interpolation artifacts near small angles.
        if (
            isinstance(config.get("numerics", {}).get("sampling_points"), list)
            and len(config["numerics"]["sampling_points"]) == 2
        ):
            config["numerics"]["sampling_points"] = [
                int(config["numerics"]["sampling_points"][0]),
                int(phase_theta.size),
            ]

        def _run_yasf_with_polarization(polarization: str) -> YASF:
            config["initial_field"]["polarization"] = polarization
            config_path.write_text(json.dumps(config))
            instance = YASF(path_config=str(config_path), preprocess=True)
            instance.run()
            return instance

        yasf_te = _run_yasf_with_polarization("TE")
        yasf_tm = _run_yasf_with_polarization("TM")

        e_theta_te, e_phi_te = (
            yasf_te.optics.compute_scattered_e_field_angle_components()
        )
        e_theta_tm, e_phi_tm = (
            yasf_tm.optics.compute_scattered_e_field_angle_components()
        )

        # Convert the scattered E-field from spherical (theta,phi) components into the
        # MSTM "scattering plane" output basis (perpendicular/parallel).
        #
        # With incidence along +z (beta=0) the scattering plane is spanned by z-hat and
        # k_sca. In that basis:
        #   E_perp = E_phi
        #   E_par  = -E_theta
        #
        # Important: YASF's TE/TM at alpha=0 are fixed lab polarizations (y/x), not
        # scattering-plane (perp/par) polarizations for each azimuth. Therefore we first
        # build a Jones matrix from two orthogonal *lab* incident states (x,y), then
        # rotate the incident basis into the local scattering-plane basis for each
        # sampling point.
        e_perp_x = e_phi_tm
        e_par_x = -e_theta_tm
        e_perp_y = e_phi_te
        e_par_y = -e_theta_te

        jones_out_sp_in_lab = np.empty(e_theta_te.shape + (2, 2), dtype=complex)
        jones_out_sp_in_lab[..., 0, 0] = e_perp_x
        jones_out_sp_in_lab[..., 1, 0] = e_par_x
        jones_out_sp_in_lab[..., 0, 1] = e_perp_y
        jones_out_sp_in_lab[..., 1, 1] = e_par_y

        # For each sampling point, rotate (x,y) -> (perp, par) for the incident field.
        # For k_inc || +z the local scattering-plane unit vectors projected onto the
        # transverse (x,y) plane are:
        #   e_perp = (-sin(phi), cos(phi))
        #   e_par  = ( cos(phi), sin(phi))
        phi = np.asarray(yasf_te.simulation.numerics.azimuthal_angles, dtype=float)
        rot = np.zeros((phi.size, 2, 2), dtype=float)
        rot[:, 0, 0] = -np.sin(phi)
        rot[:, 0, 1] = np.cos(phi)
        rot[:, 1, 0] = np.cos(phi)
        rot[:, 1, 1] = np.sin(phi)

        # Incident basis change: e_sp = rot @ e_lab, so e_lab = rot.T @ e_sp.
        # Therefore J_sp = J_out_sp_in_lab @ rot.T.
        rot_t = np.swapaxes(rot, -1, -2)
        jones_out_sp_in_sp = np.einsum("awij,ajk->awik", jones_out_sp_in_lab, rot_t)
        mueller_matrix = mueller.jones_to_mueller_numba(jones_out_sp_in_sp)

        # Average over azimuthal angle if we are on a regular (phi,theta) grid.
        if (
            yasf_te.simulation.numerics.sampling_points_number is not None
            and yasf_te.simulation.numerics.sampling_points_number.size == 2
        ):
            n_phi = int(yasf_te.simulation.numerics.sampling_points_number[0])
            n_theta = int(yasf_te.simulation.numerics.sampling_points_number[1])
            mueller_matrix = mueller_matrix.reshape(
                (n_phi, n_theta, wavelength.size, 4, 4)
            ).mean(axis=0)
            yasf_mueller_theta = _as_radians(
                np.asarray(yasf_te.optics.scattering_angles)
            )
        else:
            yasf_mueller_theta = _as_radians(
                np.asarray(yasf_te.simulation.numerics.polar_angles)
            )

        # For unpolarized incidence S_in = (I,0,0,0), so:
        #   I_out = M00 * I
        #   Q_out = M10 * I
        # and DoLP = Q_out / I_out.
        # MSTM's convention is DoLP = -S12/S11, which matches M10/M00 when
        # S11 ↔ M00 and S12 ↔ -M10.
        yasf_s11 = np.asarray(mueller_matrix[..., 0, 0], dtype=float)
        yasf_s12 = np.asarray(mueller_matrix[..., 1, 0], dtype=float)
        yasf_dolp = yasf_s12 / yasf_s11

        mstm_s12_sorted = _as_2d(
            np.asarray(mstm.output["scattering_matrix"]["12"], dtype=float)
        )[mstm_order, :]
        # MSTM output already uses normalize_s11, so column "12" is effectively
        # a normalized polarization element. For unpolarized incidence the DoLP is
        # conventionally defined as -S12/S11.
        mstm_dolp = -mstm_s12_sorted

        yasf_dolp_resampled = _resample_phase_function(
            theta_target=phase_theta,
            theta_source=yasf_mueller_theta,
            phase_source=yasf_dolp,
        )

        return CompareResult(
            wavelength=wavelength,
            yasf_q_ext=np.asarray(yasf.optics.q_ext, dtype=float),
            yasf_q_sca=np.asarray(yasf.optics.q_sca, dtype=float),
            mstm_q_ext_unp=np.asarray(
                mstm.output["efficiencies"]["q_ext_unp"], dtype=float
            ),
            mstm_q_sca_unp=np.asarray(
                mstm.output["efficiencies"]["q_sca_unp"], dtype=float
            ),
            phase_theta=phase_theta,
            yasf_phase_function=yasf_phase_norm,
            mstm_phase_function=mstm_phase_norm,
            yasf_phase_integral=yasf_phase_integral,
            mstm_phase_integral=mstm_phase_integral,
            yasf_dolp=np.asarray(yasf_dolp_resampled, dtype=float),
            mstm_dolp=np.asarray(mstm_dolp, dtype=float),
        )
    finally:
        import os

        os.chdir(cwd)
