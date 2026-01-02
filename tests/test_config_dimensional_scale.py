import json
from pathlib import Path

import numpy as np
from yasfpy.config import Config
from yasfpy.yasf import YASF


def _write_constant_nk_csv(path: Path, wavelength: float, n: complex) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "w,n,k\n"
    path.write_text(
        header
        + f"{wavelength:.16e},{float(np.real(n)):.16e},{float(np.imag(n)):.16e}\n"
    )


def _base_config(*, geometry_file: str, material_file: str) -> dict:
    return {
        "particles": {
            "geometry": {
                "file": geometry_file,
                "delimiter": ",",
                "scale": 1.0,
            },
            "material": [
                {
                    "path": material_file,
                    "w_column": "wl",
                    "scale": 1.0,
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
        "parameters": {"wavelength": {"data": [1.0], "scale": 1.0}, "medium": 1.0},
        "solver": {"type": "gmres", "tolerance": 1e-8, "max_iter": 10, "restart": 10},
        "numerics": {
            "lmax": 1,
            "sampling_points": [4, 8],
            "particle_distance_resolution": 1,
            "gpu": False,
        },
        "optics": False,
        "output": {"folder": "out", "extension": "bz2"},
    }


def test_config_honors_particles_geometry_dimensional_scale(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    geometry_path = tmp_path / "cluster.csv"
    material_path = tmp_path / "material.csv"

    geometry_path.write_text("0,0,0,1\n")
    _write_constant_nk_csv(material_path, wavelength=1.0, n=1.5 + 0j)

    config = _base_config(
        geometry_file=geometry_path.name, material_file=material_path.name
    )
    config["particles"]["geometry"]["dimensional_scale"] = 2.0
    config_path.write_text(json.dumps(config))

    cfg = Config(path_config=str(config_path), preprocess=True)

    # dimensional_scale is applied before the wavelength scaling step.
    np.testing.assert_allclose(cfg.spheres[0, :4], np.array([0.0, 0.0, 0.0, 2.0]))


def test_cli_dimensional_scale_overrides_config_for_yasf(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    geometry_path = tmp_path / "cluster.csv"
    material_path = tmp_path / "material.csv"

    geometry_path.write_text("0,0,0,1\n")
    _write_constant_nk_csv(material_path, wavelength=1.0, n=1.5 + 0j)

    config = _base_config(
        geometry_file=geometry_path.name, material_file=material_path.name
    )
    config["particles"]["geometry"]["dimensional_scale"] = 2.0
    config_path.write_text(json.dumps(config))

    yasf = YASF(
        path_config=str(config_path),
        preprocess=True,
        cluster_dimensional_scale=3.0,
        quiet=True,
    )

    np.testing.assert_allclose(
        yasf.config.spheres[0, :4], np.array([0.0, 0.0, 0.0, 3.0])
    )
