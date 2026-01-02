import bz2
import copy
import json
import logging
from datetime import datetime
from io import StringIO
from numbers import Number
from pathlib import Path
from typing import Any

import _pickle
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from numba import jit, prange

try:
    from refidxdb.handler import Handler as RefIdxHandler
except ImportError:  # pragma: no cover
    RefIdxHandler = None

# from pydantic import BaseModel
# from yasfpy.functions.misc import generate_refractive_index_table

OVERLAP_MULTIPLICATOR = 0.95


class Config:
    # class Config(BaseModel):
    # TODO: rename to data or something similar
    # config.config can get confusing and isn't very descriptive
    config: dict = {}
    path_config: Path
    path_cluster: Path
    cluster_scale: float
    cluster_dimensional_scale: float
    preprocess: bool = True

    spheres: npt.NDArray = np.zeros((0, 4))
    particle_metadata: dict[str, Any] = {}

    def __init__(
        self,
        path_config: str = "",
        preprocess: bool = True,
        path_cluster: str = "",
        cluster_scale: float = 1.0,
        cluster_dimensional_scale: float = 1.0,
    ):
        # if type(path_config) != str:
        if not isinstance(path_config, str):
            raise Exception("The config file path needs to be a string!")
        self.path_config = Path(path_config)
        # self.file_type = path_config.split(".")[-1]
        self.file_type = self.path_config.suffix
        match self.file_type:
            case ".json":
                with open(path_config) as data:
                    self.config = json.load(data)
            case ".yaml" | ".yml":
                with open(path_config) as data:
                    self.config = yaml.safe_load(data)
            case _:
                raise FileNotFoundError(
                    "The provided config file needs to be a json or yaml file!"
                )
        if self.config is None:
            raise FileNotFoundError(
                f"Could not read config file {path_config}. Check if the file exists."
            )
        if path_cluster != "":
            self.path_cluster = Path(path_cluster)
        elif "file" in self.config["particles"]["geometry"]:
            self.path_cluster = (
                self.path_config.parent / self.config["particles"]["geometry"]["file"]
            )
        else:
            raise FileNotFoundError(
                "The cluster file needs to be provided either in the config file or as an argument."
            )

        self.particles_scale = cluster_scale
        self.cluster_dimensional_scale = cluster_dimensional_scale

        # self.path_cluster = Path(
        #     self.config["particles"]["geometry"]["file"]
        #     if path_cluster == ""
        #     else path_cluster
        # )
        # if not self.path_cluster.is_absolute():
        #     self.path_cluster = self.path_config.parent / self.path_cluster
        # if not self.path_cluster.exists():
        #     raise FileNotFoundError(
        #         f"The provided cluster file {self.path_cluster} does not exist."
        #     )

        # if not self.path_cluster.startswith("/"):
        #     self.path_cluster = str(Path("") / self.path_cluster)
        # self.path_cluster = str(_path_config.parent / self.path_cluster)

        self.log = logging.getLogger(self.__class__.__module__)
        self.__read()
        self.__folder()
        if preprocess:
            self.__interpolate()

    def __read(self):
        # TODO: import of csv files (single column)
        # wavelength
        if "path" in self.config["parameters"]["wavelength"]:
            self.wavelength = np.loadtxt(
                self.config["parameters"]["wavelength"]["path"]
            )
        elif isinstance(self.config["parameters"]["wavelength"]["data"], list):
            self.wavelength = np.array(self.config["parameters"]["wavelength"]["data"])
        elif isinstance(self.config["parameters"]["wavelength"]["data"], dict):
            self.wavelength = np.arange(
                self.config["parameters"]["wavelength"]["data"]["start"],
                self.config["parameters"]["wavelength"]["data"]["stop"],
                self.config["parameters"]["wavelength"]["data"]["step"],
            )
        else:
            raise Exception(
                "Please provide the wavelength data as an array, or the (start, stop, step) numpy.arange parameters."
            )
        self.wavelength_scale = (
            self.config["parameters"]["wavelength"]["scale"]
            if "scale" in self.config["parameters"]["wavelength"]
            else 1
        )
        self.log.info(f"Loaded wavelength: {self.wavelength * self.wavelength_scale}")

        # TODO: move the interpolation of data into config and away from the YASF function
        # NOTE: Kinda done, but needs to be checked!
        # refractive indices of particles
        self.material = []
        for mat in self.config["particles"]["material"]:
            if ("url" in mat or "path" in mat) and RefIdxHandler is None:
                raise ImportError(
                    "Loading material tables via 'url'/'path' requires the optional dependency "
                    "'refidxdb'. Install with 'pip install yasfpy[explore]'."
                )

            if "url" in mat:
                handle = RefIdxHandler(url=mat["url"])  # type: ignore[misc]
            elif "path" in mat:
                path = Path(mat["path"])
                if not path.is_absolute():
                    path = self.path_config.parent / path
                w_column = mat.get("w_column", mat.get("wavelength", True))
                # `refidxdb` expects `w_column` as bool or one of {"wl","wn"}.
                # Keep backward compatibility with older configs using `wavelength`.
                handle = RefIdxHandler(  # type: ignore[misc]
                    path=path.absolute().as_posix(),
                    w_column=w_column,
                )
            else:
                raise ValueError("Material needs to have a url or path key.")
            self.material.append(dict(ref_idx=handle.nk))
        self.material_scale = [
            mat["scale"] for mat in self.config["particles"]["material"]
        ]

        # refractive index of medium
        if isinstance(self.config["parameters"]["medium"], dict) and (
            "url" in self.config["parameters"]["medium"]
        ):
            if RefIdxHandler is None:
                raise ImportError(
                    "Loading medium refractive index via 'url' requires the optional dependency "
                    "'refidxdb'. Install with 'pip install yasfpy[explore]'."
                )
            self.medium = RefIdxHandler(url=self.config["parameters"]["medium"]["url"])  # type: ignore[misc]
        else:
            self.medium = self.config["parameters"]["medium"]

        # particle geometry
        header = None
        if "delimiter" in self.config["particles"]["geometry"]:
            delim = self.config["particles"]["geometry"]["delimiter"]
            if delim == "whitespace":
                delim = r"\s+"
            spheres = pd.read_csv(
                self.path_cluster.absolute(),
                header=header,
                sep=delim,
            ).to_numpy()
        else:
            match self.path_cluster.suffix:
                case ".csv":
                    delim = ","
                    header = 0
                    spheres = pd.read_csv(
                        self.path_cluster.absolute(),
                        header=header,
                        sep=delim,
                    ).to_numpy()
                case ".dat":
                    # delim = r"\s+"
                    spheres = np.loadtxt(self.path_cluster.absolute(), ndmin=2)
                    with open(self.path_cluster) as f:
                        lines = f.readlines()
                        lines = [ln[1:] for ln in lines if ln.startswith("#")]
                        self.particle_metadata = yaml.safe_load(
                            StringIO("".join(lines))
                        )
                    print(self.particle_metadata)
                case _:
                    raise NotImplementedError(
                        f"""
                        The file type {self.path_cluster.suffix} is not supported.
                        Please specify it using the `delim` key in the geometry section.
                        """
                    )
        if spheres.shape[1] < 4:
            raise Exception(
                "The particle geometry file needs at least 4 columns (x, y, z, r) and an optinal refractive index column"
            )
        elif spheres.shape[1] == 4:
            self.log.info("4 columns have been provided")
            if "distribution" not in self.config["particles"]:
                self.log.warning(
                    "Implying a uniform distribution of all materials among the particles"
                )
                length = len(self.config["particles"]["material"])
                self.config["particles"]["distribution"] = np.ones(length) / length
            distribtion = self.config["particles"]["distribution"]
            spheres = np.hstack(
                (
                    spheres,
                    np.random.choice(
                        len(distribtion),
                        size=(spheres.shape[0], 1),
                        p=distribtion,
                    ),
                )
            )
        elif spheres.shape[1] >= 5:
            # TODO: maybe include n,k column
            self.log.warning(
                "More than 5 columns have been provided. Everything after the 5th will be ignored!"
            )
        # self.particles_scale = (
        #     self.config["particles"]["geometry"]["scale"]
        #     if "scale" in self.config["particles"]["geometry"]
        #     else 1
        # )
        if "scale" in self.config["particles"]["geometry"]:
            self.particles_scale = self.config["particles"]["geometry"]["scale"]

        # self.spheres = spheres.to_numpy()
        self.spheres = spheres * self.cluster_dimensional_scale
        self.spheres[:, 3] = fix_overlap(self.spheres[:, :3], self.spheres[:, 3])

        # NOTE: Scale the distnaces and radii to the wavelength
        # This should make the size parameter correct
        # self.spheres[:, :4] = (
        #     self.spheres[:, :4] * self.particles_scale / self.wavelength_scale
        # )
        if self.particle_metadata and "aggregate_properties" in self.particle_metadata:
            self.particle_metadata["aggregate_properties"]["radius_of_gyration"] *= (
                self.particles_scale / self.wavelength_scale
            )
            print(self.particle_metadata)
        self.spheres[:, :4] *= self.particles_scale / self.wavelength_scale
        self.log.info(
            f"Particles have been scaled by {self.particles_scale / self.wavelength_scale} to match the wavelength"
        )

        if "optics" in self.config:
            self.config["optics"] = (
                self.config["optics"]
                if isinstance(self.config["optics"], bool)
                else True
            )
        else:
            self.config["optics"] = True

        if "points" in self.config:
            points = dict(x=np.array([0]), y=np.array([0]), z=np.array([0]))

            for key, value in self.config["points"].items():
                if isinstance(value, Number):
                    points[key] = np.array([value])
                elif isinstance(value, list):
                    points[key] = np.array(value)
                elif isinstance(value, dict):
                    points[key] = np.arange(
                        value["start"], value["stop"], value["step"]
                    )
                else:
                    raise Exception(
                        f"The key {key} is not a valid type. Numbers, list of numbers and arange dicts are permited"
                    )
            x, y, z = np.meshgrid(points["x"], points["y"], points["z"], indexing="ij")
            points = dict(x=x.ravel(), y=y.ravel(), z=z.ravel())
            self.config["points"] = points
            self.config["points_shape"] = x.shape

        r_min = np.min(self.spheres[:, 3])
        r_max = np.max(self.spheres[:, 3])
        r_std = np.std(self.spheres[:, 3])
        r_avg = np.mean(self.spheres[:, 3])
        r_data = np.array([r_min, r_avg, r_std, r_max])
        self.log.info(
            f"{'Radius':10} | {'Min':10} | {'Avg':>10}±{'Std':10} | {'Max':>10}"
        )
        self.log.info(
            f"{'':10} | {r_min:<10.2e} | {r_avg:10.2e}±{r_std:<10.2e} | {r_max:10.2e}"
        )

        self.log.info("Size Parameters")
        self.log.info(
            f"{'Wavelength':^10} | {'Min':^10} | {'Avg':>10}±{'Std':10} | {'Max':^10}"
        )
        match self.wavelength_scale:
            case 1e-3:
                suffix = "mm"
            case 1e-6:
                suffix = "μm"
            case 1e-9:
                suffix = "nm"
            case 1e-12:
                suffix = "pm"
            case 1e-10:
                suffix = " Å"
            case _:
                suffix = " m"
        for w in self.wavelength:
            x = 2 * np.pi * r_data / w
            x_min = x[0]
            x_avg = x[1]
            x_std = x[2]
            x_max = x[3]
            self.log.info(
                f"{w:8.2f}{suffix} | {x_min:^10.2f} | {x_avg:10.2f}±{x_std:<10.2f} | {x_max:^10.2f}"
            )

    def __folder(self):
        if "folder" in self.config["output"]:
            folder = Path(self.config["output"]["folder"])
            if not folder.is_absolute():
                folder = Path.cwd() / folder
        else:
            folder = Path.cwd()
        folder.mkdir(parents=True, exist_ok=True)

        extension = (
            self.config["output"]["extension"]
            if "extension" in self.config["output"]
            else "pbz2"
        )
        filename = f"{self.path_config.stem}_{self.path_cluster.stem}_dimscale{self.cluster_dimensional_scale:.2f}"
        filename = filename.replace(".", "p")
        if "output" in self.config:
            if "filename" in self.config["output"]:
                filename = self.config["output"]["filename"]
            elif isinstance(self.config["output"], str):
                filename = self.config["output"]
        filename = f"{filename}.{extension}"
        self.output_filename = folder / filename if (filename is not None) else None

    def __interpolate(self):
        # Replace this with handlers interpolate function...
        # Could be even done in parallel with numba
        refractive_index_interpolated = np.zeros(
            (len(self.material), self.wavelength.size),
            dtype=complex,
        )
        for idx, data in enumerate(self.material):
            table = data["ref_idx"].to_numpy().astype(float)
            refractive_index_interpolated[idx, :] = np.interp(
                self.wavelength * self.wavelength_scale,
                # table[:, 0] * self.material_scale[idx],
                table[:, 0],
                table[:, 1] + 1j * table[:, 2],
                left=table[0, 1] + 1j * table[0, 2],
                right=table[-1, 1] + 1j * table[-1, 2],
            )
        self.refractive_index_interpolated = refractive_index_interpolated

        if isinstance(self.medium, float):
            self.medium_refractive_index = np.ones_like(self.wavelength) * self.medium
        else:
            self.medium_refractive_index = np.array(
                self.medium.interpolate(
                    target=self.wavelength,
                    scale=self.wavelength_scale,
                    complex=True,
                )
            )
        # self.medium_refractive_index = np.interp(
        #     self.wavelength * self.wavelength_scale,
        #     self.medium.nk["w"],
        #     self.medium.nk["n"].to_numpy() + 1j * self.medium.nk["k"].to_numpy(),
        # )
        # self.medium_refractive_index = np.interp(
        #     self.wavelength * self.wavelength_scale,
        #     self.medium[0]["ref_idx"]["wavelength"] * self.medium_scale,
        #     self.medium[0]["ref_idx"]["n"] + 1j * self.medium[0]["ref_idx"]["k"],
        # )

        # self.medium_refractive_index.imag = 0
        # self.medium_refractive_index = np.real(self.medium_refractive_index)
        # print(self.medium_refractive_index)
        self.log.info("Refractive index tables:")
        print(self.wavelength.shape)
        print(self.refractive_index_interpolated.shape)
        self.log.info(
            pd.DataFrame(
                data=self.refractive_index_interpolated.transpose(),
                index=self.wavelength,
            )
        )

    def process(self, output_path: str = "") -> None:
        self.export(output_path)

    def export(self, output_path: str = "") -> None:
        config = copy.deepcopy(self.config)
        if output_path == "":
            output_path = (
                f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}{self.file_type}"
            )

        config["particles"] = dict(
            position=self.spheres.tolist(),
            ref_idx_real=self.refractive_index_interpolated.real.tolist(),
            ref_idx_imag=self.refractive_index_interpolated.imag.tolist(),
        )
        config["initial_field"]["polarization"] = "TE"
        config["parameters"]["wavelength"] = (
            self.wavelength * self.wavelength_scale / self.particles_scale
        ).tolist()
        config["parameters"]["medium"] = dict(
            real=self.medium_refractive_index.real.tolist(),
            imag=self.medium_refractive_index.imag.tolist(),
        )
        config["solver"]["type"] = "GMRES"

        for key, value in config["points"].items():
            config["points"][key] = value.tolist()

        file_type = output_path.split(".")[-1] if output_path else self.file_type
        match file_type:
            case "json":
                with open(output_path, "w") as outfile:
                    json.dump(config, outfile)
            case "yaml" | "yml":
                with open(output_path, "w") as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)
            case "bz2":
                with bz2.BZ2File(output_path, "w") as outfile:
                    _pickle.dump(config, outfile)
            case _:
                raise Exception(
                    "The provided output file needs to be a json or yaml file!"
                )


@jit(parallel=False, fastmath=True)
def sphere_overlap_check(
    positions: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
):
    ind = np.stack(np.triu_indices(radii.size, k=1), axis=-1)
    mask = np.zeros(ind.shape[0], dtype=np.bool_)
    counter = np.zeros(radii.size)
    for i in prange(ind.shape[0]):
        if np.sum((positions[ind[i, 0], :] - positions[ind[i, 1], :]) ** 2) < (
            (radii[ind[i, 0]] + radii[ind[i, 1]]) ** 2
        ):
            # print(f"Overlap between particles {ind[i, 0]} and {ind[i, 1]} detected!")
            mask[i] = True
            counter[ind[i, 0]] += 1
            counter[ind[i, 1]] += 1
    # print(np.sum(mask))
    # print(np.sum(mask) / mask.size)
    # print(ind[mask, :])
    # print(np.sum(counter > 0))
    # print(np.sum(counter > 0) / counter.size)
    # print(np.max(counter))
    return ind[mask, :], counter


# @jit
def fix_overlap(
    positions: npt.NDArray,
    radii: npt.NDArray,
):
    ind, counter = sphere_overlap_check(positions, radii)
    MAX_ITER = 100
    for i in range(MAX_ITER):
        ind, counter = sphere_overlap_check(positions, radii)
        max_count = np.max(counter)
        mask = counter == max_count
        radii[mask] *= OVERLAP_MULTIPLICATOR
        print(
            f"Iteration {i: 3.0f} | relative number of overlaps: {np.sum(counter > 0) / counter.size: 0.6f} | max count: {max_count: 8.0f}"
        )
        if np.sum(counter > 0) == 0:
            print("Done!")
            break
    return radii
