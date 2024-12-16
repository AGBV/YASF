import bz2
import copy
import json
import logging
import os
from datetime import datetime
from numbers import Number
from pathlib import Path

import _pickle
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from refidxdb import Handler

from yasfpy.functions.misc import generate_refractive_index_table


class Config:
    config: dict = {}
    path_cluster: str = ""
    preprocess: bool = True

    def __init__(
        self, path_config: str, preprocess: bool = True, path_cluster: str = ""
    ):
        # if type(path_config) != str:
        if not isinstance(path_config, str):
            raise Exception("The config file path needs to be a string!")
        _path_config = Path(path_config)
        # self.file_type = path_config.split(".")[-1]
        self.file_type = _path_config.suffix
        match self.file_type:
            case ".json":
                with open(path_config) as data:
                    self.config = json.load(data)
            case ".yaml" | ".yml":
                with open(path_config) as data:
                    self.config = yaml.safe_load(data)
            case _:
                raise Exception(
                    "The provided config file needs to be a json or yaml file!"
                )
        if self.config is None:
            raise Exception(
                f"Could not read config file {path_config}. Check if the file exists."
            )
        self.path_cluster = (
            self.config["particles"]["geometry"]["file"]
            if path_cluster == ""
            else path_cluster
        )
        if not self.path_cluster.startswith("/"):
            self.path_cluster = str(_path_config.parent / self.path_cluster)

        self.log = logging.getLogger(self.__class__.__module__)
        self.__read()
        self.__folder()
        if preprocess:
            self.__interpolate()

    def __read(self):
        # TODO: import of csv files (single column)
        # wavelength
        if isinstance(self.config["parameters"]["wavelength"]["data"], list):
            self.wavelength = self.config["parameters"]["wavelength"]["data"]
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

        # TODO: move the interpolation of data into config and away from the YASF function
        # NOTE: Kinda done, but needs to be checked!
        # refractive indices of particles
        self.material = generate_refractive_index_table(
            [mat["url"] for mat in self.config["particles"]["material"]]
        )
        self.material_scale = [
            mat["scale"] for mat in self.config["particles"]["material"]
        ]

        # refractive index of medium
        medium_url = (
            self.config["parameters"]["medium"]["url"]
            if "url" in self.config["parameters"]["medium"]
            else self.config["parameters"]["medium"]
        )
        self.medium = Handler(url=medium_url)
        # self.medium = generate_refractive_index_table([medium_url])
        # self.medium_scale = (
        #     self.config["parameters"]["medium"]["scale"]
        #     if "scale" in self.config["parameters"]["medium"]
        #     else 1
        # )

        # particle geometry
        delim = (
            self.config["particles"]["geometry"]["delimiter"]
            if "delimiter" in self.config["particles"]["geometry"]
            else ","
        )
        delim = r"\s+" if delim == "whitespace" else delim
        spheres = pd.read_csv(
            # self.config["particles"]["geometry"]["file"],
            self.path_cluster,
            header=None,
            sep=delim,
        )
        if spheres.shape[1] < 4:
            raise Exception(
                "The particle geometry file needs at least 4 columns (x, y, z, r) and an optinal refractive index column"
            )
        elif spheres.shape[1] == 4:
            self.log.info(
                "4 columns have been provided. Implying that all particles belong to the same material."
            )
            spheres[4] = np.zeros((spheres.shape[0], 1))
        elif spheres.shape[1] >= 5:
            self.log.warning(
                "More than 5 columns have been provided. Everything after the 5th will be ignored!"
            )
        self.particles_scale = (
            self.config["particles"]["geometry"]["scale"]
            if "scale" in self.config["particles"]["geometry"]
            else 1
        )
        self.spheres = spheres.to_numpy()
        # NOTE: Scale the distnaces and radii to the wavelength
        # This should make the size parameter correct
        self.spheres[:, :4] = (
            self.spheres[:, :4] * self.particles_scale / self.wavelength_scale
        )
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

    def __folder(self):
        folder = (
            self.config["output"]["folder"]
            if "folder" in self.config["output"]
            else "."
        )
        folder = os.sep.join(folder.replace("\\", "/").split("/"))

        extension = (
            self.config["output"]["extension"]
            if "extension" in self.config["output"]
            else "pbz2"
        )
        # filename = ""
        # if "file" in self.config["particles"]["geometry"]:
        #     filename = self.config["particles"]["geometry"]["file"].split(os.sep)[-1]
        #     filename = filename.split(".")[0]
        filename = self.path_cluster.split(os.sep)[-1]
        # filename = filename.split(".")[0]
        filename = ".".join(filename.split(".")[:-1])
        filename = (
            self.config["output"]["filename"]
            if "filename" in self.config["output"]
            else filename
        )
        filename = (
            self.config["output"]
            if isinstance(self.config["output"], str)
            else filename
        )
        filename = (
            f"{filename}.{extension}" if len(filename.split(".")) == 1 else filename
        )
        self.output_filename = (
            os.path.join(folder, filename) if (filename is not None) else None
        )

    def __interpolate(self):
        refractive_index_interpolated = np.zeros(
            (len(self.material), self.wavelength.size),
            dtype=complex,
        )
        for idx, data in enumerate(self.material):
            table = data["ref_idx"].to_numpy().astype(float)
            refractive_index_interpolated[idx, :] = np.interp(
                self.wavelength * self.wavelength_scale,
                table[:, 0] * self.material_scale[idx],
                table[:, 1] + 1j * table[:, 2],
                left=table[0, 1] + 1j * table[0, 2],
                right=table[-1, 1] + 1j * table[-1, 1],
            )
        self.refractive_index_interpolated = refractive_index_interpolated

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
        self.medium_refractive_index.imag = 0
        # self.medium_refractive_index = np.real(self.medium_refractive_index)
        # print(self.medium_refractive_index)

    def process(self, output_path: str = "") -> None:
        self.export(output_path)

    def export(self, output_path: str = "") -> None:
        config = copy.deepcopy(self.config)
        if output_path == "":
            output_path = (
                f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.{config.file_type}"
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
