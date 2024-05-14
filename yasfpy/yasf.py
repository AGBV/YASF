import os
import json
import yaml
import warnings
import logging

import numpy as np
import pandas as pd

from yasfpy.particles import Particles
from yasfpy.initial_field import InitialField
from yasfpy.parameters import Parameters
from yasfpy.solver import Solver
from yasfpy.numerics import Numerics
from yasfpy.simulation import Simulation
from yasfpy.optics import Optics


class YASF:
    config: dict = None

    def __init__(self, path_config: str, path_particles: str = None):
        match path_config.split(".")[-1]:
            case "json":
                with open(path_config) as data:
                    self.config = json.load(data)
            case "yaml" | "yml":
                with open(path_config) as data:
                    self.config = yaml.safe_load(data)
            case _:
                raise Exception(
                    "The provided config file needs to be a json or yaml file!"
                )

        self.log = logging.getLogger(self.__class__.__module__)
        self.__setup()

    def __setup(self):
        material = Particles.generate_refractive_index_table(
            self.config["particles"]["material"]
        )
        delim = (
            self.config["particles"]["geometry"]["delimiter"]
            if "delimiter" in self.config["particles"]["geometry"]
            else ","
        )
        delim = "\s+" if delim == "whitespace" else delim
        spheres = pd.read_csv(
            self.config["particles"]["geometry"]["file"], header=None, sep=delim
        )
        if spheres.shape[1] < 4:
            raise Exception(
                "The particle geometry file needs at least 4 columns (x, y, z, r) and an optinal refractive index column"
            )
        elif spheres.shape[1] == 4:
            warnings.warn(
                "4 columns have been provided. Implying that all particles belong to the same material."
            )
            spheres[4] = np.zeros((spheres.shape[0], 1))
        elif spheres.shape[1] >= 5:
            warnings.warn(
                "More than 5 columns have been provided. Everything after the 5th will be ignored!"
            )
        spheres = spheres.to_numpy()

        if isinstance(self.config["parameters"]["wavelength"], list):
            wavelength = self.config["parameters"]["wavelength"]
        elif isinstance(self.config["parameters"]["wavelength"], dict):
            wavelength = np.arange(
                self.config["parameters"]["wavelength"]["start"],
                self.config["parameters"]["wavelength"]["stop"],
                self.config["parameters"]["wavelength"]["step"],
            )
        else:
            raise Exception(
                "Please provide the wavelength as an array, or the (start, stop, step) linspace parameters."
            )
        medium_url = (
            self.config["parameters"]["medium"]["url"]
            if "url" in self.config["parameters"]["medium"]
            else self.config["parameters"]["medium"]
        )
        medium = Particles.generate_refractive_index_table([medium_url])
        scale = (
            self.config["parameters"]["medium"]["scale"]
            if "scale" in self.config["parameters"]["medium"]
            else 1
        )
        medium_refractive_index = np.interp(
            wavelength / scale,
            medium[0]["ref_idx"]["wavelength"],
            medium[0]["ref_idx"]["n"] + 1j * medium[0]["ref_idx"]["k"],
        )

        self.particles = Particles(
            spheres[:, 0:3],
            spheres[:, 3],
            spheres[:, 4],
            refractive_index_table=material,
        )
        self.initial_field = InitialField(
            beam_width=self.config["initial_field"]["beam_width"],
            focal_point=np.array(self.config["initial_field"]["focal_point"]),
            polar_angle=self.config["initial_field"]["polar_angle"],
            azimuthal_angle=self.config["initial_field"]["azimuthal_angle"],
            polarization=self.config["initial_field"]["polarization"],
        )
        self.parameters = Parameters(
            wavelength=wavelength,
            medium_refractive_index=medium_refractive_index,
            particles=self.particles,
            initial_field=self.initial_field,
        )
        self.solver = Solver(
            solver_type=self.config["solver"]["type"],
            tolerance=self.config["solver"]["tolerance"],
            max_iter=self.config["solver"]["max_iter"],
            restart=self.config["solver"]["restart"],
        )
        self.numerics = Numerics(
            lmax=self.config["numerics"]["lmax"],
            sampling_points_number=self.config["numerics"]["sampling_points"],
            particle_distance_resolution=self.config["numerics"][
                "particle_distance_resolution"
            ],
            gpu=self.config["numerics"]["gpu"],
            solver=self.solver,
        )
        self.simulation = Simulation(self.parameters, self.numerics)
        self.optics = Optics(self.simulation)

        folder = (
            self.config["output"]["folder"]
            if "folder" in self.config["output"]
            else "."
        )
        folder = os.sep.join(folder.replace("\\", "/").split("/"))

        filename = None
        if "file" in self.config["particles"]["geometry"]:
            filename = self.config["particles"]["geometry"]["file"].split(os.sep)[-1]
            filename = filename.split(".")[0]
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
        self.output_filename = (
            os.path.join(folder, filename) if (filename is not None) else None
        )

    def run(self, points: np.ndarray = None):
        self.particles.compute_volume_equivalent_area()
        self.numerics.compute_spherical_unity_vectors()
        self.numerics.compute_translation_table()
        self.simulation.compute_mie_coefficients()
        self.simulation.compute_initial_field_coefficients()
        self.simulation.compute_right_hand_side()
        self.simulation.compute_scattered_field_coefficients()
        self.optics.compute_cross_sections()
        self.optics.compute_phase_funcition()

        if points is not None:
            self.optics.simulation.compute_fields(points)
