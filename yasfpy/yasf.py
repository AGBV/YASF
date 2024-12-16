import cProfile
import logging
import pstats

import numpy as np
import pandas as pd
import pyperf
from pydantic import BaseModel

from yasfpy.config import Config
from yasfpy.initial_field import InitialField
from yasfpy.numerics import Numerics
from yasfpy.optics import Optics
from yasfpy.parameters import Parameters
from yasfpy.particles import Particles
from yasfpy.simulation import Simulation
from yasfpy.solver import Solver

# from pyinstrument import Profiler
# from pyinstrument.renderers import SpeedscopeRenderer
# import speedscope


# class YASF(BaseModel):
class YASF:
    config: dict
    path_config: str
    path_cluster: str

    def __init__(
        self,
        path_config: str,
        preprocess: bool = True,
        path_cluster: str = "",
    ):
        # super().__init__(
        #     path_config=path_config,
        #     preprocess=preprocess,
        #     path_cluster=path_cluster,
        # )
        self.path_config = path_config
        self.config = Config(path_config, preprocess, path_cluster)

        self.particles = Particles(
            self.config.spheres[:, 0:3],
            self.config.spheres[:, 3],
            self.config.spheres[:, 4],
            refractive_index_table=self.config.refractive_index_interpolated,
            # refractive_index_table=self.config.material,
        )
        self.initial_field = InitialField(
            beam_width=self.config.config["initial_field"]["beam_width"],
            focal_point=np.array(self.config.config["initial_field"]["focal_point"]),
            polar_angle=self.config.config["initial_field"]["polar_angle"],
            azimuthal_angle=self.config.config["initial_field"]["azimuthal_angle"],
            polarization=self.config.config["initial_field"]["polarization"],
        )
        self.parameters = Parameters(
            wavelength=self.config.wavelength,
            medium_refractive_index=self.config.medium_refractive_index,
            particles=self.particles,
            initial_field=self.initial_field,
        )
        self.solver = Solver(
            solver_type=self.config.config["solver"]["type"],
            tolerance=self.config.config["solver"]["tolerance"],
            max_iter=self.config.config["solver"]["max_iter"],
            restart=self.config.config["solver"]["restart"],
        )
        self.numerics = Numerics(
            lmax=self.config.config["numerics"]["lmax"],
            sampling_points_number=self.config.config["numerics"]["sampling_points"],
            particle_distance_resolution=self.config.config["numerics"][
                "particle_distance_resolution"
            ],
            gpu=self.config.config["numerics"]["gpu"],
            solver=self.solver,
        )
        self.simulation = Simulation(self.parameters, self.numerics)
        self.optics = Optics(self.simulation)

    def run(self, points: np.ndarray = None):
        self.particles.compute_volume_equivalent_area()
        self.numerics.compute_spherical_unity_vectors()
        self.numerics.compute_translation_table()
        self.simulation.compute_mie_coefficients()
        self.simulation.compute_initial_field_coefficients()
        self.simulation.compute_right_hand_side()
        self.simulation.compute_scattered_field_coefficients()
        if self.config.config["optics"]:
            self.optics.compute_cross_sections()
            self.optics.compute_efficiencies()
            self.optics.compute_phase_funcition()

        # NOTE: Legacy, needs to be removed
        if points is not None:
            self.optics.simulation.compute_fields(points)

        if "points" in self.config.config:
            points = np.stack(
                (
                    self.config.config["points"]["x"],
                    self.config.config["points"]["y"],
                    self.config.config["points"]["z"],
                ),
                axis=1,
            )
            self.optics.simulation.compute_fields(points)

    @staticmethod
    def benchmark(config_path: str = None, runner: pyperf.Runner = None):
        if config_path is None:
            raise Exception("Plase provide a config file!")
        if runner is None:
            raise Exception("Plase provide a runner for benchmarking!")
        runner.bench_func("yasf_init", lambda: YASF(config_path))
        yasf_instance = YASF(config_path)
        runner.bench_func("yasf_run", lambda: yasf_instance.run())

    @staticmethod
    def profiler(config_path: str = None, output: str = None):
        if config_path is None:
            raise Exception("Plase provide a config file!")
        with cProfile.Profile() as pr:
            yasf_instance = YASF(config_path)
            yasf_instance.run()
            stats = pstats.Stats(pr).sort_stats(pstats.SortKey.TIME)
            stats.print_stats() if output is None else stats.dump_stats(output)
            return yasf_instance

        # with Profiler() as profiler:
        #     yasf_instance = YASF(config_path)
        #     yasf_instance.run()

        #     with open(output, 'w') as f:
        #         f.write(profiler.render(renderer=SpeedscopeRenderer()))

        # profiler = Profiler()
        # profiler.start()
        # yasf_instance = YASF(config_path)
        # yasf_instance.run()
        # profiler.stop()
        # with open(output, 'w') as f:
        #     f.write(profiler.render(renderer=SpeedscopeRenderer()))

        # with speedscope.track(output):
        #     yasf_instance = YASF(config_path)
        #     yasf_instance.run()

        #     return yasf_instance

        # https://backend.orbit.dtu.dk/ws/portalfiles/portal/5501107/paper.pdf
        # page 4
        # r = np.sqrt(self.particles.geometric_projection / np.pi)
        # im_n = self.config.medium_refractive_index.imag
        # wl = self.config.wavelength
        # alpha = 4 * np.pi * r * im_n / wl
        # gamma = 2 * (1 + (alpha - 1) * np.exp(alpha)) / alpha ** 2
        # correction = np.exp(-alpha) / gamma
        # self.optics.q_sca *= correction
        # print(r)
        # print(alpha)
        # print(gamma)
        # print(correction)
