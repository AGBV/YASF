"""High-level YASF front door.

The :class:`~yasfpy.yasf.YASF` class is a convenience wrapper that wires together
configuration parsing, particle geometry, solver/numerics settings, the core
simulation, and optics post-processing.

Notes
-----
This module currently provides a lightweight imperative API (not a Pydantic
model). The public surface is intended to stay stable even if underlying
components evolve.
"""

# ==============================================================================
#   __   __   ___    ____
#   \ \ / /  / _ \  / ___|   Y A S F
#    \ V /  | | | | \___ \   Yet Another Scattering Framework
#     | |   | |_| |  ___) |  (classic ASCII edition)
#     |_|    \___/  |____/
#
#   Joke from my friends (EE): "motorboating" oscillation
#   Low-frequency parasitic oscillation in an amplifier/power supply loop:
#   put a scope on it and it goes "putt-putt-putt".
# ==============================================================================

import cProfile
import logging
import pstats
from functools import cached_property

import numpy as np
import pyperf

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
    """Run a YASF simulation from a configuration file.

    Parameters
    ----------
    path_config:
        Path to the YAML/JSON configuration file.
    preprocess:
        If `True`, performs preprocessing steps during configuration loading
        (e.g., refractive-index interpolation).
    path_cluster:
        Optional path to the particle cluster/geometry file. If empty, the path
        is taken from the configuration.
    quiet:
        If `True`, reduces logging verbosity.

    Notes
    -----
    This class constructs the core objects (:class:`~yasfpy.simulation.Simulation`,
    :class:`~yasfpy.optics.Optics`, etc.) eagerly in ``__init__``.
    """

    path_config: str
    path_cluster: str
    preprocess: bool
    _config: Config | None
    # preprocess: bool = Field(default=True)
    # path_cluster: str = Field(default="")
    # _config: Config | None = PrivateAttr(default=None)

    def __init__(
        self,
        path_config: str,
        preprocess: bool = True,
        path_cluster: str = "",
        *,
        cluster_scale: float | None = None,
        cluster_dimensional_scale: float | None = None,
        quiet: bool = False,
    ):
        """Initialize the simulation pipeline from configuration."""
        # def model_post_init(self, __context: Any) -> None:
        self.path_config = path_config
        self.preprocess = preprocess
        self.path_cluster = path_cluster
        self.quiet = quiet
        # self.config = Config(path_config, preprocess, path_cluster)
        self._config = Config(
            path_config=self.path_config,
            path_cluster=self.path_cluster,
            preprocess=self.preprocess,
            cluster_scale=cluster_scale,
            cluster_dimensional_scale=cluster_dimensional_scale,
            quiet=self.quiet,
        )

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
        solver_cfg = self.config.config["solver"]
        self.solver = Solver(
            solver_type=solver_cfg["type"],
            tolerance=solver_cfg["tolerance"],
            max_iter=solver_cfg["max_iter"],
            restart=solver_cfg["restart"],
            warm_start=bool(solver_cfg.get("warm_start", True)),
        )
        numerics_cfg = self.config.config["numerics"]
        self.numerics = Numerics(
            lmax=numerics_cfg["lmax"],
            sampling_points_number=numerics_cfg["sampling_points"],
            particle_distance_resolution=numerics_cfg["particle_distance_resolution"],
            gpu=numerics_cfg["gpu"],
            solver=self.solver,
            coupling_backend=str(numerics_cfg.get("coupling_backend", "dense")),
            coupling_tile_size=int(numerics_cfg.get("coupling_tile_size", 64)),
            coupling_near_field_radius=numerics_cfg.get("coupling_near_field_radius"),
        )
        self.simulation = Simulation(self.parameters, self.numerics)
        self.optics = Optics(self.simulation)

    @cached_property
    def config(self) -> Config:
        """Loaded :class:`~yasfpy.config.Config` instance (cached)."""
        if self._config is None:
            self._config = Config(
                path_config=self.path_config,
                path_cluster=self.path_cluster,
                preprocess=self.preprocess,
                cluster_scale=None,
                cluster_dimensional_scale=None,
                quiet=self.quiet,
            )
        return self._config

    def run(self, points: np.ndarray | None = None) -> None:
        """Execute the simulation and (optionally) compute derived optics.

        Parameters
        ----------
        points:
            Optional Cartesian coordinates ``(M, 3)`` at which the total/scattered
            fields are evaluated. This is a legacy path and may be removed.
        """
        self.particles.compute_volume_equivalent_area()
        self.numerics.compute_spherical_unity_vectors()
        self.numerics.compute_translation_table()
        self.simulation.compute_mie_coefficients()
        self.simulation.compute_initial_field_coefficients()
        self.simulation.compute_right_hand_side()
        self.simulation.compute_scattered_field_coefficients()
        optics_cfg = self.config.config.get("optics", True)
        if optics_cfg:
            if isinstance(optics_cfg, dict):
                enabled = bool(optics_cfg.get("enabled", True))
                do_cross_sections = bool(optics_cfg.get("cross_sections", True))
                do_phase_function = bool(optics_cfg.get("phase_function", True))
            else:
                enabled = True
                do_cross_sections = True
                do_phase_function = True

            if enabled and do_cross_sections:
                self.optics.compute_cross_sections()
                self.optics.compute_efficiencies()

            if enabled and do_phase_function:
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

    def export(self) -> None:
        """Export results to a serializable record.

        Notes
        -----
        Not implemented yet. Intended to use :class:`yasfpy.export.Export`.
        """
        raise NotImplementedError(
            "Export needs to be implemented yet using the Export class!"
        )

    @staticmethod
    def benchmark(
        config_path: str | None = None,
        runner: pyperf.Runner | None = None,
    ) -> None:
        """Benchmark initialization and run time with :mod:`pyperf`."""
        if config_path is None:
            raise Exception("Plase provide a config file!")
        if runner is None:
            raise Exception("Plase provide a runner for benchmarking!")
        runner.bench_func("yasf_init", lambda: YASF(config_path, quiet=True))
        yasf_instance = YASF(config_path, quiet=True)
        runner.bench_func("yasf_run", lambda: yasf_instance.run())

    @staticmethod
    def profiler(
        config_path: str | None = None,
        *,
        cluster_path: str = "",
        output: str | None = None,
        sort: str = "cumtime",
        limit: int = 50,
        warmup: int = 1,
        include_init: bool = False,
        quiet: bool = True,
    ) -> "YASF":
        """Profile a run with :mod:`cProfile`.

        Parameters
        ----------
        config_path:
            Path to the configuration file.
        cluster_path:
            Optional override for the cluster/geometry path.
        output:
            If given, dumps stats to this file (``pstats`` format). If omitted,
            prints stats to stdout.
        sort:
            Sorting key for ``pstats`` (default ``"cumtime"``).
        limit:
            Limit number of printed lines.
        warmup:
            Number of warm-up runs before profiling.
        include_init:
            If `True`, include object construction in the profile.
        quiet:
            If `True`, reduce logging verbosity.

        Returns
        -------
        YASF
            The profiled simulation instance.
        """
        if config_path is None:
            raise ValueError("Please provide a config file path")

        sort_key_map = {
            "time": pstats.SortKey.TIME,
            "cumtime": pstats.SortKey.CUMULATIVE,
            "calls": pstats.SortKey.CALLS,
        }
        if sort not in sort_key_map:
            raise ValueError(
                f"Unsupported sort='{sort}'. Use one of: {sorted(sort_key_map)}"
            )

        if include_init:
            with cProfile.Profile() as pr:
                yasf_instance = YASF(
                    path_config=config_path,
                    path_cluster=cluster_path,
                    quiet=quiet,
                )
                for _ in range(max(int(warmup), 0)):
                    yasf_instance.run()

                stats = pstats.Stats(pr).sort_stats(sort_key_map[sort])
                if output is None:
                    stats.print_stats(int(limit))
                else:
                    stats.dump_stats(output)
                return yasf_instance

        yasf_instance = YASF(
            path_config=config_path,
            path_cluster=cluster_path,
            quiet=quiet,
        )
        for _ in range(max(int(warmup), 0)):
            yasf_instance.run()

        with cProfile.Profile() as pr:
            yasf_instance.run()

        stats = pstats.Stats(pr).sort_stats(sort_key_map[sort])
        if output is None:
            stats.print_stats(int(limit))
        else:
            stats.dump_stats(output)
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
