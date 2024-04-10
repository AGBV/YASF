#!/usr/bin/env python3
import pyperf
import re
import glob

from astropy.io import fits
import numpy as np

from yasfpy.particles import Particles
from yasfpy.initial_field import InitialField
from yasfpy.parameters import Parameters
from yasfpy.solver import Solver
from yasfpy.numerics import Numerics
from yasfpy.simulation import Simulation

data = {}


def celes_setup():
    pass


def celes_run():
    path = f"tests/data/celes_*.fits"
    id_regex = r"celes_(.+)\.fits"

    p = re.compile(id_regex)
    for data_file in glob.glob(path):
        res = p.search(data_file)
        input_file = res.group(1)

        f = fits.open(data_file)
        positions = np.array([f[1].data[c] for c in ["x", "y", "z"]]).transpose()
        radii = f[1].data["r"]
        refractive_indices = np.array([f[1].data[c] for c in ["n", "k"]]).transpose()

        beam_width = f[3].header["HIERARCH BEAM WIDTH"]
        polar_angle = f[3].header["HIERARCH POLAR ANGLE"]

        wavelength = f[4].data["wavelength"]
        refractive_index = f[4].data["medium_refractive_index"]
        azimuthal_angle = f[3].header["HIERARCH AZIMUTHAL ANGLE"]
        polarization = f[3].header["HIERARCH POLARIZATION"]

        tolerance = f[6].header["HIERARCH TOLERANCE"]
        max_iter = int(f[6].header["HIERARCH MAX ITER"])
        restart = int(f[6].header["HIERARCH RESTART"])

        npol = int(f[7].header["HIERARCH NUMBER POLAR ANGLES"])
        nazi = int(f[7].header["HIERARCH NUMBER AZIMUTHAL ANGLES"])
        pdr = int(f[7].header["HIERARCH PARTICLE DISTANCE RESOLUTION"])

        l_max = int(f[7].header["HIERARCH L MAX"])
        polar_angles = f[7].data["polar_angles_array"][:npol]
        azimuthal_angles = f[7].data["azimuthal_angles_array"][:nazi]

        particles = Particles(positions, radii, refractive_indices)
        initial_field = InitialField(
            beam_width=f[3].header["HIERARCH BEAM WIDTH"],
            focal_point=np.array((0, 0, 0)),
            polar_angle=f[3].header["HIERARCH POLAR ANGLE"],
            azimuthal_angle=f[3].header["HIERARCH AZIMUTHAL ANGLE"],
            polarization=f[3].header["HIERARCH POLARIZATION"],
        )
        parameters = Parameters(
            wavelength=f[4].data["wavelength"],
            medium_refractive_index=f[4].data["medium_refractive_index"],
            particles=particles,
            initial_field=initial_field,
        )

        solver = Solver(
            solver_type="lgmres",
            tolerance=tolerance,
            max_iter=max_iter,
            restart=restart,
        )
        numerics = Numerics(
            lmax=int(f[7].header["HIERARCH L MAX"]),
            polar_angles=f[7].data["polar_angles_array"][:npol],
            azimuthal_angles=f[7].data["azimuthal_angles_array"][:nazi],
            gpu=True,
            particle_distance_resolution=pdr,
            solver=solver,
        )

        simulation = Simulation(parameters, numerics)

        numerics.compute_translation_table()

        simulation.compute_mie_coefficients()
        simulation.compute_initial_field_coefficients()
        simulation.compute_right_hand_side()

        simulation.compute_scattered_field_coefficients()
        # cls.data[input_file]['simulation'].compute_fields(np.array(f[9].data['points']))

        f.close()


runner = pyperf.Runner()
runner.timeit(
    name="sort a sorted list",
    stmt="sorted(s, key=f)",
    setup="f = lambda x: x; s = list(range(1000))",
)
runner.dump("bench2.json")
