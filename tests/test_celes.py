import unittest
import glob
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
from astropy.io import fits

from yasfpy.particles import Particles
from yasfpy.initial_field import InitialField
from yasfpy.parameters import Parameters
from yasfpy.solver import Solver
from yasfpy.numerics import Numerics
from yasfpy.simulation import Simulation


class TestCELES(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        cls.path = f"tests/data/celes_*.fits"
        cls.id_regex = r"celes_(.+)\.fits"
        cls.relative_precision = 2e-3

        p = re.compile(cls.id_regex)
        for data_file in glob.glob(cls.path):
            res = p.search(data_file)
            input_file = res.group(1)
            cls.data[input_file] = {}

            f = fits.open(data_file)
            positions = np.array([f[1].data[c] for c in ["x", "y", "z"]]).transpose()
            radii = f[1].data["r"]
            refractive_indices = np.array(
                [f[1].data[c] for c in ["n", "k"]]
            ).transpose()

            tolerance = f[6].header["HIERARCH TOLERANCE"]
            max_iter = int(f[6].header["HIERARCH MAX ITER"])
            restart = int(f[6].header["HIERARCH RESTART"])

            npol = int(f[7].header["HIERARCH NUMBER POLAR ANGLES"])
            nazi = int(f[7].header["HIERARCH NUMBER AZIMUTHAL ANGLES"])
            pdr = int(f[7].header["HIERARCH PARTICLE DISTANCE RESOLUTION"])

            cls.data[input_file]["particles"] = Particles(
                positions, radii, refractive_indices
            )
            cls.data[input_file]["initial_field"] = InitialField(
                beam_width=f[3].header["HIERARCH BEAM WIDTH"],
                focal_point=np.array((0, 0, 0)),
                polar_angle=f[3].header["HIERARCH POLAR ANGLE"],
                azimuthal_angle=f[3].header["HIERARCH AZIMUTHAL ANGLE"],
                polarization=f[3].header["HIERARCH POLARIZATION"],
            )
            cls.data[input_file]["parameters"] = Parameters(
                wavelength=f[4].data["wavelength"],
                medium_refractive_index=f[4].data["medium_refractive_index"],
                particles=cls.data[input_file]["particles"],
                initial_field=cls.data[input_file]["initial_field"],
            )

            cls.data[input_file]["solver"] = Solver(
                solver_type="lgmres",
                tolerance=tolerance,
                max_iter=max_iter,
                restart=restart,
            )
            cls.data[input_file]["numerics"] = Numerics(
                lmax=int(f[7].header["HIERARCH L MAX"]),
                polar_angles=f[7].data["polar_angles_array"][:npol],
                azimuthal_angles=f[7].data["azimuthal_angles_array"][:nazi],
                gpu=True,
                particle_distance_resolution=pdr,
                solver=cls.data[input_file]["solver"],
            )

            cls.data[input_file]["simulation"] = Simulation(
                cls.data[input_file]["parameters"], cls.data[input_file]["numerics"]
            )

            cls.data[input_file]["numerics"].compute_translation_table(force_compute=True)

            cls.data[input_file]["simulation"].compute_mie_coefficients()
            cls.data[input_file]["simulation"].compute_initial_field_coefficients()
            cls.data[input_file]["simulation"].compute_right_hand_side()

            cls.data[input_file]["simulation"].compute_scattered_field_coefficients()
            # cls.data[input_file]['simulation'].compute_fields(np.array(f[9].data['points']))

            f.close()

    def test_initial_field_coefficients_planewave(self):
        p = re.compile(self.id_regex)
        for data_file in glob.glob(self.path):
            res = p.search(data_file)
            input_file = res.group(1)

            f = fits.open(data_file)

            initial_field_coefficients = np.array(
                f[8].data[0, :] + 1j * f[8].data[1, :]
            )

            if len(initial_field_coefficients.shape) < 3:
                initial_field_coefficients = initial_field_coefficients[
                    :, :, np.newaxis
                ]

            np.testing.assert_allclose(
                self.data[input_file]["simulation"].initial_field_coefficients,
                initial_field_coefficients,
                self.relative_precision,
                0,
                True,
                "The initial field coefficients do not match.",
            )

            f.close()

    def test_scattered_field_coefficients(self):
        p = re.compile(self.id_regex)
        for data_file in glob.glob(self.path):
            res = p.search(data_file)
            input_file = res.group(1)

            f = fits.open(data_file)

            scattered_field_coefficients = np.array(
                f[8].data[2, :] + 1j * f[8].data[3, :]
            )

            if len(scattered_field_coefficients.shape) < 3:
                scattered_field_coefficients = scattered_field_coefficients[
                    :, :, np.newaxis
                ]

            np.testing.assert_allclose(
                self.data[input_file]["simulation"].scattered_field_coefficients,
                scattered_field_coefficients,
                self.relative_precision,
                0,
                True,
                "The scattered field cofficients do not match.",
            )

            f.close()

    # def test_scattered_field(self):
    #     from plotly.subplots import make_subplots
    #     import plotly.graph_objects as go
    #     import pickle

    #     p = re.compile(self.id_regex)
    #     for data_file in glob.glob(self.path):
    #         res = p.search(data_file)
    #         input_file = res.group(1)

    #         f = fits.open(data_file)

    #         scattered_field = f[9].data['scattered']
    #         # pickle.dump( self.data[input_file]['simulation'].scattered_field[0,:], open( "scattered_field.p", "wb" ) )
    #         # with open( "scattered_field.p", "rb" ) as pick:
    #         #     scattered_field_load = pickle.load( pick )
    #         #     # print(scattered_field_load)
    #         scattered_field_load = self.data[input_file]['simulation'].scattered_field[0,:]

    #         vals = np.linalg.norm(np.abs(scattered_field_load), axis=1)

    #         vals2 = np.linalg.norm(np.abs(scattered_field), axis=1)

    #         fig = make_subplots(rows=1, cols=2)

    #         fig.add_trace(
    #             go.Heatmap(
    #                 x = f[9].data['points'][:,0],
    #                 y = f[9].data['points'][:,2],
    #                 z = vals,
    #                 type = 'heatmap',
    #                 colorscale = 'Viridis',
    #                 zmax = 2,
    #                 zmin = 0
    #             ), row=1, col=1
    #         )

    #         fig.add_trace(
    #             go.Heatmap(
    #                 x = self.data[input_file]['simulation'].sampling_points[:,0],
    #                 y = self.data[input_file]['simulation'].sampling_points[:,2],
    #                 z = vals2 / vals,
    #                 type = 'heatmap',
    #                 colorscale = 'Viridis',
    #                 zmax = 2,
    #                 zmin = 0
    #             ), row=1, col=2
    #         )

    #         fig.update_yaxes(
    #             scaleanchor="x",
    #             scaleratio=1,
    #         )
    #         fig.show()

    #         # np.testing.assert_allclose(self.data[input_file]['simulation'].scattered_field[0,:], scattered_field, self.relative_precision, 0, True, 'The scattered field does not match.')

    #         f.close()
