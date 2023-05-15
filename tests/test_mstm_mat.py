import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
from astropy.io import fits
from scipy.io import loadmat

from src.particles import Particles
from src.initial_field import InitialField
from src.parameters import Parameters
from src.solver import Solver
from src.numerics import Numerics
from src.simulation import Simulation
from src.optics import Optics

class TestMSTM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        cls.path = f'tests/data/mstm_*.mat'
        cls.id_regex = r'mstm_(.+)\.mat'
        cls.relative_precision = 1e-3

        p = re.compile(cls.id_regex)
        for data_file in glob.glob(cls.path):

            res = p.search(data_file)
            input_file = res.group(1)
            cls.data[input_file] = {}

            data = loadmat(data_file)

            lmax = 4
            spheres = data['spheres']

            wavelength = data['wavelength'].squeeze()
            medium_ref_idx = data['medium_ref_idx'].squeeze()

            polar_angle = data['polar_angle'][0][0]
            azimuthal_angle = data['azimuthal_angle'][0][0]
            polarization = str(data['polarization'][0])
            beam_width = float(data['beamwidth'][0][0])

            solver_type = str(data['solver_type'][0])
            solver_type = 'lgmres'
            tolerance = float(data['tolerance'][0][0])
            max_iter = int(data['max_iter'][0][0])
            restart = int(data['restart'][0][0])

            cls.data[input_file]['particles'] = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
            cls.data[input_file]['initial_field'] = InitialField(
                beam_width=beam_width,
                focal_point=np.array((0,0,0)),
                polar_angle=polar_angle,
                azimuthal_angle=azimuthal_angle,
                polarization=polarization)
            cls.data[input_file]['parameters'] = Parameters(
                wavelength=wavelength,
                medium_refractive_index=medium_ref_idx,
                particles=cls.data[input_file]['particles'],
                initial_field=cls.data[input_file]['initial_field'])
            
            cls.data[input_file]['solver'] = Solver(
                solver_type=solver_type,
                tolerance=tolerance,
                max_iter=max_iter,
                restart=restart)
            cls.data[input_file]['numerics'] = Numerics(
                lmax=lmax,
                sampling_points_number=[360, 181],
                gpu=True,
                particle_distance_resolution=1,
                solver=cls.data[input_file]['solver'])

            cls.data[input_file]['simulation'] = Simulation(
                cls.data[input_file]['parameters'], 
                cls.data[input_file]['numerics'])
            
            cls.data[input_file]['optics'] = Optics(cls.data[input_file]['simulation'])
            

            cls.data[input_file]['numerics'].compute_translation_table()

            cls.data[input_file]['simulation'].compute_mie_coefficients()
            cls.data[input_file]['simulation'].compute_initial_field_coefficients()
            cls.data[input_file]['simulation'].compute_right_hand_side()
            cls.data[input_file]['simulation'].compute_scattered_field_coefficients()

            cls.data[input_file]['optics'].compute_cross_sections()
      

    def test_wavelength_values(self):
        p = re.compile(self.id_regex)
        for data_file in glob.glob(self.path):
            res = p.search(data_file)
            input_file = res.group(1)

            data = loadmat(data_file)

            q_ext_mstm = data['q_ext'].squeeze()
            q_ext_yasf = self.data[input_file]['optics'].c_ext / self.data[input_file]['particles'].geometric_projection

            print('Extinction Efficiency:')
            print(q_ext_mstm)
            print(q_ext_yasf)
            print(q_ext_yasf / q_ext_mstm)

            q_sca_mstm = data['q_sca'].squeeze()
            q_sca_yasf = self.data[input_file]['optics'].c_sca / self.data[input_file]['particles'].geometric_projection

            print('Scattering Efficiency:')
            print(q_sca_mstm)
            print(q_sca_yasf)
            print(q_ext_yasf / q_ext_mstm)

            albedo_mstm = data['albedo'].squeeze()
            albedo_yasf = self.data[input_file]['optics'].albedo

            print('Albedo:')
            print(albedo_mstm)
            print(albedo_yasf)
            print(albedo_yasf / albedo_mstm)

            # print(f[1])

            # initial_field_coefficients   = np.array(f[8].data[0, :] + 1j * f[8].data[1, :])

            # if len(initial_field_coefficients.shape) < 3:
            #     initial_field_coefficients = initial_field_coefficients[:, :, np.newaxis]

            # np.testing.assert_allclose(self.data[input_file]['simulation'].initial_field_coefficients, initial_field_coefficients, self.relative_precision, 0, True, 'The initial field coefficients do not match.')

    # def test_scattered_field_coefficients(self):
    #    p = re.compile(self.id_regex)
    #    for data_file in glob.glob(self.path):
    #        res = p.search(data_file)
    #        input_file = res.group(1)

    #        f = fits.open(data_file)

    #        scattered_field_coefficients = np.array(f[8].data[2, :] + 1j * f[8].data[3, :])

    #        if len(scattered_field_coefficients.shape) < 3:
    #            scattered_field_coefficients = scattered_field_coefficients[:, :, np.newaxis]

    #        np.testing.assert_allclose(self.data[input_file]['simulation'].scattered_field_coefficients, scattered_field_coefficients, self.relative_precision, 0, True, 'The scattered field cofficients do not match.')
     
    #        f.close()
