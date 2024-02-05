import yasfpy.log as log

import numpy as np


class InitialField:
    def __init__(
        self,
        beam_width,
        focal_point,
        field_type: str = "gaussian",
        amplitude: float = 1,
        polar_angle: float = 0,
        azimuthal_angle: float = 0,
        polarization: str = "TE",
    ):
        self.field_type = field_type
        self.amplitude = amplitude
        self.polar_angle = polar_angle
        self.azimuthal_angle = azimuthal_angle
        self.polarization = polarization
        self.beam_width = beam_width
        self.focal_point = focal_point

        self.log = log.scattering_logger(__name__)
        self.__setup()

    def __set_pol_idx(self):
        if (
            isinstance(self.polarization, str) and self.polarization.lower() == "unp"
        ) or (isinstance(self.polarization, int) and self.polarization == 0):
            # Unpolarized is also present in the MSTM output
            self.pol = 0
        elif (
            isinstance(self.polarization, str) and self.polarization.lower() == "te"
        ) or (isinstance(self.polarization, int) and self.polarization == 1):
            # Coresponds to the perpendicular value found in MSTM
            self.pol = 1
        elif (
            isinstance(self.polarization, str) and self.polarization.lower() == "tm"
        ) or (isinstance(self.polarization, int) and self.polarization == 2):
            # Coresponds to the parallel value found in MSTM
            self.pol = 2
        else:
            self.pol = 0
            self.log.warning(
                "{} is not a valid polarization type. Please use TE or TM. Reverting to unpolarized".format(
                    self.polarization
                )
            )

    def __set_normal_incidence(self):
        self.normal_incidence = np.abs(np.sin(self.polar_angle)) < 1e-5

    def __setup(self):
        self.__set_pol_idx()
        self.__set_normal_incidence()
