import yasfpy.log as log

import numpy as np


class InitialField:
    """Represents an object with various parameters for a beam of light."""

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
        """Initializes an object with various parameters for a beam of light.

        Args:
            beam_width (float): The beam width parameter represents the width of the beam. It determines the spread of the beam
                and is typically measured as the full width at half maximum (FWHM) of the beam intensity
                distribution.
            focal_point (tuple): The focal point is the point in space where the beam is focused. It is a coordinate in
                three-dimensional space (x, y, z) that represents the position of the focal point.
            field_type (str, optional): The `field_type` parameter specifies the type of field for the beam. It can be set to
                "gaussian" or any other type of field that is supported by the code. Defaults to "gaussian".
            amplitude (float, optional): The amplitude parameter represents the maximum value or intensity of the beam. It determines
                the overall strength or power of the beam. Defaults to 1.
            polar_angle (float, optional): The `polar_angle` parameter represents the angle between the positive z-axis and the direction
                of propagation of the beam. It is measured in radians. Defaults to 0.
            azimuthal_angle (float, optional): The azimuthal angle is a measure of the angle between the projection of the vector onto the
                xy-plane and the positive x-axis. It determines the orientation of the beam in the xy-plane. Defaults to 0.
            polarization (str, optional): The "polarization" parameter determines the polarization of the beam. It can have two possible
                values: "TE" for transverse electric polarization and "TM" for transverse magnetic polarization. Defaults to "TE".
        """
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
        """
        Sets the polarization index based on the polarization type.

        The polarization index is determined based on the value of the `polarization` attribute.
        If the `polarization` is "unp" or 0, the polarization index is set to 0.
        If the `polarization` is "te" or 1, the polarization index is set to 1.
        If the `polarization` is "tm" or 2, the polarization index is set to 2.
        If the `polarization` is not a valid value, the polarization index is set to 0 and a warning message is logged.
        """
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
                "%s is not a valid polarization type. Please use TE or TM. Reverting to unpolarized",
                self.polarization,
            )

    def __set_normal_incidence(self):
        """
        Sets the normal incidence flag based on the polar angle.

        This method checks the value of the polar angle and determines if it is close to zero.
        If the absolute value of the sine of the polar angle is less than 1e-5, the normal incidence flag is set to True.
        Otherwise, the normal incidence flag is set to False.
        """
        self.normal_incidence = np.abs(np.sin(self.polar_angle)) < 1e-5

    def __setup(self):
        """
        Performs the initial setup of the field.

        This method sets the polarization index and normal incidence for the field.
        """
        self.__set_pol_idx()
        self.__set_normal_incidence()
