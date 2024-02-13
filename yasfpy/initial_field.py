import yasfpy.log as log

import numpy as np


class InitialField:
    """
    Represents an initial field used in a simulation.

    Args:
        beam_width (float): The beam width of the field.
        focal_point (float): The focal point of the field.
        field_type (str, optional): The type of the field. Defaults to "gaussian".
        amplitude (float, optional): The amplitude of the field. Defaults to 1.
        polar_angle (float, optional): The polar angle of the field. Defaults to 0.
        azimuthal_angle (float, optional): The azimuthal angle of the field. Defaults to 0.
        polarization (str, optional): The polarization of the field. Defaults to "TE".

    Attributes:
        field_type (str): The type of the field.
        amplitude (float): The amplitude of the field.
        polar_angle (float): The polar angle of the field.
        azimuthal_angle (float): The azimuthal angle of the field.
        polarization (str): The polarization of the field.
        beam_width (float): The beam width of the field.
        focal_point (float): The focal point of the field.
        log: The logger for logging messages.

    Methods:
        __set_pol_idx: Sets the polarization index based on the polarization type.
        __set_normal_incidence: Sets the normal incidence flag based on the polar angle.
        __setup: Performs the initial setup of the field.

    """

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
            """
            Initialize the InitialField object.

            Args:
                beam_width (float): The beam width of the field.
                focal_point (float): The focal point of the field.
                field_type (str, optional): The type of the field. Defaults to "gaussian".
                amplitude (float, optional): The amplitude of the field. Defaults to 1.
                polar_angle (float, optional): The polar angle of the field. Defaults to 0.
                azimuthal_angle (float, optional): The azimuthal angle of the field. Defaults to 0.
                polarization (str, optional): The polarization of the field. Defaults to "TE".
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

            Returns:
                None
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
                    "{} is not a valid polarization type. Please use TE or TM. Reverting to unpolarized".format(
                        self.polarization
                    )
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
