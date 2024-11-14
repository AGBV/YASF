import bz2
import logging

import _pickle
import numpy as np
from yasfpy import YASF

formatter = logging.Formatter("%(levelname)s (%(name)s): %(message)s")
console = logging.StreamHandler()
console.setFormatter(formatter)
logger = logging.getLogger("yasfpy")
logger.addHandler(console)
logger.setLevel(logging.INFO)

handler = YASF("graphite.json")
handler.run()

# cm^2
handler.optics.c_ext = (
    handler.optics.c_ext * handler.config.config["particles"]["geometry"]["scale"] ** 2
)
handler.optics.c_sca = (
    handler.optics.c_sca * handler.config.config["particles"]["geometry"]["scale"] ** 2
)

plot_data = dict(
    particles=dict(
        position=handler.particles.position,
        radii=handler.particles.r,
        material_idx=np.array(handler.parameters.ref_idx_table[0]).transpose(),
        material=handler.config.material[0]["material"],
    ),
    wavelength=dict(
        value=handler.parameters.wavelength,
        data=dict(
            extinction_cross_section=handler.optics.c_ext,
            scattering_cross_section=handler.optics.c_sca,
            extinction_efficiency=handler.optics.q_ext,
            scattering_efficiency=handler.optics.q_sca,
            single_scattering_albedo=handler.optics.albedo,
            medium_idx=handler.parameters.medium_refractive_index,
        ),
    ),
    angle=dict(
        value=handler.optics.scattering_angles,
        data=dict(
            polar_angles=handler.optics.simulation.numerics.polar_angles,
            azimuthal_angles=handler.optics.simulation.numerics.azimuthal_angles,
            phase_function=dict(
                normal=handler.optics.phase_function,
                spatial=handler.optics.phase_function_3d,
                legendre=handler.optics.phase_function_legendre_coefficients,
            ),
            degree_of_linear_polarization=dict(
                normal=handler.optics.degree_of_linear_polarization,
                spatial=handler.optics.degree_of_linear_polarization_3d,
            ),
            degree_of_linear_polarization_q=dict(
                normal=handler.optics.degree_of_linear_polarization_q,
                spatial=handler.optics.degree_of_linear_polarization_q_3d,
            ),
            degree_of_linear_polarization_u=dict(
                normal=handler.optics.degree_of_linear_polarization_u,
                spatial=handler.optics.degree_of_linear_polarization_u_3d,
            ),
            degree_of_circular_polarization=dict(
                normal=handler.optics.degree_of_circular_polarization,
                spatial=handler.optics.degree_of_circular_polarization_3d,
            ),
        ),
    ),
)

with bz2.BZ2File(handler.config.output_filename, "w") as f:
    _pickle.dump(plot_data, f)
