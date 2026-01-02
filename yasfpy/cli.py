# Guide: https://medium.com/clarityai-engineering/how-to-create-and-distribute-a-minimalist-cli-tool-with-python-poetry-click-and-pipx-c0580af4c026
import bz2
import logging
import sys
from pathlib import Path

import _pickle
import click

# from rich.traceback import install

from yasfpy import YASF
from yasfpy.benchmark import MSTM4Manager

formatter = logging.Formatter("%(levelname)s (%(name)s): %(message)s")
console = logging.StreamHandler()
console.setFormatter(formatter)
logger = logging.getLogger("yasfpy")
logger.addHandler(console)
logger.setLevel(logging.INFO)

# install(show_locals=True)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--config",
    required=True,
    type=str,
    help="Specify the path to the config file to be used.",
)
@click.option(
    "--cluster",
    type=str,
    default="",
    help="File path for particle cluster specifications. Overrides the provided path in the config.",
)
@click.option(
    "--cluster-scale",
    type=float,
    default=1.0,
    help="Scaling of the provided file. Overrides the provivded scaling in the config.",
)
@click.option(
    "--cluster-dimensional-scale",
    type=float,
    default=1.0,
    help="Scale the dimensions and radii the of the cluster. Overrides the provided dimensional scaling in the config.",
)
@click.option(
    "--backend",
    type=click.Choice(["yasf", "mstm"], case_sensitive=False),
    default="yasf",
    help="Specify an alternative backend to be used, instead of yasf itself.",
)
def compute(
    config: str,
    cluster: str,
    cluster_scale: float,
    cluster_dimensional_scale: float,
    backend: str,
) -> None:
    match backend:
        case "yasf":
            handler = YASF(path_config=config, path_cluster=cluster)
            handler.run()
        case "mstm":
            handler = MSTM4Manager(
                path_config=config,
                path_cluster=cluster,
                cluster_scale=cluster_scale,
                cluster_dimensional_scale=cluster_dimensional_scale,
                parallel=8,
                random_orientation=False,
                incidence_average=True,
                number_incident_directions=40,
                scattering_map_model=False,
                azimuthal_average=True,
            )
            handler.run(cleanup=False)
    handler.export()
    # print(handler.output)

    # cm^2
    # handler.optics.c_ext = (
    #     handler.optics.c_ext
    #     * handler.config.config["particles"]["geometry"]["scale"] ** 2
    # )
    # handler.optics.c_sca = (
    #     handler.optics.c_sca
    #     * handler.config.config["particles"]["geometry"]["scale"] ** 2
    # )

    # plot_data = dict(
    #     particles=dict(
    #         position=handler.particles.position,
    #         radii=handler.particles.r,
    #         # material_idx=np.array(handler.parameters.ref_idx_table[0]).transpose(),
    #         # material=handler.config.material[0]["material"],
    #     ),
    #     wavelength=dict(
    #         value=handler.parameters.wavelength,
    #         data=dict(
    #             extinction_cross_section=handler.optics.c_ext,
    #             scattering_cross_section=handler.optics.c_sca,
    #             extinction_efficiency=handler.optics.q_ext,
    #             scattering_efficiency=handler.optics.q_sca,
    #             single_scattering_albedo=handler.optics.albedo,
    #             medium_idx=handler.parameters.medium_refractive_index,
    #         ),
    #     ),
    #     angle=dict(
    #         value=handler.optics.scattering_angles,
    #         data=dict(
    #             polar_angles=handler.optics.simulation.numerics.polar_angles,
    #             azimuthal_angles=handler.optics.simulation.numerics.azimuthal_angles,
    #             phase_function=dict(
    #                 normal=handler.optics.phase_function,
    #                 spatial=handler.optics.phase_function_3d,
    #                 legendre=handler.optics.phase_function_legendre_coefficients,
    #             ),
    #             degree_of_linear_polarization=dict(
    #                 normal=handler.optics.degree_of_linear_polarization,
    #                 spatial=handler.optics.degree_of_linear_polarization_3d,
    #             ),
    #             degree_of_linear_polarization_q=dict(
    #                 normal=handler.optics.degree_of_linear_polarization_q,
    #                 spatial=handler.optics.degree_of_linear_polarization_q_3d,
    #             ),
    #             degree_of_linear_polarization_u=dict(
    #                 normal=handler.optics.degree_of_linear_polarization_u,
    #                 spatial=handler.optics.degree_of_linear_polarization_u_3d,
    #             ),
    #             degree_of_circular_polarization=dict(
    #                 normal=handler.optics.degree_of_circular_polarization,
    #                 spatial=handler.optics.degree_of_circular_polarization_3d,
    #             ),
    #         ),
    #     ),
    # )

    # with bz2.BZ2File(str(handler.config.output_filename), "w") as f:
    #     _pickle.dump(plot_data, f)
    print("Done")


@cli.command(help="""Explore data using Streamlit""")
@click.option(
    "--path",
    type=str,
    default="",
    help="Path where to look for data files to be displayed",
)
def explore(path: str = ""):  # pragma: no cover
    try:
        from streamlit import runtime
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:
        raise click.ClickException(
            "Streamlit is not installed. Install with 'pip install yasfpy[explore]' to enable 'yasf explore'."
        ) from exc

    if not runtime.exists():
        print(Path(__file__).parent)
        sys.argv = [
            "streamlit",
            "run",
            f"{Path(__file__).parent}/apps/main.py",
            "--",
            "--path",
            path,
        ]
        sys.exit(stcli.main())
