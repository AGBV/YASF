# Guide: https://medium.com/clarityai-engineering/how-to-create-and-distribute-a-minimalist-cli-tool-with-python-poetry-click-and-pipx-c0580af4c026

import click

from yasfpy import YASF


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
def compute(config: str, cluster: str) -> None:
    handler = YASF(config, path_cluster=cluster)
    handler.run()
