import _pickle
import bz2
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from scipy.io import savemat
from typing_extensions import Self


@dataclass
class Particles:
    position: list[list[float]] = Field()
    radii: list[float] = Field()
    refractive_index: list[list[complex]] = Field()
    radius_of_gyration: float = Field()
    metadata: dict = Field(default={})

    @model_validator(mode="after")
    def number_of_elements(self) -> Self:
        if len(self.position) != len(self.radii):
            raise ValueError(
                f"Number of elements in position ({len(self.position)}) and radii ({len(self.radii)}) are not compatible"
            )
        return self


@dataclass
class Wavelength:
    value: list[float] = Field(default=[])
    geometric_cross_section: float = Field(default=0.0)
    extinction_cross_section: list[float] = Field(default=[])
    scattering_cross_section: list[float] = Field(default=[])
    extinction_efficiency: list[float] = Field(default=[])
    scattering_efficiency: list[float] = Field(default=[])
    single_scattering_albedo: list[float] = Field(default=[])
    medium_refractive_index: list[float] | list[complex] = Field(default=[])


@dataclass
class Angle:
    theta: list[float] = Field(default=[])
    phase_function: list[list[float]] = Field(default=[])
    degree_of_linear_polarization: list[list[float]] = Field(default=[])
    degree_of_linear_polarization_q: list[list[float]] = Field(default=[])
    degree_of_linear_polarization_u: list[list[float]] = Field(default=[])
    degree_of_circular_polarization: list[list[float]] = Field(default=[])
    depolarization_ratio: list[list[float]] = Field(default=[])
    anisotropy_parameter: list[list[float]] = Field(default=[])


@dataclass
class Spatial:
    polar: list[float] = Field(default=[])
    azimuthal: list[float] = Field(default=[])
    phase_function: list[list[list[float]]] = Field(default=[])
    degree_of_linear_polarization: list[list[list[float]]] = Field(default=[])
    degree_of_linear_polarization_q: list[list[list[float]]] = Field(default=[])
    degree_of_linear_polarization_u: list[list[list[float]]] = Field(default=[])
    degree_of_circular_polarization: list[list[list[float]]] = Field(default=[])
    depolarization_ratio: list[list[list[float]]] = Field(default=[])
    anisotropy_parameter: list[list[list[float]]] = Field(default=[])


class Export(BaseModel):
    source: str = Field(default="yasf", pattern=r"yasf|mstm")
    scale: float = Field(default=1e-6)
    particles: Particles | dict = Field(default={})
    wavelength: Wavelength | dict = Field(default={})
    angle: Angle | dict = Field(default={})
    spatial: Spatial | dict = Field(default={})

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.particles, dict):
            self.particles = Particles(**self.particles)
        if isinstance(self.wavelength, dict):
            self.wavelength = Wavelength(**self.wavelength)
        if isinstance(self.angle, dict):
            self.angle = Angle(**self.angle)
        if isinstance(self.spatial, dict):
            self.spatial = Spatial(**self.spatial)

    def save(self, filename: str | Path) -> None:
        if isinstance(filename, str):
            filename = Path(filename)

        match filename.suffix:
            case ".json":
                with open(filename, "w") as f:
                    json.dump(self.model_dump(), f, indent=4)
            case ".yml" | ".yaml":
                with open(filename, "w") as f:
                    yaml.dump(self.model_dump(), f)
            case ".pkl":
                with open(filename, "wb") as f:
                    _pickle.dump(self.model_dump(), f)
            case ".bz2":
                with bz2.BZ2File(filename, "w") as outfile:
                    _pickle.dump(self.model_dump(), outfile)
            case ".mat":
                savemat(filename, self.model_dump())
            case _:
                raise ValueError(f"Unknown file extension {filename.suffix}")
