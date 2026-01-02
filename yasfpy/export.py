"""Export container models.

This module defines lightweight Pydantic models used to serialize simulation
inputs/outputs (particles, wavelengths, angular phase functions) to common file
formats.

Notes
-----
The models are intentionally permissive (``arbitrary_types_allowed=True``) and
are primarily used as an interchange format rather than as strict validation of
scientific inputs.
"""

import _pickle
import bz2
import json
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
from scipy.io import savemat


@dataclass
class Particles:
    """Particle geometry and material inputs for export.

    Attributes
    ----------
    position:
        Particle center positions ``(N, 3)``.
    radii:
        Particle radii (length ``N``).
    refractive_index:
        Per-particle complex refractive index. For multi-wavelength simulations,
        each particle may store a list of values.
    radius_of_gyration:
        Radius of gyration of the particle ensemble.
    metadata:
        Optional free-form metadata.
    """

    position: list[list[float]] = Field()
    radii: list[float] = Field()
    refractive_index: list[list[complex]] = Field()
    radius_of_gyration: float = Field()
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Wavelength:
    """Wavelength-dependent scalar outputs for export."""

    value: list[float] = Field(default_factory=list)
    geometric_cross_section: float = Field(default=0.0)
    extinction_cross_section: list[float] = Field(default_factory=list)
    scattering_cross_section: list[float] = Field(default_factory=list)
    extinction_efficiency: list[float] = Field(default_factory=list)
    scattering_efficiency: list[float] = Field(default_factory=list)
    single_scattering_albedo: list[float] = Field(default_factory=list)
    medium_refractive_index: list[float] | list[complex] = Field(default_factory=list)


@dataclass
class Angle:
    """Angle-resolved outputs for export."""

    theta: list[float] = Field(default_factory=list)
    phase_function: list[list[float]] = Field(default_factory=list)
    degree_of_linear_polarization: list[list[float]] = Field(default_factory=list)
    degree_of_linear_polarization_q: list[list[float]] = Field(default_factory=list)
    degree_of_linear_polarization_u: list[list[float]] = Field(default_factory=list)
    degree_of_circular_polarization: list[list[float]] = Field(default_factory=list)
    depolarization_ratio: list[list[float]] = Field(default_factory=list)
    anisotropy_parameter: list[list[float]] = Field(default_factory=list)


@dataclass
class Spatial:
    """Spatially resolved outputs for export."""

    polar: list[float] = Field(default_factory=list)
    azimuthal: list[float] = Field(default_factory=list)
    phase_function: list[list[list[float]]] = Field(default_factory=list)
    degree_of_linear_polarization: list[list[list[float]]] = Field(default_factory=list)
    degree_of_linear_polarization_q: list[list[list[float]]] = Field(
        default_factory=list
    )
    degree_of_linear_polarization_u: list[list[list[float]]] = Field(
        default_factory=list
    )
    degree_of_circular_polarization: list[list[list[float]]] = Field(
        default_factory=list
    )
    depolarization_ratio: list[list[list[float]]] = Field(default_factory=list)
    anisotropy_parameter: list[list[list[float]]] = Field(default_factory=list)


class Export(BaseModel):
    """Serializable export record.

    This is a convenience wrapper that collects all exportable sections.

    Notes
    -----
    The section fields accept either validated dataclass instances or raw Python
    mappings; :meth:`model_post_init` coerces mappings to the corresponding model.
    """

    source: str = Field(default="yasf", pattern=r"yasf|mstm")
    scale: float = Field(default=1e-6)

    particles: Particles | dict[str, Any] = Field(default_factory=dict)
    wavelength: Wavelength | dict[str, Any] = Field(default_factory=dict)
    angle: Angle | dict[str, Any] = Field(default_factory=dict)
    spatial: Spatial | dict[str, Any] = Field(default_factory=dict)

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Coerce mapping sections to their dataclass models."""

        if isinstance(self.particles, dict):
            particles_data = cast(dict[str, Any], self.particles)
            self.particles = Particles(**particles_data)

        if isinstance(self.wavelength, dict):
            wavelength_data = cast(dict[str, Any], self.wavelength)
            self.wavelength = Wavelength(**wavelength_data)

        if isinstance(self.angle, dict):
            angle_data = cast(dict[str, Any], self.angle)
            self.angle = Angle(**angle_data)

        if isinstance(self.spatial, dict):
            spatial_data = cast(dict[str, Any], self.spatial)
            self.spatial = Spatial(**spatial_data)

    def save(self, filename: str | Path) -> None:
        """Write the export record to disk.

        Parameters
        ----------
        filename:
            Output file path. The extension determines the serialization format
            (``.json``, ``.yml``/``.yaml``, ``.pkl``, ``.bz2``, ``.mat``).
        """

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
