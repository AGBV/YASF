"""Yet Another Scattering Framework (YASF).

This package provides a Python interface for configuring and running multiple
scattering simulations, along with utilities for post-processing and exporting
results.

The most common entry point is :class:`~yasfpy.yasf.YASF`.
"""

from .config import Config
from .initial_field import InitialField
from .numerics import Numerics
from .optics import Optics
from .parameters import Parameters
from .particles import Particles
from .simulation import Simulation
from .solver import Solver
from .yasf import YASF

__version__ = "0.0.13"
