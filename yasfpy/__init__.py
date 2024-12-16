from rich.traceback import install

from .config import Config
from .initial_field import InitialField
from .numerics import Numerics
from .optics import Optics
from .parameters import Parameters
from .particles import Particles
from .simulation import Simulation
from .solver import Solver
from .yasf import YASF

install(show_locals=True)
