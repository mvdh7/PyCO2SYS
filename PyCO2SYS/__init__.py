# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Solve the marine carbonate system and calculate related seawater properties."""

from . import (
    api,
    bio,
    buffers,
    constants,
    convert,
    engine,
    equilibria,
    gas,
    meta,
    original,
    salts,
    solubility,
    solve,
    uncertainty,
)

__all__ = [
    "api",
    "bio",
    "buffers",
    "constants",
    "convert",
    "engine",
    "equilibria",
    "gas",
    "meta",
    "original",
    "salts",
    "solubility",
    "solve",
    "uncertainty",
]
__author__ = meta.authors
__version__ = meta.version

# Aliases for top-level access
from .engine import CO2SYS
from .engine.nd import CO2SYS as sys
from .engine.nd import assemble
from .api import CO2SYS_wrap, CO2SYS_MATLABv3
from .meta import hello  # because history
from .solve.get import speciation
from .api.ezio import ezio
from .uncertainty import all_OEDG18 as uncertainty_OEDG18

# For backwards-compatibility
CO2SYS_nd = sys
say_hello = hello
