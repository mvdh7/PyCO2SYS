# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
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
    test,
)

__all__ = [
    "api",
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
    "test",
]
__author__ = meta.authors
__version__ = meta.version

# Aliases for top-level access
CO2SYS = engine.CO2SYS
say_hello = meta.say_hello  # because history
