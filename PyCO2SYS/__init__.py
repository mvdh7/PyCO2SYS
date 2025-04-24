# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
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
"""
PyCO2SYS
========
Marine carbonate system calculations in Python.
"""

import jax

from . import meta
from .engine import CO2System, sys
from .meta import hello  # because history
from .uncertainty import all_OEDG18 as uncertainty_OEDG18

jax.config.update("jax_enable_x64", True)

say_hello = hello  # for backwards compatibility
__all__ = ["CO2System", "hello", "say_hello", "sys", "uncertainty_OEDG18"]
__author__ = meta.authors
__version__ = meta.version
