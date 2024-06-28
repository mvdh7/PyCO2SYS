# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
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
    bio,
    # buffers,
    constants,
    convert,
    engine,
    equilibria,
    gas,
    meta,
    salts,
    solubility,
    solve,
    # uncertainty,
    # upsilon,
)

__author__ = meta.authors
__version__ = meta.version

# Aliases for top-level access
from .engine import CO2SYS, system

# from .engine.nd import CO2SYS as sys
from .engine.system import CO2System
from .meta import hello  # because history

# from .solve.get import speciation
# from .uncertainty import all_OEDG18 as uncertainty_OEDG18


def egrad(g):
    # https://github.com/google/jax/issues/3556#issuecomment-649779759
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped
