# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Define universal constants."""

from jax import numpy as np

RGasConstant_DOEv2 = 83.1451  # ml bar-1 K-1 mol-1, DOEv2 (always used by default)
RGasConstant_DOEv3 = 83.14472  # ml bar-1 K-1 mol-1, DOEv3 (never used in PyCO2SYS)
RGasConstant_CODATA2018 = 83.14462618  # 10^-1 J mol^-1 K^-1 (available from v1.4.1)
# Source: https://physics.nist.gov/cgi-bin/cuu/Value?r (2018 CODATA)
# RGasConstant_CODATA2018 added in v1.4.1 for consistency with CO2SYS-MATLAB v3,
# but the default remains RGasConstant_DOEv2.
Tzero = 273.15  # 0 degC in K
