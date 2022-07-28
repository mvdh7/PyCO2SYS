# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2022  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Define universal constants."""

from autograd import numpy as np

# Universal gas constant
RGasConstant_DOEv2 = 83.1451  # ml bar-1 K-1 mol-1, DOEv2 (always used by default)
RGasConstant_DOEv3 = 83.14472  # ml bar-1 K-1 mol-1, DOEv3 (never used in PyCO2SYS)
RGasConstant_CODATA2018 = 83.14462618  # 10^-1 J mol^-1 K^-1 (available from v1.4.1)
# Source: https://physics.nist.gov/cgi-bin/cuu/Value?r (2018 CODATA)
# RGasConstant_CODATA2018 added in v1.4.1 for consistency with CO2SYS-MATLAB v3,
# but the default remains RGasConstant_DOEv2.


def RGasConstant(WhichR):
    """Return the gas constant R in ml / (bar * K * mol)."""
    RGas = np.full(np.shape(WhichR), np.nan)
    RGas = np.where(WhichR == 1, RGasConstant_DOEv2, RGas)  # default, DOEv2
    RGas = np.where(WhichR == 2, RGasConstant_DOEv3, RGas)  # DOEv3
    RGas = np.where(WhichR == 3, RGasConstant_CODATA2018, RGas)  # 2018 CODATA
    return RGas


# Absolute zero
Tzero = 273.15  # 0 degC in K

# Boltzmann constant
k_boltzmann = 1.380649e-23  # m**2 * kg / (s * K)
