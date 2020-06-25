# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Define universal constants."""

# RGasConstant = 83.1451  # ml bar-1 K-1 mol-1, DOEv2 (used up to v1.4.0)
# RGasConstant = 83.14472  # ml bar-1 K-1 mol-1, DOEv3 (never used in PyCO2SYS)
RGasConstant = 83.14462618  # 10^-1 J mol^-1 K^-1 (used from v1.4.1)
# RGasConstant source: https://physics.nist.gov/cgi-bin/cuu/Value?r (2018 CODATA)
# RGasConstant value updated in v1.4.1 for consistency with CO2SYS-MATLAB v3
Tzero = 273.15  # 0 degC in K
