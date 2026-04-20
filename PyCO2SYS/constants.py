# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Define universal constants."""

# J mol^-1 K^-1, DOEv2 (previously used by default)
RGasConstant_DOEv2 = 8.31451
# J mol^-1 K^-1, DOEv3 (never used in PyCO2SYS)
RGasConstant_DOEv3 = 8.314472
# J mol^-1 K^-1 (available from v1.4.1)
RGasConstant_CODATA2018 = 8.314462618
# Source: https://physics.nist.gov/cgi-bin/cuu/Value?r (2018 CODATA)
# RGasConstant_CODATA2018 added in v1.4.1 for consistency with CO2SYS-MATLAB v3
# but the default remains RGasConstant_DOEv2.
Tzero = 273.15  # 0 degC in K
