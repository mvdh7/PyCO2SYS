# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Seawater properties with primarily biological consequences."""


def SIratio(HCO3, pHfree):
    """Substrate:inhibitor ratio (SIR) of B15 in mol-HCO3−/μmol-H+."""
    Hfree = 10.0 ** -pHfree
    return HCO3 / (Hfree * 1e6)
