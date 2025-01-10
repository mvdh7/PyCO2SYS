# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Seawater properties with primarily biological consequences."""


def substrate_inhibitor_ratio(HCO3, H_free):
    """Substrate:inhibitor ratio (SIR) of B15 in mol-HCO3−/μmol-H+."""
    return 1e-6 * HCO3 / H_free
