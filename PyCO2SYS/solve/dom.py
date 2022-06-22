# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2022  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Equations and parameters for modelling marine organic matter."""

from autograd import numpy as np


def nica_isotherm(q_max, k_nica, m_het, h_donnan):
    """Calculate the NICA isotherm for the given parameters.

    Parameters
    ----------
    q_max : array_like
        Total available binding sites in each distribution.
    k_nica : array_like
        Median value of the affinity distribution.
    m_het : array_like
        Heterogeneity of the affinity distribution.
    h_donnan : array_like
        Proton concentration at the Donnan phase.

    Returns
    -------
    array_like
        Amount of bound protons in mol/kg.
    """
    assert len(q_max) == len(k_nica) == len(m_het)
    q_h = np.zeros_like(h_donnan)
    for j in range(len(q_max)):
        q_h = q_h + q_max[j] * (k_nica[j] * h_donnan) ** m_het[j] / (
            1 + (k_nica[j] * h_donnan) ** m_het[j]
        )
    return q_h
