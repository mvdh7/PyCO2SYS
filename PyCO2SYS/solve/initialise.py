# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Estimate initial pH values for iterative alkalinity-pH equation solvers."""

from jax import numpy as np
from . import get


def _goodH0_dic(CBAlk, TC, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with TC as the second variable, assuming that
    CBAlk is within a suitable range.

    Follows M13 section 3.2.2 and its implementation in mocsy (OE15).
    """
    c2 = KB * (1 - TB / CBAlk) + K1 * (1 - TC / CBAlk)
    c1 = K1 * (KB * (1 - TB / CBAlk - TC / CBAlk) + K2 * (1 - 2 * TC / CBAlk))
    c0 = K1 * K2 * KB * (1 - (2 * TC + TB) / CBAlk)
    c21min = c2**2 - 3 * c1
    c21min_positive = c21min > 0
    sq21 = np.where(c21min_positive, np.sqrt(c21min), 0.0)
    Hmin = np.where(c2 < 0, (sq21 - c2) / 3, -c1 / (c2 + sq21))
    Hpoly = Hmin**3 + c2 * Hmin**2 + c1 * Hmin + c0
    H0 = np.where(
        c21min_positive & (Hpoly < 0),  # i.e. np.sqrt(c21min) is real
        Hmin + np.sqrt(-Hpoly / sq21),
        1e-7,  # default pH=7 if 2nd order approx has no solution
    )
    H0 = np.where(np.isnan(H0), 1e-7, H0)  # fail-safe
    return H0


def from_dic(CBAlk, TC, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with TC as the second variable.

    Follows M13 section 3.2.2 and its implementation in mocsy (OE15).
    """
    # Logical conditions and defaults from mocsy phsolvers.f90
    H0 = np.where(
        CBAlk <= 0,
        1e-3,  # default pH=3 for negative alkalinity
        1e-10,  # default pH=10 for very high alkalinity relative to DIC
    )
    # Use better estimate if alkalinity in suitable range
    H0 = np.where(
        (CBAlk > 0) & (CBAlk < 2 * TC + TB), _goodH0_dic(CBAlk, TC, TB, K1, K2, KB), H0
    )
    return -np.log10(H0)
