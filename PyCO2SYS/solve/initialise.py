# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Estimate initial pH values for iterative TA equation solvers."""

from autograd.numpy import errstate, log10, sqrt, where


@errstate(invalid="ignore")
def _goodH0_CO2(CBAlk, CO2, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with fCO2 as the second variable, assuming
    that CBAlk is within a suitable range.
    
    Inspired by M13, section 3.2.2.
    """
    c2 = KB - (TB * KB + K1 * CO2) / CBAlk
    c1 = -K1 * (2 * K2 * CO2 + KB * CO2) / CBAlk
    c0 = -2 * K1 * K2 * KB * CO2 / CBAlk
    c21min = c2 ** 2 - 3 * c1
    c21min_positive = c21min > 0
    sq21 = where(c21min_positive, sqrt(c21min), 0.0)
    Hmin = where(c2 < 0, -c2 + sq21 / 3, -c1 / (c2 + sq21))
    H0 = where(
        c21min_positive,  # i.e. sqrt(c21min) is real
        Hmin + sqrt(-(c2 * Hmin ** 2 + c1 * Hmin + c0) / sq21),
        1e-7,  # default pH=7 if 2nd order approx has no solution
    )
    return H0


@errstate(invalid="ignore")
def fromCO2(CBAlk, CO2, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with fCO2 as the second variable.
    
    Inspired by M13, section 3.2.2.
    """
    H0 = where(
        CBAlk > 0,
        _goodH0_CO2(CBAlk, CO2, TB, K1, K2, KB),
        1e-3,  # default pH=3 for negative alkalinity
    )
    return -log10(H0)


@errstate(invalid="ignore")
def _goodH0_TC(CBAlk, TC, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with TC as the second variable, assuming that
    CBAlk is within a suitable range.
    
    Follows M13 section 3.2.2 and its implementation in mocsy (OE15).
    """
    c2 = KB * (1 - TB / CBAlk) + K1 * (1 - TC / CBAlk)
    c1 = K1 * (KB * (1 - TB / CBAlk - TC / CBAlk) + K2 * (1 - 2 * TC / CBAlk))
    c0 = K1 * K2 * KB * (1 - (2 * TC + TB) / CBAlk)
    c21min = c2 ** 2 - 3 * c1
    c21min_positive = c21min > 0
    sq21 = where(c21min_positive, sqrt(c21min), 0.0)
    Hmin = where(c2 < 0, -c2 + sq21 / 3, -c1 / (c2 + sq21))
    H0 = where(
        c21min_positive,  # i.e. sqrt(c21min) is real
        Hmin + sqrt(-(c2 * Hmin ** 2 + c1 * Hmin + c0) / sq21),
        1e-7,  # default pH=7 if 2nd order approx has no solution
    )
    return H0


@errstate(invalid="ignore")
def fromTC(CBAlk, TC, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with TC as the second variable.
    
    Follows M13 section 3.2.2 and its implementation in mocsy (OE15).
    """
    # Logical conditions and defaults from mocsy phsolvers.f90
    H0 = where(
        CBAlk <= 0,
        1e-3,  # default pH=3 for negative alkalinity
        1e-10,  # default pH=10 for very high alkalinity relative to DIC
    )
    F = (CBAlk > 0) & (CBAlk < 2 * TC + TB)
    if any(F):  # use better estimate if alkalinity in suitable range
        H0 = where(F, _goodH0_TC(CBAlk, TC, TB, K1, K2, KB), H0)
    return -log10(H0)


def _goodH0_CO3(CBAlk, CARB, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with carbonate ion as the second variable, 
    assuming that CBAlk is within a suitable range.
    
    Inspired by M13, section 3.2.2.
    """
    a = CARB
    b = CARB * KB + K2 * (2 * CARB - CBAlk)
    c = K2 * KB * (2 * CARB + TB - CBAlk)
    H0 = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return H0


@errstate(invalid="ignore")
def fromCO3(CBAlk, CARB, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with carbonate ion as the second variable.
    
    Inspired by M13, section 3.2.2.
    """
    H0 = where(
        CBAlk > 2 * CARB + TB,
        _goodH0_CO3(CBAlk, CARB, TB, K1, K2, KB),
        1e-10,  # default pH=10 for low alkalinity
    )
    return -log10(H0)


def _goodH0_HCO3(CBAlk, HCO3, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with bicarbonate ion as the second variable, 
    assuming that CBAlk is within a suitable range.
    
    Inspired by M13, section 3.2.2.
    """
    a = HCO3 - CBAlk
    b = KB * (HCO3 + TB - CBAlk) + 2 * K2 * HCO3
    c = 2 * K2 * KB * HCO3
    H0 = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return H0


@errstate(invalid="ignore")
def fromHCO3(CBAlk, HCO3, TB, K1, K2, KB):
    """Find initial value for TA-pH solver with bicarbonate ion as the second variable.
    
    Inspired by M13, section 3.2.2.
    """
    H0 = where(
        CBAlk > HCO3,
        _goodH0_HCO3(CBAlk, HCO3, TB, K1, K2, KB),
        1e-3,  # default pH=3 for low alkalinity
    )
    return -log10(H0)
