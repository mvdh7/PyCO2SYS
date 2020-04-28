# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for TA-pH solvers."""

from autograd.numpy import log
from autograd import elementwise_grad as egrad
from . import get


def _pHfromTATC_r(pH, TA, TC, FREEtoTOT, Ks, totals):
    """Calculate residual alkalinity from pH and TC for solver `pHfromTATC`."""
    return TA - get.TAfromTCpH(TC, pH, Ks, totals)


# Calculate residual alkalinity slope from pH and TC for solver `pHfromTATC`
_pHfromTATC_s = egrad(_pHfromTATC_r)


def _pHfromTATC_s_approx(TC, H, BAlk, OH, K1, K2, KB):
    """Calculate residual alkalinity slope from pH and TC for solver `pHfromTATC`
    approximately, without using Autograd.
    
    This is the original equation from CO2SYS for MATLAB.  It is not used in PyCO2SYS.
    
    Based on CalculatepHfromTATC, version 04.01, 10-13-96, by Ernie Lewis.
    """
    Denom = H ** 2 + K1 * H + K1 * K2
    return log(10) * (
        TC * K1 * H * (H ** 2 + K1 * K2 + 4 * H * K2) / Denom ** 2
        + BAlk * H / (KB + H)
        + OH
        + H
    )


def pHfromTATC(pH, TA, TC, FREEtoTOT, Ks, totals):
    """Calculate delta-pH from pH and TC for solver `pHfromTATC`."""
    return -(
        _pHfromTATC_r(pH, TA, TC, FREEtoTOT, Ks, totals)
        / _pHfromTATC_s(pH, TA, TC, FREEtoTOT, Ks, totals)
    )


def _pHfromTAfCO2_r(pH, TA, fCO2, FREEtoTOT, Ks, totals):
    """Calculate residual alkalinity from pH and fCO2 for solver `pHfromTAfCO2`."""
    return TA - get.TAfrompHfCO2(pH, fCO2, Ks, totals)


# Calculate residual alkalinity slope from pH and fCO2 for solver `pHfromTAfCO2`
_pHfromTAfCO2_s = egrad(_pHfromTAfCO2_r)


def _pHfromTAfCO2_s_approx(HCO3, CO3, BAlk, H, OH, KB):
    """Calculate residual alkalinity slope from pH and fCO2 for solver `pHfromTAfCO2`
    approximately, without using Autograd.
    
    This is the original equation from CO2SYS for MATLAB.  It is not used in PyCO2SYS.
    
    Based on CalculatepHfromTAfCO2, version 04.01, 10-13-97, by Ernie Lewis.
    """
    return log(10) * (HCO3 + 4 * CO3 + BAlk * H / (KB + H) + OH + H)


def pHfromTAfCO2(pH, TA, fCO2, FREEtoTOT, Ks, totals):
    """Calculate delta-pH from pH and fCO2 for solver `pHfromTAfCO2`."""
    return -(
        _pHfromTAfCO2_r(pH, TA, fCO2, FREEtoTOT, Ks, totals)
        / _pHfromTAfCO2_s(pH, TA, fCO2, FREEtoTOT, Ks, totals)
    )


def _pHfromTACarb_r(pH, TA, CARB, FREEtoTOT, Ks, totals):
    """Calculate residual alkalinity from pH and CARB for solver `pHfromTACarb`."""
    return TA - get.TAfrompHCarb(pH, CARB, Ks, totals)


# Calculate residual alkalinity slope from pH and CARB for solver `pHfromTACarb`
_pHfromTACarb_s = egrad(_pHfromTACarb_r)


def _pHfromTACarb_s_approx(CARB, H, OH, BAlk, K2, KB):
    """Calculate residual alkalinity slope from pH and CARB for solver `pHfromTACarb`
    approximately, without using Autograd.
    
    This is the original equation from CO2SYS for MATLAB.  It is not used in PyCO2SYS.
    
    Based on CalculatepHfromTACarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    return log(10) * (-CARB * H / K2 + BAlk * H / (KB + H) + OH + H)


def pHfromTACarb(pH, TA, CARB, FREEtoTOT, Ks, totals):
    """Calculate delta-pH from pH and CARB for solver `pHfromTACarb`."""
    return -(
        _pHfromTACarb_r(pH, TA, CARB, FREEtoTOT, Ks, totals)
        / _pHfromTACarb_s(pH, TA, CARB, FREEtoTOT, Ks, totals)
    )


def _pHfromTAHCO3_r(pH, TA, HCO3, FREEtoTOT, Ks, totals):
    """Calculate residual alkalinity from pH and HCO3 for solver `pHfromTAHCO3`."""
    return TA - get.TAfrompHHCO3(pH, HCO3, Ks, totals)


# Calculate residual alkalinity slope from pH and HCO3 for solver `pHfromTAHCO3`
_pHfromTAHCO3_s = egrad(_pHfromTAHCO3_r)


def _pHfromTAHCO3_s_approx(HCO3, H, OH, BAlk, K2, KB):
    """Calculate residual alkalinity slope from pH and HCO3 for solver `pHfromTAHCO3`
    approximately, without using Autograd.
    
    This is what the original equation would have been if it were in CO2SYS for MATLAB.
    It is not used in PyCO2SYS.
    """
    return log(10) * (2 * HCO3 * K2 / H + BAlk * H / (KB + H) + OH + H)


def pHfromTAHCO3(pH, TA, HCO3, FREEtoTOT, Ks, totals):
    """Calculate delta-pH from pH and HCO3 for solver `pHfromTAHCO3`."""
    return -(
        _pHfromTAHCO3_r(pH, TA, HCO3, FREEtoTOT, Ks, totals)
        / _pHfromTAHCO3_s(pH, TA, HCO3, FREEtoTOT, Ks, totals)
    )
