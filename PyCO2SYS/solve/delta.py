# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for TA-pH solvers."""

from autograd import numpy as np
from autograd import elementwise_grad as egrad
from . import get


# Set whether to use the approximate slopes [True] or exact (Autograd) slopes [False]
use_approximate_slopes = False


def _pHfromTATC_r(pH, TA, TC, totals, Ks):
    """Calculate residual alkalinity from pH and TC for solver `pHfromTATC`."""
    return get.TAfromTCpH(TC, pH, totals, Ks) - TA


# Calculate residual alkalinity slope from pH and TC for solver `pHfromTATC`
_pHfromTATC_s = egrad(_pHfromTATC_r)


def _pHfromTATC_s_approx(pH, TA, TC, totals, Ks):
    """Calculate residual alkalinity slope from pH and TC for solver `pHfromTATC`
    approximately, without using Autograd.

    This is the original equation from CO2SYS for MATLAB.  It is not used in PyCO2SYS.

    Based on CalculatepHfromTATC, version 04.01, 10-13-96, by Ernie Lewis.
    """
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    KB = Ks["KB"]
    H = 10.0 ** -pH
    BAlk = totals["TB"] * KB / (KB + H)
    OH = Ks["KW"] / H
    Denom = H ** 2 + K1 * H + K1 * K2
    return np.log(10) * (
        TC * K1 * H * (H ** 2 + K1 * K2 + 4 * H * K2) / Denom ** 2
        + BAlk * H / (KB + H)
        + OH
        + H
    )


def pHfromTATC(pH, TA, TC, totals, Ks):
    """Calculate delta-pH from pH and TC for solver `pHfromTATC`."""
    if use_approximate_slopes:
        return -(
            _pHfromTATC_r(pH, TA, TC, totals, Ks)
            / _pHfromTATC_s_approx(pH, TA, TC, totals, Ks)
        )
    else:
        return -(
            _pHfromTATC_r(pH, TA, TC, totals, Ks)
            / _pHfromTATC_s(pH, TA, TC, totals, Ks)
        )


def _pHfromTAfCO2_r(pH, TA, fCO2, totals, Ks):
    """Calculate residual alkalinity from pH and fCO2 for solver `pHfromTAfCO2`."""
    return get.TAfrompHfCO2(pH, fCO2, totals, Ks) - TA


# Calculate residual alkalinity slope from pH and fCO2 for solver `pHfromTAfCO2`
_pHfromTAfCO2_s = egrad(_pHfromTAfCO2_r)


def _pHfromTAfCO2_s_approx(pH, TA, fCO2, totals, Ks):
    """Calculate residual alkalinity slope from pH and fCO2 for solver `pHfromTAfCO2`
    approximately, without using Autograd.

    This is the original equation from CO2SYS for MATLAB.  It is not used in PyCO2SYS.

    Based on CalculatepHfromTAfCO2, version 04.01, 10-13-97, by Ernie Lewis.
    """
    K0 = Ks["K0"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    KB = Ks["KB"]
    H = 10.0 ** -pH
    BAlk = totals["TB"] * KB / (KB + H)
    OH = Ks["KW"] / H
    HCO3 = K0 * K1 * fCO2 / H
    CO3 = K0 * K1 * K2 * fCO2 / H ** 2
    return np.log(10) * (HCO3 + 4 * CO3 + BAlk * H / (KB + H) + OH + H)


def pHfromTAfCO2(pH, TA, fCO2, totals, Ks):
    """Calculate delta-pH from pH and fCO2 for solver `pHfromTAfCO2`."""
    if use_approximate_slopes:
        return -(
            _pHfromTAfCO2_r(pH, TA, fCO2, totals, Ks)
            / _pHfromTAfCO2_s_approx(pH, TA, fCO2, totals, Ks)
        )
    else:
        return -(
            _pHfromTAfCO2_r(pH, TA, fCO2, totals, Ks)
            / _pHfromTAfCO2_s(pH, TA, fCO2, totals, Ks)
        )


def _pHfromTACarb_r(pH, TA, CARB, totals, Ks):
    """Calculate residual alkalinity from pH and CARB for solver `pHfromTACarb`."""
    return get.TAfrompHCarb(pH, CARB, totals, Ks) - TA


# Calculate residual alkalinity slope from pH and CARB for solver `pHfromTACarb`
_pHfromTACarb_s = egrad(_pHfromTACarb_r)


def _pHfromTACarb_s_approx(pH, TA, CARB, totals, Ks):
    """Calculate residual alkalinity slope from pH and CARB for solver `pHfromTACarb`
    approximately, without using Autograd.

    This is the original equation from CO2SYS for MATLAB.  It is not used in PyCO2SYS.

    Based on CalculatepHfromTACarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    K2 = Ks["K2"]
    KB = Ks["KB"]
    H = 10.0 ** -pH
    BAlk = totals["TB"] * KB / (KB + H)
    OH = Ks["KW"] / H
    return np.log(10) * (-CARB * H / K2 + BAlk * H / (KB + H) + OH + H)


def pHfromTACarb(pH, TA, CARB, totals, Ks):
    """Calculate delta-pH from pH and CARB for solver `pHfromTACarb`."""
    if use_approximate_slopes:
        return -(
            _pHfromTACarb_r(pH, TA, CARB, totals, Ks)
            / _pHfromTACarb_s_approx(pH, TA, CARB, totals, Ks)
        )
    else:
        return -(
            _pHfromTACarb_r(pH, TA, CARB, totals, Ks)
            / _pHfromTACarb_s(pH, TA, CARB, totals, Ks)
        )


def _pHfromTAHCO3_r(pH, TA, HCO3, totals, Ks):
    """Calculate residual alkalinity from pH and HCO3 for solver `pHfromTAHCO3`."""
    return get.TAfrompHHCO3(pH, HCO3, totals, Ks) - TA


# Calculate residual alkalinity slope from pH and HCO3 for solver `pHfromTAHCO3`
_pHfromTAHCO3_s = egrad(_pHfromTAHCO3_r)


def _pHfromTAHCO3_s_approx(pH, TA, HCO3, totals, Ks):
    """Calculate residual alkalinity slope from pH and HCO3 for solver `pHfromTAHCO3`
    approximately, without using Autograd.

    This is what the original equation would have been if it were in CO2SYS for MATLAB.
    It is not used in PyCO2SYS.
    """
    K2 = Ks["K2"]
    KB = Ks["KB"]
    H = 10.0 ** -pH
    BAlk = totals["TB"] * KB / (KB + H)
    OH = Ks["KW"] / H
    return np.log(10) * (2 * HCO3 * K2 / H + BAlk * H / (KB + H) + OH + H)


def pHfromTAHCO3(pH, TA, HCO3, totals, Ks):
    """Calculate delta-pH from pH and HCO3 for solver `pHfromTAHCO3`."""
    if use_approximate_slopes:
        return -(
            _pHfromTAHCO3_r(pH, TA, HCO3, totals, Ks)
            / _pHfromTAHCO3_s_approx(pH, TA, HCO3, totals, Ks)
        )
    else:
        return -(
            _pHfromTAHCO3_r(pH, TA, HCO3, totals, Ks)
            / _pHfromTAHCO3_s(pH, TA, HCO3, totals, Ks)
        )
