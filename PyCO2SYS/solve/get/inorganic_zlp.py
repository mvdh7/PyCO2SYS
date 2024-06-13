# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate one new carbonate system variable from various input pairs considering
only inorganic solutes (i.e., no DOM) and with a variable ZLP.
"""

from jax import numpy as np, lax
from ... import salts
from .. import delta, dom, initialise, residual, speciate
from .inorganic import (
    dic_from_pH_fCO2,
    dic_from_pH_carbonate,
    dic_from_pH_bicarbonate,
    dic_from_fCO2_carbonate,
    dic_from_fCO2_bicarbonate,
    dic_from_carbonate_bicarbonate,
    pH_from_dic_fCO2,
    pH_from_dic_carbonate,
    pH_from_dic_bicarbonate,
    pH_from_fCO2_carbonate,
    pH_from_fCO2_bicarbonate,
    pH_from_carbonate_bicarbonate,
    fCO2_from_carbonate_bicarbonate,
    fCO2_from_dic_pH,
    fCO2_from_pH_carbonate,
    fCO2_from_pH_bicarbonate,
    _carbonate_from_dic_H,
    carbonate_from_dic_pH,
    carbonate_from_pH_fCO2,
    carbonate_from_pH_bicarbonate,
    carbonate_from_fCO2_bicarbonate,
    _bicarbonate_from_dic_H,
    bicarbonate_from_dic_pH,
    bicarbonate_from_pH_fCO2,
    bicarbonate_from_pH_carbonate,
    bicarbonate_from_fCO2_carbonate,
    _CO2_from_dic_H,
)


def alkalinity_from_dic_pH(dic, pH, totals, k_constants, pzlp):
    """Calculate total alkalinity from dissolved inorganic carbon and pH."""
    sw = speciate.inorganic_zlp(dic, pH, totals, k_constants, pzlp)
    return sw["alkalinity"] * 1e6


def alkalinity_from_pH_fCO2(pH, fCO2, totals, k_constants, pzlp):
    """Calculate total alkalinity from dissolved inorganic carbon and CO2 fugacity."""
    dic = dic_from_pH_fCO2(pH, fCO2, totals, k_constants)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants, pzlp)


def alkalinity_from_pH_carbonate(pH, carbonate, totals, k_constants, pzlp):
    """Calculate total alkalinity from dissolved inorganic carbon and carbonate ion."""
    dic = dic_from_pH_carbonate(pH, carbonate, totals, k_constants)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants, pzlp)


def alkalinity_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants, pzlp):
    """Calculate total alkalinity from dissolved inorganic carbon and bicarbonate ion."""
    dic = dic_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants, pzlp)


def alkalinity_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants, pzlp):
    """Total alkalinity from CO2 fugacity and carbonate ion."""
    pH = pH_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants)
    return alkalinity_from_pH_fCO2(pH, fCO2, totals, k_constants, pzlp)


def alkalinity_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants, pzlp):
    """Total alkalinity from CO2 fugacity and bicarbonate ion."""
    carbonate = carbonate_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants)
    return alkalinity_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants, pzlp)


def alkalinity_from_carbonate_bicarbonate(
    carbonate, bicarbonate, totals, k_constants, pzlp
):
    """Total alkalinity from carbonate ion and carbonate ion."""
    pH = pH_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants)
    return alkalinity_from_pH_carbonate(pH, carbonate, totals, k_constants, pzlp)


def dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants, pzlp):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.
    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.
    """
    alkalinity_with_zero_dic = alkalinity_from_dic_pH(
        0.0, pH, totals, k_constants, pzlp
    )
    F = alkalinity_with_zero_dic > alkalinity
    if np.any(F):
        print("Some input pH values are impossibly high given the input alkalinity;")
        print("returning np.nan rather than negative DIC values.")
    alkalinity_carbonate = np.where(F, np.nan, alkalinity - alkalinity_with_zero_dic)
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    H = 10.0**-pH
    dic = alkalinity_carbonate * (H**2 + K1 * H + K1 * K2) / (K1 * (H + 2 * K2))
    return dic


def pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants, pzlp):
    """Calculate pH from total alkalinity and DIC, without DOM, with variable ZLP."""

    def cond(targets):
        pH = targets
        residuals = np.array(
            [
                residual.pH_from_alkalinity_dic_zlp(
                    pH, alkalinity, dic, totals, k_constants, pzlp
                )
            ]
        )
        return np.any(np.abs(residuals) > 1e-9)

    def body(targets):
        pH = targets
        deltas = delta.pH_from_alkalinity_dic_zlp(
            pH, alkalinity, dic, totals, k_constants, pzlp
        )
        deltas = np.where(deltas > 1, 1.0, deltas)
        deltas = np.where(deltas < -1, -1.0, deltas)
        return targets + deltas

    # First guess and solve
    pH_initial = 7.0
    targets = pH_initial
    targets = lax.while_loop(cond, body, targets)
    return targets


def fCO2_from_alkalinity_dic(alkalinity, dic, totals, k_constants, pzlp):
    """Calculate CO2 fugacity from total alkalinity and dissolved inorganic carbon."""
    pH = pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants, pzlp)
    return fCO2_from_dic_pH(dic, pH, totals, k_constants)


def fCO2_from_alkalinity_pH(alkalinity, pH, totals, k_constants, pzlp):
    """Calculate CO2 fugacity from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants, pzlp)
    return fCO2_from_dic_pH(dic, pH, totals, k_constants)


def carbonate_from_alkalinity_dic(alkalinity, dic, totals, k_constants, pzlp):
    """Calculate carbonate ion from total alkalinity and dissolved inorganic carbon."""
    pH = pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants, pzlp)
    return carbonate_from_dic_pH(dic, pH, totals, k_constants)


def carbonate_from_alkalinity_pH(alkalinity, pH, totals, k_constants, pzlp):
    """Calculate carbonate ion from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants, pzlp)
    return carbonate_from_dic_pH(dic, pH, totals, k_constants)


def bicarbonate_from_alkalinity_pH(alkalinity, pH, totals, k_constants, pzlp):
    """Calculate carbonate ion from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants, pzlp)
    return bicarbonate_from_dic_pH(dic, pH, totals, k_constants)
