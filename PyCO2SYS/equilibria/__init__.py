# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate equilibrium constants from temperature, salinity and pressure."""

from autograd import numpy as np
from . import p1atm, pcx, pressured
from .. import constants, convert, gas, solubility

__all__ = ["p1atm", "pcx", "pressured"]


def prepare(TempC, Pdbar, equilibria):
    """Initialise equilibria dict if needed and convert temperature/pressure units."""
    TempK = convert.TempC2K(TempC)
    Pbar = convert.Pdbar2bar(Pdbar)
    if equilibria is None:
        equilibria = {}
    return TempK, Pbar, equilibria


def assemble(
    TempC, Pdbar, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF, WhichR, Ks=None
):
    """Evaluate all stoichiometric equilibrium constants, converted to the
    chosen pH scale, and corrected for pressure.

    Inputs must first be conditioned with inputs().

    This finds the Constants of the CO2 system in seawater or freshwater,
    corrects them for pressure, and reports them on the chosen pH scale.
    The process is as follows: the Constants (except KS, KF which stay on the
    free scale - these are only corrected for pressure) are:
          1) evaluated as they are given in the literature,
          2) converted to the SWS scale in mol/kg-SW or to the NBS scale,
          3) corrected for pressure,
          4) converted to the SWS pH scale in mol/kg-SW,
          5) converted to the chosen pH scale.

    Based on a subset of Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    TempK, Pbar, Ks = prepare(TempC, Pdbar, Ks)
    Sal = totals["Sal"]
    # Set ideal gas constant
    if "RGas" not in Ks:
        Ks["RGas"] = constants.RGasConstant(WhichR)
    RGas = Ks["RGas"]
    # Get KSO4 and KF, at pressure, and always on the Free pH scale
    if "KSO4" not in Ks:
        Ks["KSO4"] = pressured.KSO4(TempK, Sal, Pbar, RGas, WhoseKSO4)
    if "KF" not in Ks:
        Ks["KF"] = pressured.KF(TempK, Sal, Pbar, RGas, WhoseKF)
    # Depressurise KSO4 and KF for pH scale conversions
    if "KSO4_P0" not in Ks:
        Ks["KSO4_P0"] = Ks["KSO4"] / pcx.KSO4fac(TempK, Pbar, RGas)
    if "KF_P0" not in Ks:
        Ks["KF_P0"] = Ks["KF"] / pcx.KFfac(TempK, Pbar, RGas)
    # Correct pH scale conversion factors for pressure.
    # Note that fH has been assumed to be independent of pressure.
    # The values KS and KF are already now pressure-corrected, so the pH scale
    # conversions are now valid at pressure.
    # Find pH scale conversion factor: this is the scale they will be put on
    Ks = convert.get_pHfactor_from_SWS(TempK, Sal, totals, Ks, pHScale, WhichKs)
    pHfactor = Ks["pHfactor_from_SWS"]  # for convenience
    SWStoTOT_P0 = convert.pH_sws_to_total_P0(TempK, totals, Ks, WhoseKSO4, WhoseKF)
    # Borate
    if "KB" not in Ks:
        Ks["KB"] = (
            pressured.KB(TempK, Sal, Pbar, RGas, WhichKs, Ks["fH"], SWStoTOT_P0)
            * pHfactor
        )
    # Water
    if "KW" not in Ks:
        Ks["KW"] = pressured.KW(TempK, Sal, Pbar, RGas, WhichKs) * pHfactor
    # Phosphate
    if ("KP1" not in Ks) or ("KP2" not in Ks) or ("KP3" not in Ks):
        KP1, KP2, KP3 = pressured.KP(TempK, Sal, Pbar, RGas, WhichKs, Ks["fH"])
        if "KP1" not in Ks:
            Ks["KP1"] = KP1 * pHfactor
        if "KP2" not in Ks:
            Ks["KP2"] = KP2 * pHfactor
        if "KP3" not in Ks:
            Ks["KP3"] = KP3 * pHfactor
    # Silicate
    if "KSi" not in Ks:
        Ks["KSi"] = pressured.KSi(TempK, Sal, Pbar, RGas, WhichKs, Ks["fH"]) * pHfactor
    # Carbonate
    if ("K1" not in Ks) or ("K2" not in Ks):
        K1, K2 = pressured.KC(TempK, Sal, Pbar, RGas, WhichKs, Ks["fH"], SWStoTOT_P0)
        if "K1" not in Ks:
            Ks["K1"] = K1 * pHfactor
        if "K2" not in Ks:
            Ks["K2"] = K2 * pHfactor
    # Sulfide
    if "KH2S" not in Ks:
        Ks["KH2S"] = (
            pressured.KH2S(TempK, Sal, Pbar, RGas, WhichKs, SWStoTOT_P0) * pHfactor
        )
    # Ammonium
    if "KNH3" not in Ks:
        Ks["KNH3"] = (
            pressured.KNH3(TempK, Sal, Pbar, RGas, WhichKs, SWStoTOT_P0) * pHfactor
        )
    # K0 for CO2 dissolution - no pressure or pH scale corrections applied
    if "K0" not in Ks:
        Ks["K0"] = p1atm.kCO2_W74(TempK, Sal)
    if "FugFac" not in Ks:
        Ks["FugFac"] = gas.fugacityfactor(TempC, WhichKs, RGas)
    if "VPFac" not in Ks:
        Ks["VPFac"] = gas.vpfactor(TempC, Sal)
    Ks = convert.get_pHfactor_to_Free(TempK, Sal, totals, Ks, pHScale, WhichKs)
    # Aragonite and calcite solubility products
    if "KAr" not in Ks:
        Ks["KAr"] = np.where(
            (WhichKs == 6) | (WhichKs == 7),  # GEOSECS values
            solubility.k_aragonite_GEOSECS(TempK, Sal, Pbar, RGas),
            solubility.k_aragonite_M83(TempK, Sal, Pbar, RGas),
        )
    if "KCa" not in Ks:
        Ks["KCa"] = np.where(
            (WhichKs == 6) | (WhichKs == 7),  # GEOSECS values
            solubility.k_calcite_I75(TempK, Sal, Pbar, RGas),
            solubility.k_calcite_M83(TempK, Sal, Pbar, RGas),
        )
    # Extra alkalinity components
    if "k_alpha" not in Ks:
        Ks["k_alpha"] = 1e-7
    if "k_beta" not in Ks:
        Ks["k_beta"] = 1e-7
    return Ks
