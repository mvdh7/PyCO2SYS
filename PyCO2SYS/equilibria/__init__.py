# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Calculate equilibrium constants from temperature, salinity and pressure."""

from autograd.numpy import full, nan, size, where
from . import p1atm, pcx, pressured
from .. import constants, convert, gas

__all__ = ["p1atm", "pcx", "pressured"]


def prepare(TempC, Pdbar, Ks):
    """Initialise Ks dict if needed and convert temperature/pressure units."""
    # Extract and convert
    TempK = convert.TempC2K(TempC)
    Pbar = convert.Pdbar2bar(Pdbar)
    # Initialise dict
    if Ks is None:
        Ks = {}
    return TempK, Pbar, Ks


def sws2tot_P0(TempK, totals, WhoseKSO4, WhoseKF):
    """Determine SWS to Total pH scale correction factor at zero pressure."""
    KSO4_P0 = pressured.KSO4(TempK, totals["Sal"], 0.0, 1.0, WhoseKSO4)
    KF_P0 = pressured.KF(TempK, totals["Sal"], 0.0, 1.0, WhoseKF)
    SWStoTOT_P0 = convert.sws2tot(totals["TSO4"], KSO4_P0, totals["TF"], KF_P0)
    return SWStoTOT_P0


def get_pHfactor(TempK, Sal, totals, Ks, pHScale, WhichKs):
    """Determine pH scale conversion factors to go from input pHScale to SWS."""
    if "fH" not in Ks:
        Ks["fH"] = pressured.fH(TempK, Sal, WhichKs)
    SWStoTOT = convert.sws2tot(totals["TSO4"], Ks["KSO4"], totals["TF"], Ks["KF"])
    FREEtoTOT = convert.free2tot(totals["TSO4"], Ks["KSO4"])
    pHfactor = full(size(pHScale), nan)
    pHfactor = where(pHScale == 1, SWStoTOT, pHfactor)  # Total
    pHfactor = where(pHScale == 2, 1.0, pHfactor)  # Seawater (already on this)
    pHfactor = where(pHScale == 3, SWStoTOT / FREEtoTOT, pHfactor)  # Free
    pHfactor = where(pHScale == 4, Ks["fH"], pHfactor)  # NBS
    return pHfactor, Ks


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
    # Correct pH scale conversion factors for pressure.
    # Note that fH has been assumed to be independent of pressure.
    # The values KS and KF are already now pressure-corrected, so the pH scale
    # conversions are now valid at pressure.
    # Find pH scale conversion factor: this is the scale they will be put on
    pHfactor, Ks = get_pHfactor(TempK, Sal, totals, Ks, pHScale, WhichKs)
    SWStoTOT_P0 = sws2tot_P0(TempK, totals, WhoseKSO4, WhoseKF)
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
    return Ks
