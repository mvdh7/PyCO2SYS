# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Calculate equilibrium constants from temperature, salinity and pressure."""

from . import p1atm, pcx, pressured
from autograd.numpy import full, nan, size, where
from .. import convert

__all__ = ["p1atm", "pcx", "pressured"]


def assemble(TempC, Pdbar, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF):
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
    # All constants are converted to the pH scale `pHScale` (the chosen one) in
    # units of mol/kg-sw, except KS and KF are on the Free scale, and KW is in
    # units of (mol/kg-sw)**2.
    TempK = convert.TempC2K(TempC)
    Pbar = convert.Pdbar2bar(Pdbar)
    K0 = p1atm.kCO2_W74(TempK, Sal)
    KSO4 = pressured.KSO4(TempK, Sal, Pbar, WhoseKSO4)
    KF = pressured.KF(TempK, Sal, Pbar, WhoseKF)
    # Calculate pH scale conversion factors - these are NOT pressure-corrected
    KSO40 = pressured.KSO4(TempK, Sal, 0.0, WhoseKSO4)
    KF0 = pressured.KF(TempK, Sal, 0.0, WhoseKF)
    fH = pressured.fH(TempK, Sal, WhichKs)
    SWStoTOT0 = convert.sws2tot(totals["TSO4"], KSO40, totals["TF"], KF0)
    # Calculate other dissociation constants
    KB = pressured.KB(TempK, Sal, Pbar, WhichKs, fH, SWStoTOT0)
    KW = pressured.KW(TempK, Sal, Pbar, WhichKs)
    KP1, KP2, KP3 = pressured.KP(TempK, Sal, Pbar, WhichKs, fH)
    KSi = pressured.KSi(TempK, Sal, Pbar, WhichKs, fH)
    K1, K2 = pressured.KC(TempK, Sal, Pbar, WhichKs, fH, SWStoTOT0)
    # From CO2SYS_v1_21.m: calculate KH2S and KNH3
    KH2S = pressured.KH2S(TempK, Sal, Pbar, WhichKs, SWStoTOT0)
    KNH3 = pressured.KNH3(TempK, Sal, Pbar, WhichKs, SWStoTOT0)
    # Correct pH scale conversions for pressure.
    # fH has been assumed to be independent of pressure.
    SWStoTOT = convert.sws2tot(totals["TSO4"], KSO4, totals["TF"], KF)
    FREEtoTOT = convert.free2tot(totals["TSO4"], KSO4)
    # The values KS and KF are already now pressure-corrected, so the pH scale
    # conversions are now valid at pressure.
    # Find pH scale conversion factor: this is the scale they will be put on
    pHfactor = full(size(TempC), nan)
    pHfactor = where(pHScale == 1, SWStoTOT, pHfactor)  # Total
    pHfactor = where(pHScale == 2, 1.0, pHfactor)  # Seawater (already on this)
    pHfactor = where(pHScale == 3, SWStoTOT / FREEtoTOT, pHfactor)  # Free
    pHfactor = where(pHScale == 4, fH, pHfactor)  # NBS
    # Convert from SWS pH scale to chosen scale
    K1 = K1 * pHfactor
    K2 = K2 * pHfactor
    KW = KW * pHfactor
    KB = KB * pHfactor
    KP1 = KP1 * pHfactor
    KP2 = KP2 * pHfactor
    KP3 = KP3 * pHfactor
    KSi = KSi * pHfactor
    KNH3 = KNH3 * pHfactor
    KH2S = KH2S * pHfactor
    # Return solution equilibrium constants and the fH factor as a dict
    return {
        "fH": fH,
        "K0": K0,
        "K1": K1,
        "K2": K2,
        "KW": KW,
        "KB": KB,
        "KF": KF,
        "KSO4": KSO4,
        "KP1": KP1,
        "KP2": KP2,
        "KP3": KP3,
        "KSi": KSi,
        "KNH3": KNH3,
        "KH2S": KH2S,
    }
