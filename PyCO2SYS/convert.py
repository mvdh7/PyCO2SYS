# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Convert units and calculate conversion factors."""

import copy
from autograd import numpy as np
from . import constants
from .equilibria import pressured


def TempC2K(TempC):
    """Convert temperature from degC to K."""
    return TempC + constants.Tzero


def TempK2C(TempK):
    """Convert temperature from K to degC."""
    return TempK - constants.Tzero


def Pdbar2bar(Pdbar):
    """Convert pressure from dbar to bar."""
    return Pdbar / 10.0


def Pbar2dbar(Pbar):
    """Convert pressure from bar to dbar."""
    return Pbar * 10.0


def free2tot(totals, k_constants):
    """Free to Total pH scale conversion factor."""
    return 1.0 + totals["TSO4"] / k_constants["KSO4"]


def free2sws(totals, k_constants):
    """Free to Seawater pH scale conversion factor."""
    return 1.0 + totals["TSO4"] / k_constants["KSO4"] + totals["TF"] / k_constants["KF"]


def sws2free(totals, k_constants):
    """Seawater to Free pH scale conversion factor."""
    return 1.0 / free2sws(totals, k_constants)


def sws2tot(totals, k_constants):
    """Seawater to Total pH scale conversion factor."""
    return sws2free(totals, k_constants) * free2tot(totals, k_constants)


def tot2free(totals, k_constants):
    """Total to Free pH scale conversion factor."""
    return 1.0 / free2tot(totals, k_constants)


def tot2sws(totals, k_constants):
    """Total to Seawater pH scale conversion factor."""
    return 1.0 / sws2tot(totals, k_constants)


def sws2nbs(totals, k_constants):
    """Seawater to NBS pH scale conversion factor."""
    return k_constants["fH"]


def nbs2sws(totals, k_constants):
    """NBS to Seawater pH scale conversion factor."""
    return 1.0 / sws2nbs(totals, k_constants)


def tot2nbs(totals, k_constants):
    """Total to NBS pH scale conversion factor."""
    return tot2sws(totals, k_constants) * sws2nbs(totals, k_constants)


def nbs2tot(totals, k_constants):
    """NBS to Total pH scale conversion factor."""
    return 1.0 / tot2nbs(totals, k_constants)


def free2nbs(totals, k_constants):
    """Free to NBS pH scale conversion factor."""
    return free2sws(totals, k_constants) * sws2nbs(totals, k_constants)


def nbs2free(totals, k_constants):
    """NBS to Free pH scale conversion factor."""
    return 1.0 / free2nbs(totals, k_constants)


def fH_PTBO87(TempK, Sal):
    """fH following PTBO87."""
    # === CO2SYS.m comments: =======
    # Peng et al, Tellus 39B:439-458, 1987:
    # They reference the GEOSECS report, but round the value
    # given there off so that it is about .008 (1#) lower. It
    # doesn't agree with the check value they give on p. 456.
    return 1.29 - 0.00204 * TempK + (0.00046 - 0.00000148 * TempK) * Sal ** 2


def fH_TWB82(TempK, Sal):
    """fH following TWB82."""
    # === CO2SYS.m comments: =======
    # Takahashi et al, Chapter 3 in GEOSECS Pacific Expedition,
    # v. 3, 1982 (p. 80).
    return 1.2948 - 0.002036 * TempK + (0.0004607 - 0.000001475 * TempK) * Sal ** 2


def pH2allscales(pH, pHScale, totals, k_constants):
    """Calculate pH on all scales.

    This takes the pH on the given pHScale and finds the pH on all scales.

    Based on FindpHOnAllScales, version 01.02, 01-08-97, by Ernie Lewis.
    """
    FREEtoTOT = free2tot(totals, k_constants)
    SWStoTOT = sws2tot(totals, k_constants)
    factor = np.full(np.shape(pH), np.nan)
    factor = np.where(pHScale == 1, 0.0, factor)  # Total
    factor = np.where(pHScale == 2, np.log10(SWStoTOT), factor)  # Seawater
    factor = np.where(pHScale == 3, np.log10(FREEtoTOT), factor)  # Free
    factor = np.where(
        pHScale == 4, np.log10(SWStoTOT / k_constants["fH"]), factor
    )  # NBS
    pHtot = pH - factor  # pH comes into this function on the given scale
    pHNBS = pHtot + np.log10(SWStoTOT / k_constants["fH"])
    pHfree = pHtot + np.log10(FREEtoTOT)
    pHsws = pHtot + np.log10(SWStoTOT)
    return pHtot, pHsws, pHfree, pHNBS


def sws2tot_P0(TempK, totals, k_constants, WhoseKSO4, WhoseKF):
    """Determine SWS to Total pH scale correction factor at zero pressure."""
    k_constants_P0 = copy.deepcopy(k_constants)
    k_constants_P0["KSO4"] = pressured.KSO4(TempK, totals["Sal"], 0.0, 1.0, WhoseKSO4)
    k_constants_P0["KF"] = pressured.KF(TempK, totals["Sal"], 0.0, 1.0, WhoseKF)
    return sws2tot(totals, k_constants_P0)


def get_pHfactor_from_SWS(TempK, Sal, totals, k_constants, pHScale, WhichKs):
    """Determine pH scale conversion factors to go from SWS to input pHScale(s).
    The raw K values (not pK) should be multiplied by these to make the conversion.
    """
    if "fH" not in k_constants:
        k_constants["fH"] = pressured.fH(TempK, Sal, WhichKs)
    pHfactor = np.full(np.shape(pHScale), np.nan)
    pHfactor = np.where(pHScale == 1, sws2tot(totals, k_constants), pHfactor)  # Total
    pHfactor = np.where(pHScale == 2, 1.0, pHfactor)  # Seawater (SWS)
    pHfactor = np.where(pHScale == 3, sws2free(totals, k_constants), pHfactor)  # Free
    pHfactor = np.where(pHScale == 4, sws2nbs(totals, k_constants), pHfactor)  # NBS
    k_constants["pHfactor_from_SWS"] = pHfactor
    return k_constants


def get_pHfactor_to_Free(TempK, Sal, totals, k_constants, pHScale, WhichKs):
    """Determine pH scale conversion factors to go from input pHScale(s) to Free.
    The raw K values (not pK) should be multiplied by these to make the conversion.
    """
    if "fH" not in k_constants:
        k_constants["fH"] = pressured.fH(TempK, Sal, WhichKs)
    pHfactor = np.full(np.shape(pHScale), np.nan)
    pHfactor = np.where(pHScale == 1, tot2free(totals, k_constants), pHfactor)  # Total
    pHfactor = np.where(
        pHScale == 2, sws2free(totals, k_constants), pHfactor
    )  # Seawater (SWS)
    pHfactor = np.where(pHScale == 3, 1.0, pHfactor)  # Free
    pHfactor = np.where(pHScale == 4, nbs2free(totals, k_constants), pHfactor)  # NBS
    k_constants["pHfactor_to_Free"] = pHfactor
    return k_constants


def options_old2new(KSO4CONSTANTS):
    """Convert traditional CO2SYS `KSO4CONSTANTS` input to new separated format."""
    if np.shape(KSO4CONSTANTS) == ():
        KSO4CONSTANTS = np.array([KSO4CONSTANTS])
    only2KSO4 = {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
    }
    only2BORON = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
    }
    KSO4CONSTANT = np.array([only2KSO4[K] for K in KSO4CONSTANTS.ravel()])
    BORON = np.array([only2BORON[K] for K in KSO4CONSTANTS.ravel()])
    return KSO4CONSTANT, BORON


def _flattenfirst(args, dtype):
    # Determine and check lengths of input vectors
    arglengths = np.array([np.size(arg) for arg in args])
    assert (
        np.size(np.unique(arglengths[arglengths != 1])) <= 1
    ), "Inputs must all be the same length as each other or of length 1."
    # Make vectors of all inputs
    npts = np.max(arglengths)
    return (
        [
            np.full(npts, arg, dtype=dtype)
            if np.size(arg) == 1
            else arg.ravel().astype(dtype)
            for arg in args
        ],
        npts,
    )


def _flattenafter(args, npts, dtype):
    # Determine and check lengths of input vectors
    arglengths = np.array([np.size(arg) for arg in args])
    assert np.all(
        np.isin(arglengths, [1, npts])
    ), "Inputs must all be the same length as each other or of length 1."
    # Make vectors of all inputs
    return [
        np.full(npts, arg, dtype=dtype)
        if np.size(arg) == 1
        else arg.ravel().astype(dtype)
        for arg in args
    ]


def _flattentext(args, npts):
    # Determine and check lengths of input vectors
    arglengths = np.array([np.size(arg) for arg in args])
    assert np.all(
        np.isin(arglengths, [1, npts])
    ), "Inputs must all be the same length as each other or of length 1."
    # Make vectors of all inputs
    return [np.full(npts, arg) if np.size(arg) == 1 else arg.ravel() for arg in args]


def options_new2old(KSO4CONSTANT, BORON):
    """Convert separated `KSO4CONSTANT` and `BORON` options into traditional CO2SYS
    `KSO4CONSTANTS` input.
    """
    pair2one = {
        (1, 1): 1,
        (2, 1): 2,
        (1, 2): 3,
        (2, 2): 4,
    }
    KSO4CONSTANT, BORON = _flattenfirst((KSO4CONSTANT, BORON), int)[0]
    pairs = zip(KSO4CONSTANT, BORON)
    return np.array([pair2one[pair] for pair in pairs])
