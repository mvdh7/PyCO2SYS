# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Convert units and calculate conversion factors."""

import copy
from autograd import numpy as np
from .constants import Tzero
from .equilibria import pressured

def TempC2K(TempC):
    """Convert temperature from degC to K."""
    return TempC + Tzero


def TempK2C(TempK):
    """Convert temperature from K to degC."""
    return TempK - Tzero


def Pdbar2bar(Pdbar):
    """Convert pressure from dbar to bar."""
    return Pdbar / 10.0


def Pbar2dbar(Pbar):
    """Convert pressure from bar to dbar."""
    return Pbar * 10.0


def free2tot(totals, equilibria):
    """Free to Total pH scale conversion factor."""
    return 1.0 + totals["TSO4"] / equilibria["KSO4"]


def free2sws(totals, equilibria):
    """Free to Seawater pH scale conversion factor."""
    return 1.0 + totals["TSO4"] / equilibria["KSO4"] + totals["TF"] / equilibria["KF"]


def sws2free(totals, equilibria):
    """Seawater to Free pH scale conversion factor."""
    return 1.0 / free2sws(totals, equilibria)


def sws2tot(totals, equilibria):
    """Seawater to Total pH scale conversion factor."""
    return sws2free(totals, equilibria) * free2tot(totals, equilibria)


def tot2free(totals, equilibria):
    """Total to Free pH scale conversion factor."""
    return 1.0 / free2tot(totals, equilibria)


def tot2sws(totals, equilibria):
    """Total to Seawater pH scale conversion factor."""
    return 1.0 / sws2tot(totals, equilibria)


def sws2nbs(totals, equilibria):
    """Seawater to NBS pH scale conversion factor."""
    return equilibria["fH"]


def nbs2sws(totals, equilibria):
    """NBS to Seawater pH scale conversion factor."""
    return 1.0 / sws2nbs(totals, equilibria)


def tot2nbs(totals, equilibria):
    """Total to NBS pH scale conversion factor."""
    return tot2sws(totals, equilibria) * sws2nbs(totals, equilibria)


def nbs2tot(totals, equilibria):
    """NBS to Total pH scale conversion factor."""
    return 1.0 / tot2nbs(totals, equilibria)


def free2nbs(totals, equilibria):
    """Free to NBS pH scale conversion factor."""
    return free2sws(totals, equilibria) * sws2nbs(totals, equilibria)


def nbs2free(totals, equilibria):
    """NBS to Free pH scale conversion factor."""
    return 1.0 / free2nbs(totals, equilibria)


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


def pH2allscales(pH, pHScale, totals, equilibria):
    """Calculate pH on all scales.

    This takes the pH on the given pHScale and finds the pH on all scales.

    Based on FindpHOnAllScales, version 01.02, 01-08-97, by Ernie Lewis.
    """
    FREEtoTOT = free2tot(totals, equilibria)
    SWStoTOT = sws2tot(totals, equilibria)
    factor = np.full(np.size(pH), np.nan)
    factor = np.where(pHScale == 1, 0.0, factor)  # Total
    factor = np.where(pHScale == 2, np.log10(SWStoTOT), factor)  # Seawater
    factor = np.where(pHScale == 3, np.log10(FREEtoTOT), factor)  # Free
    factor = np.where(
        pHScale == 4, np.log10(SWStoTOT / equilibria["fH"]), factor
    )  # NBS
    pHtot = pH - factor  # pH comes into this function on the given scale
    pHNBS = pHtot + np.log10(SWStoTOT / equilibria["fH"])
    pHfree = pHtot + np.log10(FREEtoTOT)
    pHsws = pHtot + np.log10(SWStoTOT)
    return pHtot, pHsws, pHfree, pHNBS


def sws2tot_P0(TempK, totals, equilibria, WhoseKSO4, WhoseKF):
    """Determine SWS to Total pH scale correction factor at zero pressure."""
    equilibria_P0 = copy.deepcopy(equilibria)
    equilibria_P0["KSO4"] = pressured.KSO4(TempK, totals["Sal"], 0.0, 1.0, WhoseKSO4)
    equilibria_P0["KF"] = pressured.KF(TempK, totals["Sal"], 0.0, 1.0, WhoseKF)
    SWStoTOT_P0 = sws2tot(totals, equilibria_P0)
    return SWStoTOT_P0


def get_pHfactor_from_SWS(TempK, Sal, totals, equilibria, pHScale, WhichKs):
    """Determine pH scale conversion factors to go from SWS to input pHScale(s).
    The raw K values (not pK) should be multiplied by these to make the conversion.
    """
    if "fH" not in equilibria:
        equilibria["fH"] = pressured.fH(TempK, Sal, WhichKs)
    SWStoTOT = sws2tot(totals, equilibria)
    SWStoFREE = sws2free(totals, equilibria)
    SWStoNBS = sws2nbs(totals, equilibria)
    pHfactor = np.full(np.size(pHScale), np.nan)
    pHfactor = np.where(pHScale == 1, SWStoTOT, pHfactor)  # Total
    pHfactor = np.where(pHScale == 2, 1.0, pHfactor)  # Seawater (SWS)
    pHfactor = np.where(pHScale == 3, SWStoFREE, pHfactor)  # Free
    pHfactor = np.where(pHScale == 4, SWStoNBS, pHfactor)  # NBS
    equilibria["pHfactor_from_SWS"] = pHfactor
    return equilibria


def get_pHfactor_to_Free(TempK, Sal, totals, equilibria, pHScale, WhichKs):
    """Determine pH scale conversion factors to go from input pHScale(s) to Free.
    The raw K values (not pK) should be multiplied by these to make the conversion.
    """
    if "fH" not in equilibria:
        equilibria["fH"] = pressured.fH(TempK, Sal, WhichKs)
    TOTtoFREE = tot2free(totals, equilibria)
    SWStoFREE = sws2free(totals, equilibria)
    NBStoFREE = nbs2free(totals, equilibria)
    pHfactor = np.full(np.size(pHScale), np.nan)
    pHfactor = np.where(pHScale == 1, TOTtoFREE, pHfactor)  # Total
    pHfactor = np.where(pHScale == 2, SWStoFREE, pHfactor)  # Seawater (SWS)
    pHfactor = np.where(pHScale == 3, 1.0, pHfactor)  # Free
    pHfactor = np.where(pHScale == 4, NBStoFREE, pHfactor)  # NBS
    equilibria["pHfactor_to_Free"] = pHfactor
    return equilibria


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
