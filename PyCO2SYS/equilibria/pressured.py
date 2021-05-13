# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Correct equilibrium constants for pressure."""

from autograd import numpy as np
from . import p1atm, pcx
from .. import convert


def KSO4(TempK, Sal, Pbar, RGas, WhoseKSO4):
    """Calculate bisulfate ion dissociation constant for the given options."""
    assert np.all(
        np.isin(WhoseKSO4, [1, 2, 3])
    ), "Valid WhoseKSO4 values are: 1, 2 or 3."
    # Evaluate at atmospheric pressure
    KSO4 = np.full(np.shape(TempK), np.nan)
    KSO4 = np.where(WhoseKSO4 == 1, p1atm.kHSO4_FREE_D90a(TempK, Sal), KSO4)
    KSO4 = np.where(WhoseKSO4 == 2, p1atm.kHSO4_FREE_KRCB77(TempK, Sal), KSO4)
    KSO4 = np.where(WhoseKSO4 == 3, p1atm.kHSO4_FREE_WM13(TempK, Sal), KSO4)
    # Now correct for seawater pressure
    KSO4 = KSO4 * pcx.KSO4fac(TempK, Pbar, RGas)
    return KSO4


def KF(TempK, Sal, Pbar, RGas, WhoseKF):
    """Calculate HF dissociation constant for the given options."""
    assert np.all(np.isin(WhoseKF, [1, 2])), "Valid WhoseKF values are: 1 or 2."
    # Evaluate at atmospheric pressure
    KF = np.full(np.shape(TempK), np.nan)
    KF = np.where(WhoseKF == 1, p1atm.kHF_FREE_DR79(TempK, Sal), KF)
    KF = np.where(WhoseKF == 2, p1atm.kHF_FREE_PF87(TempK, Sal), KF)
    # Now correct for seawater pressure
    KF = KF * pcx.KFfac(TempK, Pbar, RGas)
    return KF


def fH(TempK, Sal, WhichKs):
    """Calculate NBS to Seawater pH scale conversion factor for the given options."""
    fH = np.where(WhichKs == 8, 1.0, np.nan)
    fH = np.where(WhichKs == 7, convert.fH_PTBO87(TempK, Sal), fH)
    # Use GEOSECS's value for all other cases
    fH = np.where((WhichKs != 7) & (WhichKs != 8), convert.fH_TWB82(TempK, Sal), fH)
    return fH


def KB(TempK, Sal, Pbar, RGas, WhichKs, fH, SWStoTOT0):
    """Calculate boric acid dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KB = np.full(np.shape(TempK), np.nan)
    KB = np.where(WhichKs == 8, 0.0, KB)  # pure water case
    KB = np.where(
        (WhichKs == 6) | (WhichKs == 7), p1atm.kBOH3_NBS_LTB69(TempK, Sal) / fH, KB
    )  # convert NBS to SWS
    KB = np.where(
        (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8),
        p1atm.kBOH3_TOT_D90b(TempK, Sal) / SWStoTOT0,
        KB,
    )  # convert TOT to SWS
    # Now correct for seawater pressure
    KB = KB * pcx.KBfac(TempK, Pbar, RGas, WhichKs)
    return KB


def KW(TempK, Sal, Pbar, RGas, WhichKs):
    """Calculate water dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KW = np.full(np.shape(TempK), np.nan)
    KW = np.where(WhichKs == 6, 0.0, KW)  # GEOSECS doesn't include OH effects
    KW = np.where(WhichKs == 7, p1atm.kH2O_SWS_M79(TempK, Sal), KW)
    KW = np.where(WhichKs == 8, p1atm.kH2O_SWS_HO58_M79(TempK, Sal), KW)
    KW = np.where(
        (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8),
        p1atm.kH2O_SWS_M95(TempK, Sal),
        KW,
    )
    # Now correct for seawater pressure
    KW = KW * pcx.KWfac(TempK, Pbar, RGas, WhichKs)
    return KW


def KP(TempK, Sal, Pbar, RGas, WhichKs, fH):
    """Calculate phosphoric acid dissociation constants for the given options."""
    # Evaluate at atmospheric pressure
    KP1 = np.full(np.shape(TempK), np.nan)
    KP2 = np.full(np.shape(TempK), np.nan)
    KP3 = np.full(np.shape(TempK), np.nan)
    F = WhichKs == 7
    KP1_KP67, KP2_KP67, KP3_KP67 = p1atm.kH3PO4_NBS_KP67(TempK, Sal)
    KP1 = np.where(F, KP1_KP67, KP1)  # already on SWS!
    KP2 = np.where(F, KP2_KP67 / fH, KP2)  # convert NBS to SWS
    KP3 = np.where(F, KP3_KP67 / fH, KP3)  # convert NBS to SWS
    F = (WhichKs == 6) | (WhichKs == 8)
    # Note: neither the GEOSECS choice nor the freshwater choice include
    # contributions from phosphate or silicate.
    KP1 = np.where(F, 0.0, KP1)
    KP2 = np.where(F, 0.0, KP2)
    KP3 = np.where(F, 0.0, KP3)
    F = (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8)
    KP1_YM95, KP2_YM95, KP3_YM95 = p1atm.kH3PO4_SWS_YM95(TempK, Sal)
    KP1 = np.where(F, KP1_YM95, KP1)
    KP2 = np.where(F, KP2_YM95, KP2)
    KP3 = np.where(F, KP3_YM95, KP3)
    # Now correct for seawater pressure
    # === CO2SYS.m comments: =======
    # These corrections don't matter for the GEOSECS choice (WhichKs = 6) and
    # the freshwater choice (WhichKs = 8). For the Peng choice I assume that
    # they are the same as for the other choices (WhichKs = 1 to 5).
    # The corrections for KP1, KP2, and KP3 are from Millero, 1995, which are
    # the same as Millero, 1983.
    KP1 = KP1 * pcx.KP1fac(TempK, Pbar, RGas)
    KP2 = KP2 * pcx.KP2fac(TempK, Pbar, RGas)
    KP3 = KP3 * pcx.KP3fac(TempK, Pbar, RGas)
    return KP1, KP2, KP3


def KSi(TempK, Sal, Pbar, RGas, WhichKs, fH):
    """Calculate silicate dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KSi = np.full(np.shape(TempK), np.nan)
    KSi = np.where(
        WhichKs == 7, p1atm.kSi_NBS_SMB64(TempK, Sal) / fH, KSi
    )  # convert NBS to SWS
    # Note: neither the GEOSECS choice nor the freshwater choice include
    # contributions from phosphate or silicate.
    KSi = np.where((WhichKs == 6) | (WhichKs == 8), 0.0, KSi)
    KSi = np.where(
        (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8),
        p1atm.kSi_SWS_YM95(TempK, Sal),
        KSi,
    )
    # Now correct for seawater pressure
    KSi = KSi * pcx.KSifac(TempK, Pbar, RGas)
    return KSi


def KH2S(TempK, Sal, Pbar, RGas, WhichKs, SWStoTOT0):
    """Calculate hydrogen disulfide dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KH2S = np.where(
        (WhichKs == 6) | (WhichKs == 7) | (WhichKs == 8),
        0.0,
        p1atm.kH2S_TOT_YM95(TempK, Sal) / SWStoTOT0,
    )  # convert TOT to SWS
    # Now correct for seawater pressure
    KH2S = KH2S * pcx.KH2Sfac(TempK, Pbar, RGas)
    return KH2S


def KNH3(TempK, Sal, Pbar, RGas, WhichKs, SWStoTOT0):
    """Calculate ammonium dissociation constant for the given options."""
    # Evaluate at atmospheric pressure
    KNH3 = np.where(
        (WhichKs == 6) | (WhichKs == 7) | (WhichKs == 8),
        0.0,
        p1atm.kNH3_TOT_CW95(TempK, Sal) / SWStoTOT0,
    )  # convert TOT to SWS
    # Now correct for seawater pressure
    KNH3 = KNH3 * pcx.KNH3fac(TempK, Pbar, RGas)
    return KNH3


def _getKC(F, Kfunc, pHcx, K1, K2, ts):
    """Convenience function for getting and setting K1 and K2 values."""
    K1_F, K2_F = Kfunc(*ts)
    K1 = np.where(F, K1_F / pHcx, K1)
    K2 = np.where(F, K2_F / pHcx, K2)
    return K1, K2


def KC(TempK, Sal, Pbar, RGas, WhichKs, fH, SWStoTOT0):
    """Calculate carbonic acid dissociation constants for the given options."""
    # Evaluate at atmospheric pressure
    K1 = np.full(np.shape(TempK), np.nan)
    K2 = np.full(np.shape(TempK), np.nan)
    ts = (TempK, Sal)  # for convenience
    K1, K2 = _getKC(WhichKs == 1, p1atm.kH2CO3_TOT_RRV93, SWStoTOT0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 2, p1atm.kH2CO3_SWS_GP89, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 3, p1atm.kH2CO3_SWS_H73_DM87, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 4, p1atm.kH2CO3_SWS_MCHP73_DM87, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 5, p1atm.kH2CO3_SWS_HM_DM87, 1.0, K1, K2, ts)
    K1, K2 = _getKC(
        (WhichKs == 6) | (WhichKs == 7), p1atm.kH2CO3_NBS_MCHP73, fH, K1, K2, ts
    )
    K1, K2 = _getKC(WhichKs == 8, p1atm.kH2CO3_SWS_M79, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 9, p1atm.kH2CO3_NBS_CW98, fH, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 10, p1atm.kH2CO3_TOT_LDK00, SWStoTOT0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 11, p1atm.kH2CO3_SWS_MM02, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 12, p1atm.kH2CO3_SWS_MPL02, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 13, p1atm.kH2CO3_SWS_MGH06, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 14, p1atm.kH2CO3_SWS_M10, 1.0, K1, K2, ts)
    K1, K2 = _getKC(WhichKs == 15, p1atm.kH2CO3_SWS_WMW14, 1.0, K1, K2, ts)
    # Added v1.4.1:
    K1, K2 = _getKC(WhichKs == 16, p1atm.kH2CO3_TOT_SLH20, SWStoTOT0, K1, K2, ts)
    # Added v1.7.0:
    K1, K2 = _getKC(WhichKs == 17, p1atm.kH2CO3_TOT_SB21, SWStoTOT0, K1, K2, ts)
    # Now correct for seawater pressure
    K1 = K1 * pcx.K1fac(TempK, Pbar, RGas, WhichKs)
    K2 = K2 * pcx.K2fac(TempK, Pbar, RGas, WhichKs)
    return K1, K2


# Original notes from CO2SYS-MATLAB regarding pressure corrections:
# ****************************************************************************
# Correct dissociation constants for pressure
# Currently: For WhichKs# = 1 to 7, all Ks (except KF and KS, which are on
#       the free scale) are on the SWS scale.
#       For WhichKs# = 6, KW set to 0, KP1, KP2, KP3, KSi don't matter.
#       For WhichKs# = 8, K1, K2, and KW are on the "pH" pH scale
#       (the pH scales are the same in this case); the other Ks don't
#       matter.
#
# No salinity dependence is given for the pressure coefficients here.
# It is assumed that the salinity is at or very near Sali = 35.
# These are valid for the SWS pH scale, but the difference between this and
# the total only yields a difference of .004 pH units at 1000 bars, much
# less than the uncertainties in the values.
# ****************************************************************************
# The sources used are:
# Millero, 1995:
#       Millero, F. J., Thermodynamics of the carbon dioxide system in the
#       oceans, Geochemica et Cosmochemica Acta 59:661-677, 1995.
#       See table 9 and eqs. 90-92, p. 675.
#       TYPO: a factor of 10^3 was left out of the definition of Kappa
#       TYPO: the value of R given is incorrect with the wrong units
#       TYPO: the values of the a's for H2S and H2O are from the 1983
#                values for fresh water
#       TYPO: the value of a1 for B(OH)3 should be +.1622
#        Table 9 on p. 675 has no values for Si.
#       There are a variety of other typos in Table 9 on p. 675.
#       There are other typos in the paper, and most of the check values
#       given don't check.
# Millero, 1992:
#       Millero, Frank J., and Sohn, Mary L., Chemical Oceanography,
#       CRC Press, 1992. See chapter 6.
#       TYPO: this chapter has numerous typos (eqs. 36, 52, 56, 65, 72,
#               79, and 96 have typos).
# Millero, 1983:
#       Millero, Frank J., Influence of pressure on chemical processes in
#       the sea. Chapter 43 in Chemical Oceanography, eds. Riley, J. P. and
#       Chester, R., Academic Press, 1983.
#       TYPO: p. 51, p1atm. 94: the value -26.69 should be -25.59
#       TYPO: p. 51, p1atm. 95: the term .1700t should be .0800t
#       these two are necessary to match the values given in Table 43.24
# Millero, 1979:
#       Millero, F. J., The thermodynamics of the carbon dioxide system
#       in seawater, Geochemica et Cosmochemica Acta 43:1651-1661, 1979.
#       See table 5 and eqs. 7, 7a, 7b on pp. 1656-1657.
# Takahashi et al, in GEOSECS Pacific Expedition, v. 3, 1982.
#       TYPO: the pressure dependence of K2 should have a 16.4, not 26.4
#       This matches the GEOSECS results and is in Edmond and Gieskes.
# Culberson, C. H. and Pytkowicz, R. M., Effect of pressure on carbonic acid,
#       boric acid, and the pH of seawater, Limnology and Oceanography
#       13:403-417, 1968.
# Edmond, John M. and Gieskes, J. M. T. M., The calculation of the degree of
#       seawater with respect to calcium carbonate under in situ conditions,
#       Geochemica et Cosmochemica Acta, 34:1261-1291, 1970.
# ****************************************************************************
# These references often disagree and give different fits for the same thing.
# They are not always just an update either; that is, Millero, 1995 may agree
#       with Millero, 1979, but differ from Millero, 1983.
# For WhichKs# = 7 (Peng choice) I used the same factors for KW, KP1, KP2,
#       KP3, and KSi as for the other cases. Peng et al didn't consider the
#       case of P different from 0. GEOSECS did consider pressure, but didn't
#       include Phos, Si, or OH, so including the factors here won't matter.
# For WhichKs# = 8 (freshwater) the values are from Millero, 1983 (for K1, K2,
#       and KW). The other aren't used (TB = TS = TF = TP = TSi = 0.), so
#       including the factors won't matter.
# ****************************************************************************
#       deltaVs are in cm3/mole
#       Kappas are in cm3/mole/bar
# ****************************************************************************
