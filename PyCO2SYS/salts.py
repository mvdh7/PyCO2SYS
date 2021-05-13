# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Estimate total molinities of seawater solutes from practical salinity."""

from autograd import numpy as np


def ionstr_DOE94(salinity):
    """Ionic strength following DOE94."""
    # === CO2SYS.m comments: =======
    # This is from the DOE handbook, Chapter 5, p. 13/22, eq. 7.2.4.
    return 19.924 * salinity / (1000 - 1.005 * salinity)


def borate_C65(salinity):
    """Total borate in mol/kg-sw following C65."""
    # === CO2SYS.m comments: =======
    # This is .00001173*Sali, about 1% lower than Uppstrom's value
    # Culkin, F., in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    # GEOSECS references this, but this value is not explicitly given here
    # Output in mol/kg-SW
    return 0.0004106 * salinity / 35


def borate_U74(salinity):
    """Total borate in mol/kg-sw following U74."""
    # === CO2SYS.m comments: =======
    # Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
    # this is .000416*Sali/35. = .0000119*Sali
    # TB[FF] = (0.000232/10.811)*(Sal[FF]/1.80655); in mol/kg-SW.
    return 0.0004157 * salinity / 35


def borate_LKB10(salinity):
    """Total borate in mol/kg-sw following LKB10."""
    # === CO2SYS.m comments: =======
    # Lee, Kim, Byrne, Millero, Feely, Yong-Ming Liu. 2010.
    # Geochimica Et Cosmochimica Acta 74 (6): 1801-1811.
    # Output in mol/kg-SW.
    return 0.0004326 * salinity / 35


def calcium_C65(salinity):
    """Calcium in mol/kg-sw following C65."""
    # === CO2SYS.m comments: =======
    # *** CalculateCaforGEOSECS:
    # Culkin, F, in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
    # Culkin gives Ca = (.0213/40.078)*(Sal/1.80655) in mol/kg-SW
    # which corresponds to Ca = .01030*Sal/35.
    return 0.01026 * salinity / 35


def calcium_RT67(salinity):
    """Calcium in mol/kg-sw following RT67."""
    # === CO2SYS.m comments: =======
    # This is .010285*Sal/35
    return 0.02128 / 40.087 * salinity / 1.80655


def fluoride_R65(salinity):
    """Total fluoride in mol/kg-sw following R65."""
    # === CO2SYS.m comments: =======
    # Riley, J. P., Deep-Sea Research 12:219-220, 1965:
    # this is .000068*Sali/35. = .00000195*Sali; in mol/kg-SW.
    return (0.000067 / 18.998) * salinity / 1.80655


def sulfate_MR66(salinity):
    """Total sulfate in mol/kg-sw following MR66."""
    # === CO2SYS.m comments: =======
    # Morris, A. W., and Riley, J. P., Deep-Sea Research 13:699-705, 1966:
    # this is .02824*Sali/35. = .0008067*Sali; in mol/kg-SW.
    return (0.14 / 96.062) * salinity / 1.80655


def _co2sys_TB(salinity, WhichKs, WhoseTB):
    """Calculate total borate from salinity for the given options."""
    TB = np.where(WhichKs == 8, 0.0, np.nan)  # pure water
    TB = np.where((WhichKs == 6) | (WhichKs == 7), borate_C65(salinity), TB)
    F = (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8)
    TB = np.where(F & (WhoseTB == 1), borate_U74(salinity), TB)
    TB = np.where(F & (WhoseTB == 2), borate_LKB10(salinity), TB)
    return TB


def _co2sys_TCa(salinity, WhichKs):
    """Calculate total calcium from salinity for the given options."""
    F = (WhichKs == 6) | (WhichKs == 7)  # GEOSECS values
    TCa = np.where(F, calcium_C65(salinity), calcium_RT67(salinity))
    return TCa


def fromSal(salinity, WhichKs, WhoseTB, totals=None):
    """Estimate total molinities of calcium, borate, fluoride and sulfate from salinity.

    Subfunctions based on Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    if totals is None:
        totals = {}
    if "TB" not in totals:
        TB = _co2sys_TB(salinity, WhichKs, WhoseTB)
        totals["TB"] = TB
    if "TF" not in totals:
        TF = fluoride_R65(salinity)
        totals["TF"] = TF
    if "TSO4" not in totals:
        TSO4 = sulfate_MR66(salinity)
        totals["TSO4"] = TSO4
    if "TCa" not in totals:
        TCa = _co2sys_TCa(salinity, WhichKs)
        totals["TCa"] = TCa
    if "total_alpha" not in totals:
        totals["total_alpha"] = 0.0
    if "total_beta" not in totals:
        totals["total_beta"] = 0.0
    return totals


def assemble(
    salinity,
    total_silicate,
    total_phosphate,
    total_ammonia,
    total_sulfide,
    WhichKs,
    WhoseTB,
    totals=None,
):
    """Estimate total molinities from salinity and assemble along with other salts and
    related variables.
    """
    # Pure Water case:
    salinity = np.where(WhichKs == 8, 0.0, salinity)
    # GEOSECS and Pure Water:
    F = (WhichKs == 6) | (WhichKs == 8)
    total_phosphate = np.where(F, 0.0, total_phosphate)
    total_silicate = np.where(F, 0.0, total_silicate)
    total_ammonia = np.where(F, 0.0, total_ammonia)
    total_sulfide = np.where(F, 0.0, total_sulfide)
    # Convert micromol to mol
    total_phosphate = total_phosphate * 1e-6
    total_silicate = total_silicate * 1e-6
    total_ammonia = total_ammonia * 1e-6
    total_sulfide = total_sulfide * 1e-6
    totals = fromSal(salinity, WhichKs, WhoseTB, totals=totals)
    # The vector `PengCorrection` is used to modify the value of TA, for those
    # cases where WhichKs==7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, PengCorrection is 0 for all cases where WhichKs is not 7.
    peng_correction = np.where(WhichKs == 7, total_phosphate, 0.0)
    # Add everything else to `totals` dict
    totals["TPO4"] = total_phosphate
    totals["TSi"] = total_silicate
    totals["TNH3"] = total_ammonia
    totals["TH2S"] = total_sulfide
    totals[
        "Sal"
    ] = salinity  # this is input `Sal` but with pure water case forced to 0.0
    totals["PengCorrection"] = peng_correction
    return totals
