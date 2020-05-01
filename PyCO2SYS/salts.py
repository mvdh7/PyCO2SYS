# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Estimate total molinities of seawater solutes from practical salinity."""

from autograd.numpy import nan, where


def ionstr_DOE94(Sal):
    """Ionic strength following DOE94."""
    # === CO2SYS.m comments: =======
    # This is from the DOE handbook, Chapter 5, p. 13/22, eq. 7.2.4.
    return 19.924 * Sal / (1000 - 1.005 * Sal)


def borate_C65(Sal):
    """Total borate in mol/kg-sw following C65."""
    # === CO2SYS.m comments: =======
    # This is .00001173*Sali, about 1% lower than Uppstrom's value
    # Culkin, F., in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    # GEOSECS references this, but this value is not explicitly given here
    # Output in mol/kg-SW
    return 0.0004106 * Sal / 35


def borate_U74(Sal):
    """Total borate in mol/kg-sw following U74."""
    # === CO2SYS.m comments: =======
    # Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
    # this is .000416*Sali/35. = .0000119*Sali
    # TB[FF] = (0.000232/10.811)*(Sal[FF]/1.80655); in mol/kg-SW.
    return 0.0004157 * Sal / 35


def borate_LKB10(Sal):
    """Total borate in mol/kg-sw following LKB10."""
    # === CO2SYS.m comments: =======
    # Lee, Kim, Byrne, Millero, Feely, Yong-Ming Liu. 2010.
    # Geochimica Et Cosmochimica Acta 74 (6): 1801-1811.
    # Output in mol/kg-SW.
    return 0.0004326 * Sal / 35


def calcium_C65(Sal):
    """Calcium in mol/kg-sw following C65."""
    # === CO2SYS.m comments: =======
    # *** CalculateCaforGEOSECS:
    # Culkin, F, in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
    # Culkin gives Ca = (.0213/40.078)*(Sal/1.80655) in mol/kg-SW
    # which corresponds to Ca = .01030*Sal/35.
    return 0.01026 * Sal / 35


def calcium_RT67(Sal):
    """Calcium in mol/kg-sw following RT67."""
    # === CO2SYS.m comments: =======
    # This is .010285*Sal/35
    return 0.02128 / 40.087 * Sal / 1.80655


def fluoride_R65(Sal):
    """Total fluoride in mol/kg-sw following R65."""
    # === CO2SYS.m comments: =======
    # Riley, J. P., Deep-Sea Research 12:219-220, 1965:
    # this is .000068*Sali/35. = .00000195*Sali; in mol/kg-SW.
    return (0.000067 / 18.998) * Sal / 1.80655


def sulfate_MR66(Sal):
    """Total sulfate in mol/kg-sw following MR66."""
    # === CO2SYS.m comments: =======
    # Morris, A. W., and Riley, J. P., Deep-Sea Research 13:699-705, 1966:
    # this is .02824*Sali/35. = .0008067*Sali; in mol/kg-SW.
    return (0.14 / 96.062) * Sal / 1.80655


def _co2sys_TB(Sal, WhichKs, WhoseTB):
    """Calculate total borate from salinity for the given options."""
    TB = where(WhichKs == 8, 0.0, nan)  # pure water
    TB = where((WhichKs == 6) | (WhichKs == 7), borate_C65(Sal), TB)
    F = (WhichKs != 6) & (WhichKs != 7) & (WhichKs != 8)
    TB = where(F & (WhoseTB == 1), borate_U74(Sal), TB)
    TB = where(F & (WhoseTB == 2), borate_LKB10(Sal), TB)
    return TB


def _co2sys_TCa(Sal, WhichKs):
    """Calculate total calcium from salinity for the given options."""
    F = (WhichKs == 6) | (WhichKs == 7)  # GEOSECS values
    TCa = where(F, calcium_C65(Sal), calcium_RT67(Sal))
    return TCa


def fromSal(Sal, WhichKs, WhoseTB):
    """Estimate total molinities of calcium, borate, fluoride and sulfate from salinity.

    Subfunctions based on Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    TB = _co2sys_TB(Sal, WhichKs, WhoseTB)
    TF = fluoride_R65(Sal)
    TS = sulfate_MR66(Sal)
    TCa = _co2sys_TCa(Sal, WhichKs)
    # Return equilibrating results as a dict for stability
    return {"TB": TB, "TF": TF, "TSO4": TS, "TCa": TCa}


def assemble(Sal, TSi, TPO4, TNH3, TH2S, WhichKs, WhoseTB):
    """Estimate total molinities from salinity and assemble along with other salts and
    related variables.
    """
    # Pure Water case:
    Sal = where(WhichKs == 8, 0.0, Sal)
    # GEOSECS and Pure Water:
    F = (WhichKs == 6) | (WhichKs == 8)
    TPO4 = where(F, 0.0, TPO4)
    TSi = where(F, 0.0, TSi)
    TNH3 = where(F, 0.0, TNH3)
    TH2S = where(F, 0.0, TH2S)
    # Convert micromol to mol
    TPO4 = TPO4 * 1e-6
    TSi = TSi * 1e-6
    TNH3 = TNH3 * 1e-6
    TH2S = TH2S * 1e-6
    totals = fromSal(Sal, WhichKs, WhoseTB)
    # The vector `PengCorrection` is used to modify the value of TA, for those
    # cases where WhichKs==7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, PengCorrection is 0 for all cases where WhichKs is not 7.
    PengCorrection = where(WhichKs == 7, TPO4, 0.0)
    # Add everything else to `totals` dict
    totals["TPO4"] = TPO4
    totals["TSi"] = TSi
    totals["TNH3"] = TNH3
    totals["TH2S"] = TH2S
    totals["Sal"] = Sal  # this is input `Sal` but with pure water case forced to 0.0
    totals["PengCorrection"] = PengCorrection
    return totals
