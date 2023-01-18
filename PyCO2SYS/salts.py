# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Estimate total substance contents of seawater solutes from practical salinity."""

from autograd import numpy as np


def ionic_strength_DOE94(salinity):
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


def borate_KSK18(salinity):
    """Total borate in mol/kg-sw following KSK18."""
    # Note that the reference provided an equation for μmol/kg-sw
    # This function divides it by a factor of 1e6 to convert to mol/kg-sw
    return (10.838 * salinity + 13.821) / 1e6


def borate_U74(salinity):
    """Total borate in mol/kg-sw following U74."""
    # === CO2SYS.m comments: =======
    # Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
    # this is .000416*Sali/35. = .0000119*Sali
    # total_borate[FF] = (0.000232/10.811)*(Sal[FF]/1.80655); in mol/kg-SW.
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


def get_total_borate(salinity, opt_k_carbonic, opt_total_borate):
    """Calculate total borate in mol/kg-sw from salinity for the given settings.

    GEOSECS cases follow C65, irrespective of user-provided opt_total_borate.
    All other cases follow user's opt_total_borate (i.e. U74 or LKB10).
    """
    # Pure water: zero total borate (case 8 stays like this)
    total_borate = np.zeros_like(salinity)
    # GEOSECS cases: follow C65, irrespective of opt_total_borate
    total_borate = np.where(
        (opt_k_carbonic == 6) | (opt_k_carbonic == 7),
        borate_C65(salinity),
        total_borate,
    )
    # All other cases: follow opt_total_borate
    F = (opt_k_carbonic != 6) & (opt_k_carbonic != 7) & (opt_k_carbonic != 8)
    total_borate = np.where(
        F & (opt_total_borate == 1), borate_U74(salinity), total_borate
    )
    total_borate = np.where(
        F & (opt_total_borate == 2), borate_LKB10(salinity), total_borate
    )
    total_borate = np.where(
        F & (opt_total_borate == 3), borate_KSK18(salinity), total_borate
    )
    return total_borate


def get_total_calcium(salinity, opt_k_carbonic):
    """Calculate total calcium in mol/kg-sw from salinity for the given settings.

    GEOSECS cases follow C65; all others (including freshwater) follow RT67.
    """
    F = (opt_k_carbonic == 6) | (opt_k_carbonic == 7)  # identify GEOSECS cases
    total_calcium = np.where(F, calcium_C65(salinity), calcium_RT67(salinity))
    return total_calcium


def from_salinity(salinity, opt_k_carbonic, opt_total_borate, totals=None):
    """Estimate total substance contents of calcium, borate, fluoride and sulfate, all
    in mol/kg-sw, from salinity, for the given settings.

    Subfunctions based on Constants, version 04.01, 1997-10-13, by Ernie Lewis.
    """
    if totals is None:
        totals = {}
    if "TB" not in totals:
        total_borate = get_total_borate(salinity, opt_k_carbonic, opt_total_borate)
        totals["TB"] = total_borate
    if "TF" not in totals:
        total_fluoride = fluoride_R65(salinity)
        totals["TF"] = total_fluoride
    if "TSO4" not in totals:
        total_sulfate = sulfate_MR66(salinity)
        totals["TSO4"] = total_sulfate
    if "TCa" not in totals:
        total_calcium = get_total_calcium(salinity, opt_k_carbonic)
        totals["TCa"] = total_calcium
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
    opt_k_carbonic,
    opt_total_borate,
    totals=None,
):
    """Estimate all total substance contents from salinity for the given settings and
    assemble into a dict along with user-provided total salt contents and related
    variables.
    """
    # Pure water case: set salinity to zero
    salinity = np.where(opt_k_carbonic == 8, 0.0, salinity)
    # GEOSECS and pure water cases: set nutrients to zero
    F = (opt_k_carbonic == 6) | (opt_k_carbonic == 8)
    total_phosphate = np.where(F, 0.0, total_phosphate)
    total_silicate = np.where(F, 0.0, total_silicate)
    total_ammonia = np.where(F, 0.0, total_ammonia)
    total_sulfide = np.where(F, 0.0, total_sulfide)
    # Convert µmol to mol for user-provided nutrients
    total_phosphate = total_phosphate * 1e-6
    total_silicate = total_silicate * 1e-6
    total_ammonia = total_ammonia * 1e-6
    total_sulfide = total_sulfide * 1e-6
    # Assemble dict of all
    totals = from_salinity(salinity, opt_k_carbonic, opt_total_borate, totals=totals)
    # `peng_correction` is used to modify the value of total alkalinity, for those
    # cases where opt_k_carbonic is 7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, PengCorrection is 0 for all cases where opt_k_carbonic is not 7.
    peng_correction = np.where(opt_k_carbonic == 7, total_phosphate, 0.0)
    totals["PengCorrection"] = peng_correction
    # Insert everything else into the `totals` dict
    totals["TPO4"] = total_phosphate
    totals["TSi"] = total_silicate
    totals["TNH3"] = total_ammonia
    totals["TH2S"] = total_sulfide
    # This is input `salinity` but with freshwater case values set to zero
    totals["Sal"] = salinity
    return totals


def from_salinity_nodict(
    salinity,
    opt_k_carbonic,
    opt_total_borate,
    total_alpha=None,
    total_beta=None,
    total_borate=None,
    total_calcium=None,
    total_fluoride=None,
    total_sulfate=None,
):
    """Estimate total substance contents of calcium, borate, fluoride and sulfate, all
    in mol/kg-sw, from salinity, for the given settings.

    Subfunctions based on Constants, version 04.01, 1997-10-13, by Ernie Lewis.
    """
    if total_alpha is None:
        total_alpha = 0.0
    if total_beta is None:
        total_beta = 0.0
    if total_borate is None:
        total_borate = get_total_borate(salinity, opt_k_carbonic, opt_total_borate)
    if total_calcium is None:
        total_calcium = get_total_calcium(salinity, opt_k_carbonic)
    if total_fluoride is None:
        total_fluoride = fluoride_R65(salinity)
    if total_sulfate is None:
        total_sulfate = sulfate_MR66(salinity)
    return (
        total_alpha,
        total_beta,
        total_borate,
        total_calcium,
        total_fluoride,
        total_sulfate,
    )


def assemble_nodict(
    salinity,
    total_silicate,
    total_phosphate,
    total_ammonia,
    total_sulfide,
    opt_k_carbonic,
    opt_total_borate,
    total_alpha=None,
    total_beta=None,
    total_borate=None,
    total_calcium=None,
    total_fluoride=None,
    total_sulfate=None,
):
    """Estimate all total substance contents from salinity for the given settings and
    assemble into variables along with user-provided total salt contents and related
    variables.
    """
    # Pure water case: set salinity to zero
    salinity = np.where(opt_k_carbonic == 8, 0.0, salinity)
    # GEOSECS and pure water cases: set nutrients to zero
    F = (opt_k_carbonic == 6) | (opt_k_carbonic == 8)
    total_phosphate = np.where(F, 0.0, total_phosphate)
    total_silicate = np.where(F, 0.0, total_silicate)
    total_ammonia = np.where(F, 0.0, total_ammonia)
    total_sulfide = np.where(F, 0.0, total_sulfide)
    # Convert µmol to mol for user-provided nutrients
    total_phosphate = total_phosphate * 1e-6
    total_silicate = total_silicate * 1e-6
    total_ammonia = total_ammonia * 1e-6
    total_sulfide = total_sulfide * 1e-6
    # Assemble variables of all salinity-derived totals
    (
        total_alpha,
        total_beta,
        total_borate,
        total_calcium,
        total_fluoride,
        total_sulfate,
    ) = from_salinity_nodict(
        salinity,
        opt_k_carbonic,
        opt_total_borate,
        total_alpha=total_alpha,
        total_beta=total_beta,
        total_borate=total_borate,
        total_calcium=total_calcium,
        total_fluoride=total_fluoride,
        total_sulfate=total_sulfate,
    )
    # `peng_correction` is used to modify the value of total alkalinity, for those
    # cases where opt_k_carbonic is 7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, peng_correction is 0 for all cases where opt_k_carbonic is not 7.
    peng_correction = np.where(opt_k_carbonic == 7, total_phosphate, 0.0)
    return (
        salinity,
        total_alpha,
        total_ammonia,
        total_beta,
        total_borate,
        total_calcium,
        total_fluoride,
        total_phosphate,
        total_silicate,
        total_sulfate,
        total_sulfide,
        peng_correction,
    )
