# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
"""
PyCO2SYS.salts
==============
Calculate substance contents of various seawater solutes from practical salinity.
All values are returned in µmol/kg-sw.

Every function has the following signature:

  >>> substance_content = pyco2.salts.parameter_citation(salinity)

Functions
---------
ionic_strength_DOE94
    Ionic strength following DOE94.
total_borate_U74
    Total borate in µmol/kg-sw following U74.  Used when opt_total_borate = 1.
total_borate_LKB10
    Total borate in µmol/kg-sw following LKB10.  Used when opt_total_borate = 2.
total_borate_KSK18
    Total borate in µmol/kg-sw following KSK18.  Used when opt_total_borate = 3.
total_borate_C65
    Total borate in µmol/kg-sw following C65.  Used when opt_total_borate = 4.
total_fluoride_R65
    Total fluoride in µmol/kg-sw following R65.
total_sulfate_MR66
    Total sulfate in µmol/kg-sw following MR66.
Ca_RT67
    Calcium in µmol/kg-sw following RT67.  Used when opt_Ca = 1.
Ca_C65
    Calcium in µmol/kg-sw following C65.  Used when opt_Ca = 2.
"""

from .meta import valid


def ionic_strength_DOE94(salinity):
    """Ionic strength following DOE94.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Ionic strength following DOE94.
    """
    # === CO2SYS.m comments: =======
    # This is from the DOE handbook, Chapter 5, p. 13/22, eq. 7.2.4.
    return 19.924 * salinity / (1000 - 1.005 * salinity)


@valid(salinity=[34.1, 36.3])
def total_borate_U74(salinity):
    """Total borate in µmol/kg-sw following U74.  Used when opt_total_borate = 1.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Total borate in µmol/kg-sw following U74.
    """
    # === CO2SYS.m comments: =======
    # Uppstrom, L., Deep-Sea Research 21:161-162, 1974:
    # this is .000416*Sali/35. = .0000119*Sali
    # total_borate[FF] = (0.000232/10.811)*(Sal[FF]/1.80655); in mol/kg-SW.
    return 415.7 * salinity / 35


@valid(salinity=[34.1, 36.9])
def total_borate_LKB10(salinity):
    """Total borate in µmol/kg-sw following LKB10.  Used when opt_total_borate = 2.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Total borate in µmol/kg-sw following LKB10.
    """
    # === CO2SYS.m comments: =======
    # Lee, Kim, Byrne, Millero, Feely, Yong-Ming Liu. 2010.
    # Geochimica Et Cosmochimica Acta 74 (6): 1801-1811.
    # Output in mol/kg-SW.
    return 432.6 * salinity / 35


@valid(salinity=[0, 20])
def total_borate_KSK18(salinity):
    """Total borate in µmol/kg-sw following KSK18.  Used when opt_total_borate = 3.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Total borate in µmol/kg-sw following KSK18.
    """
    return 10.838 * salinity + 13.821


def total_borate_C65(salinity):
    """Total borate in µmol/kg-sw following C65.  Used when opt_total_borate = 4.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Total borate in µmol/kg-sw following C65.
    """
    # === CO2SYS.m comments: =======
    # This is .00001173*Sali, about 1% lower than Uppstrom's value
    # Culkin, F., in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    # GEOSECS references this, but this value is not explicitly given here
    # Output in mol/kg-SW
    return 410.6 * salinity / 35


def total_fluoride_R65(salinity):
    """Total fluoride in µmol/kg-sw following R65.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Total fluoride in µmol/kg-sw following R65.
    """
    # === CO2SYS.m comments: =======
    # Riley, J. P., Deep-Sea Research 12:219-220, 1965:
    # this is .000068*Sali/35. = .00000195*Sali; in mol/kg-SW.
    return 1e6 * (0.000067 / 18.998) * salinity / 1.80655


def total_sulfate_MR66(salinity):
    """Total sulfate in µmol/kg-sw following MR66.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Total sulfate in µmol/kg-sw following MR66.
    """
    # === CO2SYS.m comments: =======
    # Morris, A. W., and Riley, J. P., Deep-Sea Research 13:699-705, 1966:
    # this is .02824*Sali/35. = .0008067*Sali; in mol/kg-SW.
    return 1e6 * (0.14 / 96.062) * salinity / 1.80655


def Ca_RT67(salinity):
    """Calcium in µmol/kg-sw following RT67.  Used when opt_Ca = 1.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Calcium in µmol/kg-sw following RT67.
    """
    # === CO2SYS.m comments: =======
    # This is .010285*Sal/35
    return 1e6 * 0.02128 / 40.087 * salinity / 1.80655


def Ca_C65(salinity):
    """Calcium in µmol/kg-sw following C65.  Used when opt_Ca = 2.

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Calcium in µmol/kg-sw following C65.
    """
    # === CO2SYS.m comments: =======
    # *** CalculateCaforGEOSECS:
    # Culkin, F, in Chemical Oceanography, ed. Riley and Skirrow, 1965:
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
    # Culkin gives Ca = (.0213/40.078)*(Sal/1.80655) in mol/kg-SW
    # which corresponds to Ca = .01030*Sal/35.
    return 1e6 * 0.01026 * salinity / 35


def Mg_reference_composition(salinity):
    """Magnesium in µmol/kg-sw following the reference composition (MFWM08).

    Parameters
    ----------
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        Magnesium in µmol/kg-sw following the reference composition.
    """
    return 1e6 * 0.0547421 * salinity / 35
