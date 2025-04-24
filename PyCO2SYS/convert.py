# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
"""
PyCO2SYS.convert
================
Convert units and calculate conversions.

pH scale conversions
--------------------
There is a function for every variant of pH_<scale1>_to_<scale2>, where <scale1>
and <scale2> are any of free, total, sws and nbs.  Each function has a different
set of arguments, depending on what is needed.  For example, to get the conversion
to go from the total to the NBS scale, use:

  >>> converter = pyco2.convert.pH_tot_to_nbs(
        total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free, fH
      )

``converter`` should be added to pH or pK value(s) on the total scale to convert
to the NBS scale.

CO2 conversions
---------------
There is a function for every variant of <aCO2>_to_<bCO2>, where <aCO2> and <bCO2>
are any of pCO2, fCO2, CO2aq and xCO2.  Each function has a different set of arguments,
depending on what is needed.  For example, to convert pCO2 to fCO2, use:

  >>> fCO2 = pyco2.convert.pCO2_to_fCO2(pCO2, fugacity_factor)

Other conversions
-----------------
celsius_to_kelvin
kelvin_to_celsius
decibar_to_bar
bar_to_decibar
"""

import copy

from jax import numpy as np

from . import constants


def pCO2_to_fCO2(pCO2, fugacity_factor):
    """Convert CO2 partial pressure to fugacity."""
    return pCO2 * fugacity_factor


def fCO2_to_pCO2(fCO2, fugacity_factor):
    """Convert CO2 fugacity to partial pressure."""
    return fCO2 / fugacity_factor


def xCO2_to_fCO2(xCO2, fugacity_factor, vp_factor):
    """Convert CO2 dry mole fraction to fugacity."""
    return xCO2 * fugacity_factor * vp_factor


def fCO2_to_xCO2(fCO2, fugacity_factor, vp_factor):
    """Convert CO2 fugacity to dry mole fraction."""
    return fCO2 / (fugacity_factor * vp_factor)


def CO2aq_to_fCO2(CO2, pk_CO2):
    """Convert aqueous CO2 content to fugacity.

    Parameters
    ----------
    CO2 : float
        Aqueous CO2 content in µmol/kg-sw.
    pk_CO2 : float
        CO2 solubility constant.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    return CO2 / 10**-pk_CO2


def fCO2_to_CO2aq(fCO2, pk_CO2):
    """Convert CO2 fugacity to aqueous content.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    pk_CO2 : float
        CO2 solubility constant.

    Returns
    -------
    float
        Aqueous CO2 content in µmol/kg-sw.
    """
    return fCO2 * 10**-pk_CO2


def celsius_to_kelvin(temperature):
    """Convert temperature from degC to K."""
    return temperature + constants.Tzero


def kelvin_to_celsius(temperature_K):
    """Convert temperature from K to degC."""
    return temperature_K - constants.Tzero


def decibar_to_bar(pressure):
    """Convert pressure from dbar to bar."""
    return pressure / 10.0


def bar_to_decibar(pressure_bar):
    """Convert pressure from bar to dbar."""
    return pressure_bar * 10.0


def decibar_to_pascal(pressure):
    """Convert pressure from dbar to bar."""
    return pressure * 10000.0


def pH_free_to_tot(total_sulfate, pk_HSO4_free):
    """Free to total pH scale conversion.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -np.log10(1 + 1e-6 * total_sulfate / 10**-pk_HSO4_free)


def pH_free_to_sws(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free):
    """Free to seawater pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -np.log10(
        1
        + 1e-6 * total_sulfate / 10**-pk_HSO4_free
        + 1e-6 * total_fluoride / 10**-pk_HF_free
    )


def pH_sws_to_free(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free):
    """Seawater to free pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -pH_free_to_sws(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free)


def pH_sws_to_tot(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free):
    """Seawater to total pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return pH_sws_to_free(
        total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free
    ) + pH_free_to_tot(total_sulfate, pk_HSO4_free)


def pH_tot_to_free(total_sulfate, pk_HSO4_free):
    """Total to free pH scale conversion.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -pH_free_to_tot(total_sulfate, pk_HSO4_free)


def pH_tot_to_sws(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free):
    """Total to seawater pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -pH_sws_to_tot(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free)


def pH_sws_to_nbs(fH):
    """Seawater to NBS pH scale conversion.

    Parameters
    ----------
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -np.log10(fH)


def pH_nbs_to_sws(fH):
    """NBS to Seawater pH scale conversion.

    Parameters
    ----------
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -pH_sws_to_nbs(fH)


def pH_tot_to_nbs(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free, fH):
    """Total to NBS pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return pH_tot_to_sws(
        total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free
    ) + pH_sws_to_nbs(fH)


def pH_nbs_to_tot(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free, fH):
    """NBS to Total pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -pH_tot_to_nbs(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free, fH)


def pH_free_to_nbs(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free, fH):
    """Free to NBS pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return pH_free_to_sws(
        total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free
    ) + pH_sws_to_nbs(fH)


def pH_nbs_to_free(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free, fH):
    """NBS to Free pH scale conversion.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.
    pk_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion; add to pH or pK to convert scale.
    """
    return -pH_free_to_nbs(total_fluoride, total_sulfate, pk_HF_free, pk_HSO4_free, fH)


def fH_PTBO87(temperature, salinity):
    """fH following PTBO87."""
    # === CO2SYS.m comments: =======
    # Peng et al, Tellus 39B:439-458, 1987:
    # They reference the GEOSECS report, but round the value
    # given there off so that it is about .008 (1#) lower. It
    # doesn't agree with the check value they give on p. 456.
    TempK = temperature + 273.15
    return 1.29 - 0.00204 * TempK + (0.00046 - 0.00000148 * TempK) * salinity**2


def fH_TWB82(temperature, salinity):
    """fH following TWB82."""
    # === CO2SYS.m comments: =======
    # Takahashi et al, Chapter 3 in GEOSECS Pacific Expedition,
    # v. 3, 1982 (p. 80).
    TempK = temperature + 273.15
    return 1.2948 - 0.002036 * TempK + (0.0004607 - 0.000001475 * TempK) * salinity**2
