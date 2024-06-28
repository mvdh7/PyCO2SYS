# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
"""
PyCO2SYS.convert
================
Convert units and calculate conversion factors.

pH scale conversions
--------------------
There is a function for every variant of pH_<scale1>_to_<scale2>, where <scale1> and
<scale2> are any of free, total, sws and nbs.  Each function has a different set of
arguments, depending on what is needed.  For example, to get the conversion factor to
go from the total to the NBS scale, use:

  >>> factor = pyco2.convert.pH_tot_to_nbs(
        total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH
      )

``factor`` should be multiplied by [H+] or K value(s) on the total scale to convert to
the NBS scale.

Other functions
---------------
CO2aq_to_fCO2
    Convert aqueous CO2 content to fugacity.
fCO2_to_CO2aq
    Convert CO2 fugacity to aqueous content.
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


def CO2aq_to_fCO2(CO2, k_CO2):
    """Convert aqueous CO2 content to fugacity.

    Parameters
    ----------
    CO2 : float
        Aqueous CO2 content in µmol/kg-sw.
    k_CO2 : float
        CO2 solubility constant.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    return CO2 / k_CO2


def fCO2_to_CO2aq(fCO2, k_CO2):
    """Convert CO2 fugacity to aqueous content.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    k_CO2 : float
        CO2 solubility constant.

    Returns
    -------
    float
        Aqueous CO2 content in µmol/kg-sw.
    """
    return fCO2 * k_CO2


def celsius_to_kelvin(TempC):
    """Convert temperature from degC to K."""
    return TempC + constants.Tzero


def kelvin_to_celsius(TempK):
    """Convert temperature from K to degC."""
    return TempK - constants.Tzero


def decibar_to_bar(Pdbar):
    """Convert pressure from dbar to bar."""
    return Pdbar / 10.0


def bar_to_decibar(Pbar):
    """Convert pressure from bar to dbar."""
    return Pbar * 10.0


def pH_free_to_tot(total_sulfate, k_HSO4_free):
    """Free to total pH scale conversion factor.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 + 1e-6 * total_sulfate / k_HSO4_free


def pH_free_to_sws(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free):
    """Free to seawater pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 + 1e-6 * total_sulfate / k_HSO4_free + 1e-6 * total_fluoride / k_HF_free


def pH_sws_to_free(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free):
    """Seawater to free pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 / pH_free_to_sws(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free)


def pH_sws_to_tot(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free):
    """Seawater to total pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return pH_sws_to_free(
        total_fluoride, total_sulfate, k_HF_free, k_HSO4_free
    ) * pH_free_to_tot(total_sulfate, k_HSO4_free)


def pH_tot_to_free(total_sulfate, k_HSO4_free):
    """Total to free pH scale conversion factor.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 / pH_free_to_tot(total_sulfate, k_HSO4_free)


def pH_tot_to_sws(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free):
    """Total to seawater pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 / pH_sws_to_tot(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free)


def pH_sws_to_nbs(fH):
    """Seawater to NBS pH scale conversion factor.

    Parameters
    ----------
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return fH


def pH_nbs_to_sws(fH):
    """NBS to Seawater pH scale conversion factor.

    Parameters
    ----------
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 / pH_sws_to_nbs(fH)


def pH_tot_to_nbs(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH):
    """Total to NBS pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return pH_tot_to_sws(
        total_fluoride, total_sulfate, k_HF_free, k_HSO4_free
    ) * pH_sws_to_nbs(fH)


def pH_nbs_to_tot(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH):
    """NBS to Total pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 / pH_tot_to_nbs(
        total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH
    )


def pH_free_to_nbs(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH):
    """Free to NBS pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return pH_free_to_sws(
        total_fluoride, total_sulfate, k_HF_free, k_HSO4_free
    ) * pH_sws_to_nbs(fH)


def pH_nbs_to_free(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH):
    """NBS to Free pH scale conversion factor.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride (HF + F) in µmol/kg-sw.
    total_sulfate : float
        Total sulfate (HSO4 + SO4) in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.
    k_HSO4_free : float
        HSO4 dissociation constant on the free scale.
    fH : float
        Hydrogen ion activity coefficient.

    Returns
    -------
    float
        The conversion factor; multiply by [H+] to convert scale.
    """
    return 1.0 / pH_free_to_nbs(
        total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH
    )


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


def pH_to_all_scales(pH, pH_scale, totals, k_constants):
    """Calculate pH on all scales.

    This takes the pH on the given pH_scale and finds the pH on all scales.

    Based on FindpHOnAllScales, version 01.02, 01-08-97, by Ernie Lewis.
    """
    f2t = pH_free_to_tot(totals, k_constants)
    s2t = pH_sws_to_tot(totals, k_constants)
    factor = np.full(np.shape(pH), np.nan)
    factor = np.where(pH_scale == 1, 0.0, factor)  # Total
    factor = np.where(pH_scale == 2, np.log10(s2t), factor)  # Seawater
    factor = np.where(pH_scale == 3, np.log10(f2t), factor)  # Free
    factor = np.where(pH_scale == 4, np.log10(s2t / k_constants["fH"]), factor)  # NBS
    pH_total = pH - factor  # pH comes into this function on the given scale
    pH_sws = pH_total + np.log10(s2t)
    pH_free = pH_total + np.log10(f2t)
    pH_nbs = pH_total + np.log10(s2t / k_constants["fH"])
    return pH_total, pH_sws, pH_free, pH_nbs


def pH_sws_to_tot_P0(TempK, totals, k_constants, WhoseKSO4, WhoseKF):
    """Determine SWS to Total pH scale correction factor at zero pressure."""
    k_constants_P0 = copy.deepcopy(k_constants)
    k_constants_P0["KSO4"] = k_constants["KSO4_P0"]
    k_constants_P0["KF"] = k_constants["KF_P0"]
    return pH_sws_to_tot(totals, k_constants_P0)


def get_pHfactor_from_SWS(TempK, Sal, totals, k_constants, pHScale, WhichKs):
    """Determine pH scale conversion factors to go from SWS to input pHScale(s).
    The raw K values (not pK) should be multiplied by these to make the conversion.
    """
    if "fH" not in k_constants:
        k_constants["fH"] = pressured.fH(TempK, Sal, WhichKs)
    pHfactor = np.full(np.shape(pHScale), np.nan)
    pHfactor = np.where(
        pHScale == 1, pH_sws_to_tot(totals, k_constants), pHfactor
    )  # Total
    pHfactor = np.where(pHScale == 2, 1.0, pHfactor)  # Seawater (SWS)
    pHfactor = np.where(
        pHScale == 3,
        pH_sws_to_free(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free),
        pHfactor,
    )  # Free
    pHfactor = np.where(
        pHScale == 4, pH_sws_to_nbs(totals, k_constants), pHfactor
    )  # NBS
    k_constants["pHfactor_from_SWS"] = pHfactor
    return k_constants


def get_pHfactor_to_Free(TempK, Sal, totals, k_constants, pHScale, WhichKs):
    """Determine pH scale conversion factors to go from input pHScale(s) to Free.
    The raw K values (not pK) should be multiplied by these to make the conversion.
    """
    if "fH" not in k_constants:
        k_constants["fH"] = pressured.fH(TempK, Sal, WhichKs)
    pHfactor = np.full(np.shape(pHScale), np.nan)
    pHfactor = np.where(
        pHScale == 1, pH_tot_to_free(totals, k_constants), pHfactor
    )  # Total
    pHfactor = np.where(
        pHScale == 2,
        pH_sws_to_free(total_fluoride, total_sulfate, k_HF_free, k_HSO4_free),
        pHfactor,
    )  # Seawater (SWS)
    pHfactor = np.where(pHScale == 3, 1.0, pHfactor)  # Free
    pHfactor = np.where(
        pHScale == 4, pH_nbs_to_free(totals, k_constants), pHfactor
    )  # NBS
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
            (
                np.full(npts, arg, dtype=dtype)
                if np.size(arg) == 1
                else arg.ravel().astype(dtype)
            )
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
        (
            np.full(npts, arg, dtype=dtype)
            if np.size(arg) == 1
            else arg.ravel().astype(dtype)
        )
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
        (3, 1): 5,  # these two don't actually exist, but are needed for the
        (3, 2): 6,  # validation tests
    }
    KSO4CONSTANT, BORON = _flattenfirst((KSO4CONSTANT, BORON), int)[0]
    pairs = zip(KSO4CONSTANT, BORON)
    return np.array([pair2one[pair] for pair in pairs])
