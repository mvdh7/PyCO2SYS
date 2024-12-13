# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
"""
PyCO2SYS.equilibria.p1atm
=========================
Calculate stoichiometric equilibrium constants under standard atmospheric pressure.

Functions
---------
k_CO2_W74
    Henry's constant for CO2 solubility in mol/kg-sw/atm following W74.
k_BOH3_total_D90b
    Boric acid dissociation constant following D90b.  Used when opt_k_BOH3 = 1.
k_BOH3_nbs_LTB69
    Boric acid dissociation constant following LTB69.  Used when opt_k_BOH3 = 2.
k_H2O_sws_M95
    Water dissociation constant on the seawater scale following M95.
    Used when opt_k_H2O = 1.
k_H2O_sws_M79
    Water dissociation constant on the seawater scale following M79, for freshwater.
    Used when opt_k_H2O = 2.
k_H2O_sws_HO58_M79
    Water dissociation constant on the seawater scale following HO58 refit by M79,
    for freshwater.  Used when opt_k_H2O = 3.
k_H2S_total_YM95
    Hydrogen sulfide dissociation constant on the total scale following YM95.
k_HF_free_DR79
    Hydrogen fluoride dissociation constant on the free scale following DR79.
    Used when opt_k_HF = 1.
k_HF_free_PF87
    Hydrogen fluoride dissociation constant on the free scale following PF87.
    Used when opt_k_HF = 2.
k_H3PO4_sws_YM95
    First phosphate dissociation constant on the seawater scale following YM95.
    Used when opt_k_phosphate = 1.
k_H2PO4_sws_YM95
    Second phosphate dissociation constant on the seawater scale following YM95.
    Used when opt_k_phosphate = 1.
k_HPO4_sws_YM95
    Third phosphate dissociation constant on the seawater scale following YM95.
    Used when opt_k_phosphate = 1.
k_H3PO4_sws_KP67
    First phosphate dissociation constant on the seawater scale following KP67.
    Used when opt_k_phosphate = 2.
k_H2PO4_nbs_KP67
    Second phosphate dissociation constant on the NBS scale following KP67.
    Used when opt_k_phosphate = 2.
k_HPO4_nbs_KP67
    Third phosphate dissociation constant on the NBS scale following KP67.
    Used when opt_k_phosphate = 2.
k_HSO4_free_D90a
    Bisulfate dissociation constant in mol/kg-sw on the free scale following D90a.
    Used when opt_k_HSO4 = 1.
k_HSO4_free_KRCB77
    Bisulfate dissociation constant in mol/kg-sw on the free scale following KRCB77.
    Used when opt_k_HSO4 = 2.
k_HSO4_free_WM13
    Bisulfate dissociation constant in mol/kg-sw on the free scale following WM13,
    with the corrections of WMW14.  Used when opt_k_HSO4 = 3.
k_Si_sws_YM95
    Silicate dissociation constant on the seawater scale following YM95.
    Used when opt_k_Si = 1.
k_Si_nbs_SMB64
    Silicate dissociation constant on the NBS scale following SMB64.
    Used when opt_k_Si = 2.
k_NH3_sws_YM95
    Ammonium association constant following YM95.  Used when opt_k_NH3 = 1.
k_NH3_tot_CW95
    Ammonium association constant following CW95.  Used when opt_k_NH3 = 2.
k_H2CO3_total_RRV93
    First carbonic acid dissociation constant following RRV93.
    Used when opt_k_carbonic = 1.
k_HCO3_total_RRV93
    Second carbonic acid dissociation constant following RRV93.
    Used when opt_k_carbonic = 1.
k_H2CO3_sws_GP89
    First carbonic acid dissociation constant following GP89.
    Used when opt_k_carbonic = 2.
k_HCO3_sws_GP89
    Second carbonic acid dissociation constant following GP89.
    Used when opt_k_carbonic = 2.
k_H2CO3_sws_H73_DM87
    First carbonic acid dissociation constant following DM87 refit of H73a and H73b.
    Used when opt_k_carbonic = 3.
k_HCO3_sws_H73_DM87
    Second carbonic acid dissociation constant following DM87 refit of H73a and H73b.
    Used when opt_k_carbonic = 3.
k_H2CO3_sws_MCHP73_DM87
    First carbonic acid dissociation constant following DM87 refit of MCHP73.
    Used when opt_k_carbonic = 4.
k_HCO3_sws_MCHP73_DM87
    Second carbonic acid dissociation constant following DM87 refit of MCHP73.
    Used when opt_k_carbonic = 4.
k_H2CO3_sws_HM_DM87
    First carbonic acid dissociation constant following DM87 refit of MCHP73 plus
    Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.
k_HCO3_sws_HM_DM87
    Second carbonic acid dissociation constant following DM87 refit of MCHP73 plus
    Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.
k_H2CO3_nbs_MCHP73
    First carbonic acid dissociation constant following MCHP73.
    Used when opt_k_carbonic = 6 or 7.
k_HCO3_nbs_MCHP73
    Second carbonic acid dissociation constant following MCHP73.
    Used when opt_k_carbonic = 6 or 7.
k_H2CO3_sws_M79
    First carbonic acid dissociation constant following M79, pure water case.
    Used when opt_k_carbonic = 8.
k_HCO3_sws_M79
    Second carbonic acid dissociation constant following M79, pure water case.
    Used when opt_k_carbonic = 8.
k_H2CO3_nbs_CW98
    First carbonic acid dissociation constant following CW98.
    Used when opt_k_carbonic = 9.
k_HCO3_nbs_CW98
    Second carbonic acid dissociation constant following CW98.
    Used when opt_k_carbonic = 9.
k_H2CO3_total_LDK00, k_HCO3_total_LDK00
    Carbonic acid dissociation constants following LDK00.
    Used when opt_k_carbonic = 10.
k_H2CO3_sws_MM02, k_HCO3_sws_MM02
    Carbonic acid dissociation constants following MM02.
    Used when opt_k_carbonic = 11.
k_H2CO3_sws_MPL02, k_H2CO3_sws_MPL02
    Carbonic acid dissociation constants following MPL02.
    Used when opt_k_carbonic = 12.
k_H2CO3_sws_MGH06, k_HCO3_sws_MGH06
    Carbonic acid dissociation constants following MGH06.
    Used when opt_k_carbonic = 13.
k_H2CO3_sws_M10, k_HCO3_sws_M10
    Carbonic acid dissociation constants following M10.
    Used when opt_k_carbonic = 14.
k_H2CO3_sws_WMW14, k_HCO3_sws_WMW14
    Carbonic acid dissociation constants following WM13/WMW14.
    Used when opt_k_carbonic = 15.
k_H2CO3_total_SLH20, k_HCO3_total_SLH20
    Carbonic acid dissociation constants following SLH20.
    Used when opt_k_carbonic = 16.
k_HCO3_total_SB21
    Second carbonic acid dissociation constant following SB21.
    Used when opt_k_carbonic = 17 together with K1 from WMW14.
k_H2CO3_total_PLR18, k_HCO3_total_PLR18
    Carbonic acid dissociation constants following PLR18.
    Used when opt_k_carbonic = 18.
"""

from jax import numpy as np

from .. import convert


def k_CO2_W74(temperature, salinity):
    """Henry's constant for CO2 solubility in mol/kg-sw/atm following W74.

    Parameters
    ----------

    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        CO2 solubility constant.
    """
    # === CO2SYS.m comments: =======
    # Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    # This is in mol/kg-SW/atm.
    TempK100 = (temperature + 273.15) / 100
    lnK0 = (
        -60.2409
        + 93.4517 / TempK100
        + 23.3585 * np.log(TempK100)
        + salinity * (0.023517 - 0.023656 * TempK100 + 0.0047036 * TempK100**2)
    )
    return np.exp(lnK0)


def k_HSO4_free_D90a(temperature, salinity, ionic_strength):
    """Bisulfate dissociation constant in mol/kg-sw on the free scale following D90a.
    Used when opt_k_HSO4 = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.
    ionic_strength : float
        Ionic strength in mol/kg-sw.

    Returns
    -------
    float
        HSO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Dickson, A. G., J. Chemical Thermodynamics, 22:113-127, 1990
    # The goodness of fit is .021.
    # It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    # TYPO on p. 121: the constant e9 should be e8.
    # Output KS is on the free pH scale in mol/kg-sw.
    # This is from eqs 22 and 23 on p. 123, and Table 4 on p 121:
    TempK = convert.celsius_to_kelvin(temperature)
    logTempK = np.log(TempK)
    lnk_HSO4 = (
        -4276.1 / TempK
        + 141.328
        - 23.093 * logTempK
        + (-13856 / TempK + 324.57 - 47.986 * logTempK) * np.sqrt(ionic_strength)
        + (35474 / TempK - 771.54 + 114.723 * logTempK) * ionic_strength
        + (-2698 / TempK) * np.sqrt(ionic_strength) * ionic_strength
        + (1776 / TempK) * ionic_strength**2
    )
    return np.exp(lnk_HSO4) * (1 - 0.001005 * salinity)


def k_HSO4_free_KRCB77(temperature, salinity, ionic_strength):
    """Bisulfate dissociation constant in mol/kg-sw on the free scale following KRCB77.
    Used when opt_k_HSO4 = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.
    ionic_strength : float
        Ionic strength in mol/kg-sw.

    Returns
    -------
    float
        HSO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Khoo, et al, Analytical Chemistry, 49(1):29-34, 1977
    # KS was found by titrations with a hydrogen electrode
    # of artificial seawater containing sulfate (but without F)
    # at 3 salinityinities from 20 to 45 and artificial seawater NOT
    # containing sulfate (nor F) at 16 salinityinities from 15 to 45,
    # both at temperatures from 5 to 40 deg C.
    # KS is on the Free pH scale (inherently so).
    # It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    # He finds log(beta) which = my pKS;
    # his beta is an association constant.
    # The rms error is .0021 in pKS, or about .5% in KS.
    # This is equation 20 on p. 33:
    # Output KS is on the free pH scale in mol/kg-sw.
    TempK = temperature + 273.15
    pk_HSO4 = (
        647.59 / TempK - 6.3451 + 0.019085 * TempK - 0.5208 * np.sqrt(ionic_strength)
    )
    return 10.0**-pk_HSO4 * (1 - 0.001005 * salinity)


def k_HSO4_free_WM13(temperature, salinity):
    """Bisulfate dissociation constant in mol/kg-sw on the free scale following WM13,
    with the corrections of WMW14.  Used when opt_k_HSO4 = 3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HSO4 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    logKS0 = (
        562.69486
        - 102.5154 * np.log(TempK)
        - 0.0001117033 * TempK**2
        + 0.2477538 * TempK
        - 13273.76 / TempK
    )
    logKSK0 = (
        (
            4.24666
            - 0.152671 * TempK
            + 0.0267059 * TempK * np.log(TempK)
            - 0.000042128 * TempK**2
        )
        * salinity**0.5
        + (0.2542181 - 0.00509534 * TempK + 0.00071589 * TempK * np.log(TempK))
        * salinity
        + (-0.00291179 + 0.0000209968 * TempK) * salinity**1.5
        + -0.0000403724 * salinity**2
    )
    k_HSO4 = (1 - 0.001005 * salinity) * 10.0 ** (logKSK0 + logKS0)
    return k_HSO4


def k_HF_free_DR79(temperature, salinity, ionic_strength):
    """Hydrogen fluoride dissociation constant on the free scale following DR79.
    Used when opt_k_HF = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HF dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979:
    # this is on the free pH scale in mol/kg-sw
    lnKF = 1590.2 / (temperature + 273.15) - 12.641 + 1.525 * ionic_strength**0.5
    return np.exp(lnKF) * (1 - 0.001005 * salinity)


def k_HF_free_PF87(temperature, salinity):
    """Hydrogen fluoride dissociation constant on the free scale following PF87.
    Used when opt_k_HF = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HF dissociation constant.
    """
    # Note that this is not currently used or an option in CO2SYS,
    # despite the equations below appearing in CO2SYS.m (commented out).
    # === CO2SYS.m comments: =======
    # Another expression exists for KF: Perez and Fraga 1987. Not used here
    # since ill defined for low salinityinity. (to be used for S: 10-40, T: 9-33)
    # Nonetheless, P&F87 might actually be better than the fit of D&R79 above,
    # which is based on only three salinityinities: [0 26.7 34.6]
    # Output is on the free pH scale in mol/kg-SW.
    lnKF = 874 / (temperature + 273.15) - 9.68 + 0.111 * salinity**0.5
    return np.exp(lnKF)


def k_BOH3_total_D90b(temperature, salinity):
    """Boric acid dissociation constant following D90b.  Used when opt_k_BOH3 = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        B(OH)3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Dickson, A. G., Deep-Sea Research 37:755-766, 1990.
    # lnKB is on Total pH scale
    sqrsalinity = np.sqrt(salinity)
    TempK = convert.celsius_to_kelvin(temperature)
    lnKBtop = (
        -8966.9
        - 2890.53 * sqrsalinity
        - 77.942 * salinity
        + 1.728 * sqrsalinity * salinity
        - 0.0996 * salinity**2
    )
    lnKB = (
        lnKBtop / TempK
        + 148.0248
        + 137.1942 * sqrsalinity
        + 1.62142 * salinity
        + (-24.4344 - 25.085 * sqrsalinity - 0.2474 * salinity) * np.log(TempK)
        + 0.053105 * sqrsalinity * TempK
    )
    return np.exp(lnKB)


def k_BOH3_nbs_LTB69(temperature, salinity):
    """Boric acid dissociation constant following LTB69.  Used when opt_k_BOH3 = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        B(OH)3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # This is for GEOSECS and Peng et al.
    # Lyman, John, UCLA Thesis, 1957
    # fit by Li et al, JGR 74:5507-5525, 1969.
    # logKB is on NBS pH scale
    logKB = -9.26 + 0.00886 * salinity + 0.01 * temperature
    return 10.0**logKB


def k_H2O_sws_M95(temperature, salinity):
    """Water dissociation constant on the seawater scale following M95.
    Used when opt_k_H2O = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2O dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Millero, Geochemica et Cosmochemica Acta 59:661-677, 1995.
    # his check value of 1.6 umol/kg-SW should be 6.2 (for ln(k))
    TempK = convert.celsius_to_kelvin(temperature)
    return np.exp(
        148.9802
        - 13847.26 / TempK
        - 23.6521 * np.log(TempK)
        + (-5.977 + 118.67 / TempK + 1.0495 * np.log(TempK)) * np.sqrt(salinity)
        - 0.01615 * salinity
    )


def k_H2O_sws_M79(temperature, salinity):
    """Water dissociation constant on the seawater scale following M79.
    Used when opt_k_H2O = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2O dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    TempK = convert.celsius_to_kelvin(temperature)
    return np.exp(
        148.9802
        - 13847.26 / TempK
        - 23.6521 * np.log(TempK)
        + (-79.2447 + 3298.72 / TempK + 12.0408 * np.log(TempK)) * np.sqrt(salinity)
        - 0.019813 * salinity
    )


def k_H2O_sws_HO58_M79(temperature):
    """Water dissociation constant on the seawater scale following HO58 refit by M79,
    for freshwater.  Used when opt_k_H2O = 3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2O dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    # refit data of Harned and Owen, The Physical Chemistry of
    # Electrolyte Solutions, 1958
    TempK = convert.celsius_to_kelvin(temperature)
    return np.exp(148.9802 - 13847.26 / TempK - 23.6521 * np.log(TempK))


def k_H3PO4_sws_KP67():
    """First phosphate dissociation constant on the seawater scale following KP67.
    Used when opt_k_phosphate = 2.

    Returns
    -------
    float
        H3PO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Peng et al don't include the contribution from the KP1 term,
    # but it is so small it doesn't contribute. It needs to be
    # kept so that the routines work ok.
    return 0.02  # This is already on the seawater scale!


def k_H2PO4_nbs_KP67(temperature):
    """Second phosphate dissociation constant on the NBS scale following KP67.
    Used when opt_k_phosphate = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        H2PO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Peng et al don't include the contribution from the KP1 term,
    # but it is so small it doesn't contribute. It needs to be
    # kept so that the routines work ok.
    # KP2, KP3 from Kester, D. R., and Pytkowicz, R. M.,
    # Limnology and Oceanography 12:243-252, 1967:
    # these are only for sals 33 to 36 and are on the NBS scale.
    return np.exp(-9.039 - 1450 / (temperature + 273.15))


def k_HPO4_nbs_KP67(temperature):
    """Third phosphate dissociation constant on the NBS scale following KP67.
    Used when opt_k_phosphate = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        H3PO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Peng et al don't include the contribution from the KP1 term,
    # but it is so small it doesn't contribute. It needs to be
    # kept so that the routines work ok.
    # KP2, KP3 from Kester, D. R., and Pytkowicz, R. M.,
    # Limnology and Oceanography 12:243-252, 1967:
    # these are only for sals 33 to 36 and are on the NBS scale.
    return np.exp(4.466 - 7276 / (temperature + 273.15))


def k_H3PO4_sws_YM95(temperature, salinity):
    """First phosphate dissociation constant on the seawater scale following YM95.
    Used when opt_k_phosphate = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H3PO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
    # KP1, KP2, KP3 are on the SWS pH scale in mol/kg-SW.
    TempK = convert.celsius_to_kelvin(temperature)
    lnKP1 = (
        -4576.752 / TempK
        + 115.54
        - 18.453 * np.log(TempK)
        + (-106.736 / TempK + 0.69171) * np.sqrt(salinity)
        + (-0.65643 / TempK - 0.01844) * salinity
    )
    return np.exp(lnKP1)


def k_H2PO4_sws_YM95(temperature, salinity):
    """Second phosphate dissociation constant on the seawater scale following YM95.
    Used when opt_k_phosphate = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2PO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
    # KP1, KP2, KP3 are on the SWS pH scale in mol/kg-SW.
    TempK = convert.celsius_to_kelvin(temperature)
    lnKP2 = (
        -8814.715 / TempK
        + 172.1033
        - 27.927 * np.log(TempK)
        + (-160.34 / TempK + 1.3566) * np.sqrt(salinity)
        + (0.37335 / TempK - 0.05778) * salinity
    )
    return np.exp(lnKP2)


def k_HPO4_sws_YM95(temperature, salinity):
    """Third phosphate dissociation constant on the seawater scale following YM95.
    Used when opt_k_phosphate = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HPO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
    # KP1, KP2, KP3 are on the SWS pH scale in mol/kg-SW.
    TempK = convert.celsius_to_kelvin(temperature)
    lnKP3 = (
        -3070.75 / TempK
        - 18.126
        + (17.27039 / TempK + 2.81197) * np.sqrt(salinity)
        + (-44.99486 / TempK - 0.09984) * salinity
    )
    return np.exp(lnKP3)


def k_Si_nbs_SMB64():
    """Silicate dissociation constant on the NBS scale following SMB64.
    Used when opt_k_Si = 2.

    Returns
    -------
    float
        Si(OH)4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Sillen, Martell, and Bjerrum,  Stability Constants of metal-ion
    # complexes, The Chemical Society (London), Special Publ. 17:751, 1964.
    return 0.0000000004


def k_Si_sws_YM95(temperature, salinity, ionic_strength):
    """Silicate dissociation constant on the seawater scale following YM95.
    Used when opt_k_Si = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.
    ionic_strength : float
        Ionic strength in mol/kg-sw.

    Returns
    -------
    float
        Si(OH)4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
    # KSi was given on the SWS pH scale in mol/kg-H2O, but is converted here
    # to mol/kg-sw.
    TempK = convert.celsius_to_kelvin(temperature)
    lnKSi = (
        -8904.2 / TempK
        + 117.4
        - 19.334 * np.log(TempK)
        + (-458.79 / TempK + 3.5913) * np.sqrt(ionic_strength)
        + (188.74 / TempK - 1.5998) * ionic_strength
        + (-12.1652 / TempK + 0.07871) * ionic_strength**2
    )
    return np.exp(lnKSi) * (1 - 0.001005 * salinity)


def k_H2CO3_total_RRV93(temperature, salinity):
    """First carbonic acid dissociation constant following RRV93.
    Used when opt_k_carbonic = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # ROY et al, Marine Chemistry, 44:249-267, 1993
    # (see also: Erratum, Marine Chemistry 45:337, 1994
    # and Erratum, Marine Chemistry 52:183, 1996)
    # Typo: in the abstract on p. 249: in the eq. for lnK1* the
    # last term should have S raised to the power 1.5.
    # They claim standard deviations (p. 254) of the fits as
    # .0048 for lnK1 (.5% in K1) and .007 in lnK2 (.7% in K2).
    # They also claim (p. 258) 2s precisions of .004 in pK1 and
    # .006 in pK2. These are consistent, but Andrew Dickson
    # (personal communication) obtained an rms deviation of about
    # .004 in pK1 and .003 in pK2. This would be a 2s precision
    # of about 2% in K1 and 1.5% in K2.
    # T:  0-45  S:  5-45. Total Scale. Artificial sewater.
    TempK = convert.celsius_to_kelvin(temperature)
    # This is eq. 29 on p. 254 and what they use in their abstract:
    lnK1 = (
        2.83655
        - 2307.1266 / TempK
        - 1.5529413 * np.log(TempK)
        + (-0.20760841 - 4.0484 / TempK) * np.sqrt(salinity)
        + 0.08468345 * salinity
        - 0.00654208 * np.sqrt(salinity) * salinity
    )
    return np.exp(lnK1) * (  # this is on the total pH scale in mol/kg-H2O
        1 - 0.001005 * salinity
    )  # convert to mol/kg-SW


def k_HCO3_total_RRV93(temperature, salinity):
    """Second carbonic acid dissociation constant following RRV93.
    Used when opt_k_carbonic = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    # This is eq. 30 on p. 254 and what they use in their abstract:
    lnK2 = (
        -9.226508
        - 3351.6106 / TempK
        - 0.2005743 * np.log(TempK)
        + (-0.106901773 - 23.9722 / TempK) * np.sqrt(salinity)
        + 0.1130822 * salinity
        - 0.00846934 * np.sqrt(salinity) * salinity
    )
    return np.exp(lnK2) * (  # this is on the total pH scale in mol/kg-H2O
        1 - 0.001005 * salinity
    )  # convert to mol/kg-SW


def k_H2CO3_sws_GP89(temperature, salinity):
    """First carbonic acid dissociation constant following GP89.
    Used when opt_k_carbonic = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # GOYET AND POISSON, Deep-Sea Research, 36(11):1635-1654, 1989
    # The 2s precision in pK1 is .011, or 2.5% in K1.
    # The 2s precision in pK2 is .02, or 4.5% in K2.
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1652 and what they use in the abstract:
    pK1 = (
        812.27 / TempK
        + 3.356
        - 0.00171 * salinity * np.log(TempK)
        + 0.000091 * salinity**2
    )
    return 10.0**-pK1  # this is on the SWS pH scale in mol/kg-SW


def k_HCO3_sws_GP89(temperature, salinity):
    """Second carbonic acid dissociation constant following GP89.
    Used when opt_k_carbonic = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1652 and what they use in the abstract:
    pK2 = (
        1450.87 / TempK
        + 4.604
        - 0.00385 * salinity * np.log(TempK)
        + 0.000182 * salinity**2
    )
    return 10.0**-pK2  # this is on the SWS pH scale in mol/kg-SW


def k_H2CO3_sws_H73_DM87(temperature, salinity):
    """First carbonic acid dissociation constant following DM87 refit of H73a and H73b.
    Used when opt_k_carbonic = 3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # HANSSON refit BY DICKSON AND MILLERO
    # Dickson and Millero, Deep-Sea Research, 34(10):1733-1743, 1987
    # (see also Corrigenda, Deep-Sea Research, 36:983, 1989)
    # refit data of Hansson, Deep-Sea Research, 20:461-478, 1973
    # and Hansson, Acta Chemica Scandanavia, 27:931-944, 1973.
    # on the SWS pH scale in mol/kg-SW.
    # Hansson gave his results on the Total scale (he called it
    # the seawater scale) and in mol/kg-SW.
    # Typo in DM on p. 1739 in Table 4: the equation for pK2*
    # for Hansson should have a .000132 *S^2
    # instead of a .000116 *S^2.
    # The 2s precision in pK1 is .013, or 3% in K1.
    # The 2s precision in pK2 is .017, or 4.1% in K2.
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from Table 4 on p. 1739.
    pK1 = 851.4 / TempK + 3.237 - 0.0106 * salinity + 0.000105 * salinity**2
    return 10.0**-pK1  # this is on the SWS pH scale in mol/kg-SW


def k_HCO3_sws_H73_DM87(temperature, salinity):
    """Second carbonic acid dissociation constant following DM87 refit of H73a and H73b.
    Used when opt_k_carbonic = 3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from Table 4 on p. 1739.
    pK2 = (
        -3885.4 / TempK
        + 125.844
        - 18.141 * np.log(TempK)
        - 0.0192 * salinity
        + 0.000132 * salinity**2
    )
    return 10.0**-pK2  # this is on the SWS pH scale in mol/kg-SW


def k_H2CO3_sws_MCHP73_DM87(temperature, salinity):
    """First carbonic acid dissociation constant following DM87 refit of MCHP73.
    Used when opt_k_carbonic = 4.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # MEHRBACH refit BY DICKSON AND MILLERO
    # Dickson and Millero, Deep-Sea Research, 34(10):1733-1743, 1987
    # (see also Corrigenda, Deep-Sea Research, 36:983, 1989)
    # refit data of Mehrbach et al, Limn Oc, 18(6):897-907, 1973
    # on the SWS pH scale in mol/kg-SW.
    # Mehrbach et al gave results on the NBS scale.
    # The 2s precision in pK1 is .011, or 2.6% in K1.
    # The 2s precision in pK2 is .020, or 4.6% in K2.
    # Valid for salinity 20-40.
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 4 on p. 1739.
    pK1 = (
        3670.7 / TempK
        - 62.008
        + 9.7944 * np.log(TempK)
        - 0.0118 * salinity
        + 0.000116 * salinity**2
    )
    return 10.0**-pK1  # this is on the SWS pH scale in mol/kg-SW


def k_HCO3_sws_MCHP73_DM87(temperature, salinity):
    """Second carbonic acid dissociation constant following DM87 refit of MCHP73.
    Used when opt_k_carbonic = 4.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 4 on p. 1739.
    pK2 = 1394.7 / TempK + 4.777 - 0.0184 * salinity + 0.000118 * salinity**2
    return 10.0**-pK2  # this is on the SWS pH scale in mol/kg-SW


def k_H2CO3_sws_HM_DM87(temperature, salinity):
    """First carbonic acid dissociation constant following DM87 refit of MCHP73 plus
    Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # HANSSON and MEHRBACH refit BY DICKSON AND MILLERO
    # Dickson and Millero, Deep-Sea Research,34(10):1733-1743, 1987
    # (see also Corrigenda, Deep-Sea Research, 36:983, 1989)
    # refit data of Hansson, Deep-Sea Research, 20:461-478, 1973,
    # Hansson, Acta Chemica Scandanavia, 27:931-944, 1973,
    # and Mehrbach et al, Limnol. Oceanogr.,18(6):897-907, 1973
    # on the SWS pH scale in mol/kg-SW.
    # Typo in DM on p. 1740 in Table 5: the second equation
    # should be pK2* =, not pK1* =.
    # The 2s precision in pK1 is .017, or 4% in K1.
    # The 2s precision in pK2 is .026, or 6% in K2.
    # Valid for salinity 20-40.
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1740.
    pK1 = 845 / TempK + 3.248 - 0.0098 * salinity + 0.000087 * salinity**2
    return 10.0**-pK1  # this is on the SWS pH scale in mol/kg-SW


def k_HCO3_sws_HM_DM87(temperature, salinity):
    """Second carbonic acid dissociation constant following DM87 refit of MCHP73 plus
    Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1740.
    pK2 = 1377.3 / TempK + 4.824 - 0.0185 * salinity + 0.000122 * salinity**2
    return 10.0**-pK2  # this is on the SWS pH scale in mol/kg-SW


def k_H2CO3_nbs_MCHP73(temperature, salinity):
    """First carbonic acid dissociation constant following MCHP73.
    Used when opt_k_carbonic = 6 or 7.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # GEOSECS and Peng et al use K1, K2 from Mehrbach et al,
    # Limnology and Oceanography, 18(6):897-907, 1973.
    # I.e., these are the original Mehrbach dissociation constants.
    # The 2s precision in pK1 is .005, or 1.2% in K1.
    # The 2s precision in pK2 is .008, or 2% in K2.
    salinity = np.where(salinity < 1e-16, 1e-16, salinity)
    # ^ added in v1.8.3, because salinity=0 gives log10(salinity)=-inf
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        -13.7201
        + 0.031334 * TempK
        + 3235.76 / TempK
        + 1.3e-5 * salinity * TempK
        - 0.1032 * salinity**0.5
    )
    return 10.0**-pK1  # this is on the NBS scale


def k_HCO3_nbs_MCHP73(temperature, salinity):
    """Second carbonic acid dissociation constant following MCHP73.
    Used when opt_k_carbonic = 6 or 7.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        5371.9645
        + 1.671221 * TempK
        + 0.22913 * salinity
        + 18.3802 * np.log10(salinity)
        - 128375.28 / TempK
        - 2194.3055 * np.log10(TempK)
        - 8.0944e-4 * salinity * TempK
        - 5617.11 * np.log10(salinity) / TempK
        + 2.136 * salinity / TempK
    )  # pK2 is not defined for salinity=0, since log10(0)=-inf, but since v1.8.3 we
    # return the value for salinity=1e-16 instead (this option shouldn't be used in
    # such low salinities anyway, it's only valid above 19!)
    return 10.0**-pK2  # this is on the NBS scale


def k_H2CO3_sws_M79(temperature):
    """First carbonic acid dissociation constant following M79, pure water case.
    Used when opt_k_carbonic = 8.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # PURE WATER CASE
    # Millero, F. J., Geochemica et Cosmochemica Acta 43:1651-1661, 1979:
    # K1 from refit data from Harned and Davis,
    # J American Chemical Society, 65:2030-2037, 1943.
    # K2 from refit data from Harned and Scholes,
    # J American Chemical Society, 43:1706-1709, 1941.
    # This is only to be used for salinity=0 water (note the absence of S in the
    # below formulations).
    # These are the thermodynamic Constants:
    TempK = convert.celsius_to_kelvin(temperature)
    lnK1 = 290.9097 - 14554.21 / TempK - 45.0575 * np.log(TempK)
    return np.exp(lnK1)


def k_HCO3_sws_M79(temperature):
    """Second carbonic acid dissociation constant following M79, pure water case.
    Used when opt_k_carbonic = 8.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    lnK2 = 207.6548 - 11843.79 / TempK - 33.6485 * np.log(TempK)
    return np.exp(lnK2)


def k_H2CO3_nbs_CW98(temperature, salinity):
    """First carbonic acid dissociation constant following CW98.
    Used when opt_k_carbonic = 9.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # From Cai and Wang 1998, for estuarine use.
    # Data used in this work is from:
    # K1: Merhback (1973) for S>15, for S<15: Mook and Keone (1975)
    # K2: Merhback (1973) for S>20, for S<20: Edmond and Gieskes (1970)
    # Sigma of residuals between fits and above data: Â±0.015, +0.040 for K1
    # and K2, respectively.
    # salinity 0-40, Temp 0.2-30
    # Limnol. Oceanogr. 43(4) (1998) 657-668
    # On the NBS scale
    # Their check values for F1 don't work out, not sure if this was correctly
    # published...
    # Conversion to SWS scale by division by fH is uncertain at low salinity due to
    # junction potential.
    TempK = convert.celsius_to_kelvin(temperature)
    F1 = 200.1 / TempK + 0.3220
    pK1 = (
        3404.71 / TempK
        + 0.032786 * TempK
        - 14.8435
        - 0.071692 * F1 * salinity**0.5
        + 0.0021487 * salinity
    )
    return 10.0**-pK1  # this is on the NBS scale


def k_HCO3_nbs_CW98(temperature, salinity):
    """Second carbonic acid dissociation constant following CW98.
    Used when opt_k_carbonic = 9.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    F2 = -129.24 / TempK + 1.4381
    pK2 = (
        2902.39 / TempK
        + 0.02379 * TempK
        - 6.4980
        - 0.3191 * F2 * salinity**0.5
        + 0.0198 * salinity
    )
    return 10.0**-pK2  # this is on the NBS scale


def k_H2CO3_total_LDK00(temperature, salinity):
    """First carbonic acid dissociation constant following LDK00.
    Used when opt_k_carbonic = 10.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # From Lueker, Dickson, Keeling, 2000
    # This is Mehrbach's data refit after conversion to the Total scale, for
    # comparison with their equilibrator work.
    # Mar. Chem. 70 (2000) 105-119
    # Total scale and kg-sw
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        3633.86 / TempK
        - 61.2172
        + 9.6777 * np.log(TempK)
        - 0.011555 * salinity
        + 0.0001152 * salinity**2
    )
    return 10.0**-pK1  # this is on the Total pH scale in mol/kg-SW


def k_HCO3_total_LDK00(temperature, salinity):
    """Second carbonic acid dissociation constant following LDK00.
    Used when opt_k_carbonic = 10.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        471.78 / TempK
        + 25.929
        - 3.16967 * np.log(TempK)
        - 0.01781 * salinity
        + 0.0001122 * salinity**2
    )
    return 10.0**-pK2  # this is on the Total pH scale in mol/kg-SW


def k_H2CO3_sws_MM02(temperature, salinity):
    """First carbonic acid dissociation constant following MM02.
    Used when opt_k_carbonic = 11.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Mojica Prieto and Millero 2002. Geochim. et Cosmochim. Acta. 66(14),
    # 2529-2540.
    # sigma for pK1 is reported to be 0.0056
    # sigma for pK2 is reported to be 0.010
    # This is from the abstract and pages 2536-2537
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        -43.6977
        - 0.0129037 * salinity
        + 1.364e-4 * salinity**2
        + 2885.378 / TempK
        + 7.045159 * np.log(TempK)
    )
    return 10.0**-pK1  # this is on the SWS pH scale in mol/kg-SW


def k_HCO3_sws_MM02(temperature, salinity):
    """Second carbonic acid dissociation constant following MM02.
    Used when opt_k_carbonic = 11.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        -452.0940
        + 13.142162 * salinity
        - 8.101e-4 * salinity**2
        + 21263.61 / TempK
        + 68.483143 * np.log(TempK)
        + (-581.4428 * salinity + 0.259601 * salinity**2) / TempK
        - 1.967035 * salinity * np.log(TempK)
    )
    return 10.0**-pK2  # this is on the SWS pH scale in mol/kg-SW


def k_H2CO3_sws_MPL02(temperature, salinity):
    """First carbonic acid dissociation constant following MPL02.
    Used when opt_k_carbonic = 12.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Millero et al., 2002. Deep-Sea Res. I (49) 1705-1723.
    # Calculated from overdetermined WOCE-era field measurements
    # sigma for pK1 is reported to be 0.005
    # sigma for pK2 is reported to be 0.008
    # This is from page 1715
    pK1 = 6.359 - 0.00664 * salinity - 0.01322 * temperature + 4.989e-5 * temperature**2
    return 10.0**-pK1  # this is on the SWS pH scale in mol/kg-SW


def k_HCO3_sws_MPL02(temperature, salinity):
    """Second carbonic acid dissociation constant following MPL02.
    Used when opt_k_carbonic = 12.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    pK2 = 9.867 - 0.01314 * salinity - 0.01904 * temperature + 2.448e-5 * temperature**2
    return 10.0**-pK2  # this is on the SWS pH scale in mol/kg-SW


def k_H2CO3_sws_MGH06(temperature, salinity):
    """First carbonic acid dissociation constant following MGH06.
    Used when opt_k_carbonic = 13.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # From Millero 2006 work on pK1 and pK2 from titrations
    # Millero, Graham, Huang, Bustos-Serrano, Pierrot. Mar.Chem. 100 (2006)
    # 80-94.
    # S=1 to 50, T=0 to 50. On seawater scale (SWS). From titrations in Gulf
    # Stream seawater.
    TempK = convert.celsius_to_kelvin(temperature)
    pK1_0 = -126.34048 + 6320.813 / TempK + 19.568224 * np.log(TempK)
    A_1 = 13.4191 * salinity**0.5 + 0.0331 * salinity - 5.33e-5 * salinity**2
    B_1 = -530.123 * salinity**0.5 - 6.103 * salinity
    C_1 = -2.06950 * salinity**0.5
    pK1 = A_1 + B_1 / TempK + C_1 * np.log(TempK) + pK1_0  # pK1 sigma = 0.0054
    return 10.0**-(pK1)


def k_HCO3_sws_MGH06(temperature, salinity):
    """Second carbonic acid dissociation constant following MGH06.
    Used when opt_k_carbonic = 13.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK2_0 = -90.18333 + 5143.692 / TempK + 14.613358 * np.log(TempK)
    A_2 = 21.0894 * salinity**0.5 + 0.1248 * salinity - 3.687e-4 * salinity**2
    B_2 = -772.483 * salinity**0.5 - 20.051 * salinity
    C_2 = -3.3336 * salinity**0.5
    pK2 = A_2 + B_2 / TempK + C_2 * np.log(TempK) + pK2_0  # pK2 sigma = 0.011
    return 10.0**-(pK2)


def k_H2CO3_sws_M10(temperature, salinity):
    """First carbonic acid dissociation constant following M10.
    Used when opt_k_carbonic = 14.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # From Millero, 2010, also for estuarine use.
    # Marine and Freshwater Research, v. 61, p. 139-142.
    # Fits through compilation of real seawater titration results:
    # Mehrbach et al. (1973), Mojica-Prieto & Millero (2002), Millero et al.
    # (2006)
    # Constants for K's on the SWS;
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from page 141
    pK10 = -126.34048 + 6320.813 / TempK + 19.568224 * np.log(TempK)
    # This is from their table 2, page 140.
    A1 = 13.4038 * salinity**0.5 + 0.03206 * salinity - 5.242e-5 * salinity**2
    B1 = -530.659 * salinity**0.5 - 5.8210 * salinity
    C1 = -2.0664 * salinity**0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    return 10.0**-pK1


def k_HCO3_sws_M10(temperature, salinity):
    """Second carbonic acid dissociation constant following M10.
    Used when opt_k_carbonic = 14.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from page 141
    pK20 = -90.18333 + 5143.692 / TempK + 14.613358 * np.log(TempK)
    # This is from their table 3, page 140.
    A2 = 21.3728 * salinity**0.5 + 0.1218 * salinity - 3.688e-4 * salinity**2
    B2 = -788.289 * salinity**0.5 - 19.189 * salinity
    C2 = -3.374 * salinity**0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    return 10.0**-pK2


def _kH2CO3_WMW14(TempK):
    """Scale-independent part of WMW14 carbonic acid dissociation constants."""
    pK1_0 = -126.34048 + 6320.813 / TempK + 19.568224 * np.log(TempK)
    pK2_0 = -90.18333 + 5143.692 / TempK + 14.613358 * np.log(TempK)
    return pK1_0, pK2_0


def k_H2CO3_sws_WMW14(temperature, salinity):
    """First carbonic acid dissociation constant following WM13/WMW14.
    Used when opt_k_carbonic = 15.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # From Waters, Millero, Woosley 2014
    # Mar. Chem., 165, 66-67, 2014
    # Corrigendum to "The free proton concentration scale for seawater pH".
    # Effectively, this is an update of Millero (2010) formulation
    # (WhichKs==14)
    # Constants for K's on the SWS;
    TempK = convert.celsius_to_kelvin(temperature)
    pK10 = _kH2CO3_WMW14(TempK)[0]
    A1 = 13.409160 * salinity**0.5 + 0.031646 * salinity - 5.1895e-5 * salinity**2
    B1 = -531.3642 * salinity**0.5 - 5.713 * salinity
    C1 = -2.0669166 * salinity**0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    return 10.0**-pK1


def k_HCO3_sws_WMW14(temperature, salinity):
    """Second carbonic acid dissociation constant following WM13/WMW14.
    Used when opt_k_carbonic = 15.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK20 = _kH2CO3_WMW14(TempK)[1]
    A2 = 21.225890 * salinity**0.5 + 0.12450870 * salinity - 3.7243e-4 * salinity**2
    B2 = -779.3444 * salinity**0.5 - 19.91739 * salinity
    C2 = -3.3534679 * salinity**0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    return 10.0**-pK2


def k_H2CO3_TOT_WMW14(TempK, salinity):
    """Carbonic acid dissociation constants, Total scale, following WM13/WMW14."""
    # Coefficients from the corrigendum document [WMW14]
    pK10, pK20 = _kH2CO3_WMW14(TempK)
    A1 = 13.568513 * salinity**0.5 + 0.031645 * salinity - 5.3834e-5 * salinity**2
    B1 = -539.2304 * salinity**0.5 - 5.635 * salinity
    C1 = -2.0901396 * salinity**0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    K1 = 10.0**-pK1
    A2 = 21.389248 * salinity**0.5 + 0.12452358 * salinity - 3.7447e-4 * salinity**2
    B2 = -787.3736 * salinity**0.5 - 19.84233 * salinity
    C2 = -3.3773006 * salinity**0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    K2 = 10.0**-pK2
    return K1, K2


def k_H2CO3_FREE_WMW14(TempK, salinity):
    """Carbonic acid dissociation constants, Free scale, following WM13/WMW14."""
    # Coefficients from the corrigendum document [WMW14]
    pK10, pK20 = _kH2CO3_WMW14(TempK)
    A1 = 5.592953 * salinity**0.5 + 0.028845 * salinity - 6.388e-5 * salinity**2
    B1 = -225.7489 * salinity**0.5 - 4.761 * salinity
    C1 = -0.8715109 * salinity**0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    K1 = 10.0**-pK1
    A2 = 13.396949 * salinity**0.5 + 0.12193009 * salinity - 3.8362e-4 * salinity**2
    B2 = -472.8633 * salinity**0.5 - 19.03634 * salinity
    C2 = -2.1563270 * salinity**0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    K2 = 10.0**-pK2
    return K1, K2


def k_H2CO3_total_SLH20(temperature, salinity):
    """First carbonic acid dissociation constant following SLH20.
    Used when opt_k_carbonic = 16.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # Coefficients and their 95% confidence intervals from SLH20 Table 1.
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        8510.63 / TempK  # ±1139.8
        - 172.4493  # ±26.131
        + 26.32996 * np.log(TempK)  # ±3.9161
        - 0.011555 * salinity
        + 0.0001152 * salinity**2
    )
    return 10.0**-pK1  # this is on the Total pH scale in mol/kg-SW


def k_HCO3_total_SLH20(temperature, salinity):
    """Second carbonic acid dissociation constant following SLH20.
    Used when opt_k_carbonic = 16.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        4226.23 / TempK  # ±1050.8
        - 59.4636  # ±24.016
        + 9.60817 * np.log(TempK)  # ±3.5966
        - 0.01781 * salinity
        + 0.0001122 * salinity**2
    )
    return 10.0**-pK2  # this is on the Total pH scale in mol/kg-SW


def k_HCO3_total_SB21(temperature, salinity):
    """Second carbonic acid dissociation constant following SB21.
    Used when opt_k_carbonic = 17 together with K1 from WMW14.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        116.8067
        - 3655.02 / TempK
        - 16.45817 * np.log(TempK)
        + 0.04523 * salinity
        - 0.615 * np.sqrt(salinity)
        - 0.0002799 * salinity**2
        + 4.969 * salinity / TempK
    )
    return 10.0**-pK2


def k_H2CO3_total_PLR18(temperature, salinity):
    """First carbonic acid dissociation constant following PLR18.
    Used when opt_k_carbonic = 18.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2CO3 dissociation constant.
    """
    # For 33 < salinity < 100, -6 < temperature < 25 °C.
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        -176.48
        + 6.14528 * salinity**0.5
        - 0.127714 * salinity
        + 7.396e-5 * salinity**2
        + (9914.37 - 622.886 * salinity**0.5 + 29.714 * salinity) / TempK
        + (26.05129 - 0.666812 * salinity**0.5) * np.log(TempK)
    )
    return 10.0**-pK1


def k_HCO3_total_PLR18(temperature, salinity):
    """Second carbonic acid dissociation constant following PLR18.
    Used when opt_k_carbonic = 18.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        -323.52692
        + 27.557655 * salinity**0.5
        + 0.154922 * salinity
        - 2.48396e-4 * salinity**2
        + (14763.287 - 1014.819 * salinity**0.5 - 14.35223 * salinity) / TempK
        + (50.385807 - 4.4630415 * salinity**0.5) * np.log(TempK)
    )
    return 10.0**-pK2


def k_H2S_total_YM95(temperature, salinity):
    """Hydrogen sulfide dissociation constant on the total scale following YM95.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        H2S dissociation constant.
    """
    # === CO2SYS_v1_21.m comments: =======
    # H2S  Millero et. al.( 1988)  Limnol. Oceanogr. 33,269-274.
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995. Total Scale.
    # Yao Millero say equations have been refitted to SWS scale but not true as
    # they agree with Millero 1988 which are on Total Scale.
    # Also, calculations agree at high H2S with AquaEnv when assuming it is on
    # Total Scale.
    TempK = convert.celsius_to_kelvin(temperature)
    lnkH2S = (
        225.838
        - 13275.3 / TempK
        - 34.6435 * np.log(TempK)
        + 0.3449 * np.sqrt(salinity)
        - 0.0274 * salinity
    )
    return np.exp(lnkH2S)


def k_NH3_sws_YM95(temperature, salinity):
    """Ammonium association constant following YM95.  Used when opt_k_NH3 = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        NH3 association constant.
    """
    # === CO2SYS_v1_21.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995   SWS
    TempK = convert.celsius_to_kelvin(temperature)
    lnkNH3 = (
        -6285.33 / TempK
        + 0.0001635 * TempK
        - 0.25444
        + (0.46532 - 123.7184 / TempK) * np.sqrt(salinity)
        + (-0.01992 + 3.17556 / TempK) * salinity
    )
    return np.exp(lnkNH3)


def k_NH3_total_CW95(temperature, salinity):
    """Ammonium association constant following CW95.  Used when opt_k_NH3 = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    salinity : float
        Practical salinity.

    Returns
    -------
    float
        NH3 association constant.
    """
    # === CO2SYS_v1_21.m comments: =======
    # Clegg Whitfield 1995
    # Geochimica et Cosmochimica Acta, Vol. 59, No. 12. pp. 2403-2421
    # eq (18)  Total scale   t=[-2 to 40 oC]  S=[0 to 40 ppt]   pK=+-0.00015
    TempK = convert.celsius_to_kelvin(temperature)
    PKNH3expCW = 9.244605 - 2729.33 * (1 / 298.15 - 1 / TempK)
    PKNH3expCW += (0.04203362 - 11.24742 / TempK) * salinity**0.25
    PKNH3expCW += (
        -13.6416 + 1.176949 * TempK**0.5 - 0.02860785 * TempK + 545.4834 / TempK
    ) * salinity**0.5
    PKNH3expCW += (
        -0.1462507 + 0.0090226468 * TempK**0.5 - 0.0001471361 * TempK + 10.5425 / TempK
    ) * salinity**1.5
    PKNH3expCW += (
        0.004669309 - 0.0001691742 * TempK**0.5 - 0.5677934 / TempK
    ) * salinity**2
    PKNH3expCW += (-2.354039e-05 + 0.009698623 / TempK) * salinity**2.5
    KNH3 = 10.0**-PKNH3expCW  # this is on the total pH scale in mol/kg-H2O
    KNH3 = KNH3 * (1 - 0.001005 * salinity)  # convert to mol/kg-SW
    return KNH3
