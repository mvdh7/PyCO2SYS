# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
"""
PyCO2SYS.equilibria.p1atm
=========================
Calculate stoichiometric equilibrium constants under standard atmospheric
pressure.

Each function has a corresponding function with the same name prefixed by
`coeffs_`, which returns the set of coefficients needed as the first argument
for the pK function.

Functions
---------
pk_CO2_W74
    Henry's constant for CO2 solubility in mol/kg-sw/atm following W74.
pk_BOH3_total_D90b
    Boric acid dissociation constant following D90b.
    Used when opt_k_BOH3 = 1.
pk_BOH3_nbs_LTB69
    Boric acid dissociation constant following LTB69.
    Used when opt_k_BOH3 = 2.
pk_H2O_sws_M95
    Water dissociation constant on the seawater scale following M95.
    Used when opt_k_H2O = 1.
pk_H2O_sws_M79
    Water dissociation constant on the seawater scale following M79, for
    freshwater.
    Used when opt_k_H2O = 2.
pk_H2O_sws_HO58_M79
    Water dissociation constant on the seawater scale following HO58 refit by
    M79, for freshwater.  Used when opt_k_H2O = 3.
pk_H2S_total_YM95
    Hydrogen sulfide dissociation constant on the total scale following YM95.
pk_HF_free_DR79
    Hydrogen fluoride dissociation constant on the free scale following DR79.
    Used when opt_k_HF = 1.
pk_HF_free_PF87
    Hydrogen fluoride dissociation constant on the free scale following PF87.
    Used when opt_k_HF = 2.
pk_H3PO4_sws_YM95
    First phosphate dissociation constant on the seawater scale following YM95.
    Used when opt_k_phosphate = 1.
pk_H2PO4_sws_YM95
    Second phosphate dissociation constant on the seawater scale following
    YM95.
    Used when opt_k_phosphate = 1.
pk_HPO4_sws_YM95
    Third phosphate dissociation constant on the seawater scale following
    YM95.
    Used when opt_k_phosphate = 1.
pk_H3PO4_sws_KP67
    First phosphate dissociation constant on the seawater scale following KP67.
    Used when opt_k_phosphate = 2.
pk_H2PO4_nbs_KP67
    Second phosphate dissociation constant on the NBS scale following KP67.
    Used when opt_k_phosphate = 2.
pk_HPO4_nbs_KP67
    Third phosphate dissociation constant on the NBS scale following KP67.
    Used when opt_k_phosphate = 2.
pk_HSO4_free_D90a
    Bisulfate dissociation constant in mol/kg-sw on the free scale following
    D90a.
    Used when opt_k_HSO4 = 1.
pk_HSO4_free_KRCB77
    Bisulfate dissociation constant in mol/kg-sw on the free scale following
    KRCB77.
    Used when opt_k_HSO4 = 2.
pk_HSO4_free_WM13
    Bisulfate dissociation constant in mol/kg-sw on the free scale following
    WM13, with the corrections of WMW14.  Used when opt_k_HSO4 = 3.
pk_Si_sws_YM95
    Silicate dissociation constant on the seawater scale following YM95.
    Used when opt_k_Si = 1.
pk_Si_nbs_SMB64
    Silicate dissociation constant on the NBS scale following SMB64.
    Used when opt_k_Si = 2.
pk_NH3_sws_YM95
    Ammonium association constant following YM95.  Used when opt_k_NH3 = 1.
pk_NH3_tot_CW95
    Ammonium association constant following CW95.  Used when opt_k_NH3 = 2.
pk_H2CO3_total_RRV93
    First carbonic acid dissociation constant following RRV93.
    Used when opt_k_carbonic = 1.
pk_HCO3_total_RRV93
    Second carbonic acid dissociation constant following RRV93.
    Used when opt_k_carbonic = 1.
pk_H2CO3_sws_GP89
    First carbonic acid dissociation constant following GP89.
    Used when opt_k_carbonic = 2.
pk_HCO3_sws_GP89
    Second carbonic acid dissociation constant following GP89.
    Used when opt_k_carbonic = 2.
pk_H2CO3_sws_H73_DM87
    First carbonic acid dissociation constant following DM87 refit of H73a and
    H73b.  Used when opt_k_carbonic = 3.
pk_HCO3_sws_H73_DM87
    Second carbonic acid dissociation constant following DM87 refit of H73a and
    H73b.  Used when opt_k_carbonic = 3.
pk_H2CO3_sws_MCHP73_DM87
    First carbonic acid dissociation constant following DM87 refit of MCHP73.
    Used when opt_k_carbonic = 4.
pk_HCO3_sws_MCHP73_DM87
    Second carbonic acid dissociation constant following DM87 refit of MCHP73.
    Used when opt_k_carbonic = 4.
pk_H2CO3_sws_HM_DM87
    First carbonic acid dissociation constant following DM87 refit of MCHP73
    plus Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.
pk_HCO3_sws_HM_DM87
    Second carbonic acid dissociation constant following DM87 refit of MCHP73
    plus Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.
pk_H2CO3_nbs_MCHP73
    First carbonic acid dissociation constant following MCHP73.
    Used when opt_k_carbonic = 6 or 7.
pk_HCO3_nbs_MCHP73
    Second carbonic acid dissociation constant following MCHP73.
    Used when opt_k_carbonic = 6 or 7.
pk_H2CO3_sws_M79
    First carbonic acid dissociation constant following M79, pure water case.
    Used when opt_k_carbonic = 8.
pk_HCO3_sws_M79
    Second carbonic acid dissociation constant following M79, pure water case.
    Used when opt_k_carbonic = 8.
pk_H2CO3_nbs_CW98
    First carbonic acid dissociation constant following CW98.
    Used when opt_k_carbonic = 9.
pk_HCO3_nbs_CW98
    Second carbonic acid dissociation constant following CW98.
    Used when opt_k_carbonic = 9.
pk_H2CO3_total_LDK00, pk_HCO3_total_LDK00
    Carbonic acid dissociation constants following LDK00.
    Used when opt_k_carbonic = 10.
pk_H2CO3_sws_MM02, pk_HCO3_sws_MM02
    Carbonic acid dissociation constants following MM02.
    Used when opt_k_carbonic = 11.
pk_H2CO3_sws_MPL02, pk_H2CO3_sws_MPL02
    Carbonic acid dissociation constants following MPL02.
    Used when opt_k_carbonic = 12.
pk_H2CO3_sws_MGH06, pk_HCO3_sws_MGH06
    Carbonic acid dissociation constants following MGH06.
    Used when opt_k_carbonic = 13.
pk_H2CO3_sws_M10, pk_HCO3_sws_M10
    Carbonic acid dissociation constants following M10.
    Used when opt_k_carbonic = 14.
pk_H2CO3_sws_WMW14, pk_HCO3_sws_WMW14
    Carbonic acid dissociation constants following WM13/WMW14.
    Used when opt_k_carbonic = 15.
pk_H2CO3_total_SLH20, pk_HCO3_total_SLH20
    Carbonic acid dissociation constants following SLH20.
    Used when opt_k_carbonic = 16.
pk_HCO3_total_SB21
    Second carbonic acid dissociation constant following SB21.
    Used when opt_k_carbonic = 17 together with K1 from WMW14.
pk_H2CO3_total_PLR18, pk_HCO3_total_PLR18
    Carbonic acid dissociation constants following PLR18.
    Used when opt_k_carbonic = 18.
pk_HNO2_total_BBWB24
    Nitrous acid dissociation constant in artificial seawater following BBWB24.
pk_HNO2_nbs_BBWB24_freshwater
    Nitrous acid dissociation constant in freshwater following BBWB24.
"""

from jax import numpy as np

from .. import convert
from ..meta import valid


def coeffs_pk_CO2_W74():
    # The sixth coefficient is temporary, to allow ±pK uncertainties to be set
    # until we know the full uncertainty matrix
    return np.array(
        [-60.2409, 93.4517, 23.3585, 0.023517, -0.023656, 0.0047036, 0.0]
    )


@valid(temperature=[-1, 40], salinity=[0, 40])
def pk_CO2_W74(coeffs_pk_CO2, temperature, salinity):
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
    cf = coeffs_pk_CO2
    TempK100 = (temperature + 273.15) / 100
    lnK0 = (
        cf[0]
        + cf[1] / TempK100
        + cf[2] * np.log(TempK100)
        + salinity * (cf[3] + cf[4] * TempK100 + cf[5] * TempK100**2)
    )
    return cf[6] - lnK0 / np.log(10)


def coeffs_pk_HSO4_free_D90a():
    return np.array(
        [
            -4276.1,
            141.328,
            -23.093,
            -13856,
            324.57,
            -47.986,
            35474,
            -771.54,
            114.723,
            -2698,
            1776,
            0.0,
        ]
    )


@valid(
    temperature=[0, 45],
    salinity=[5, 45],
    ionic_strength=[0.10012312, 0.93904847],
)
def pk_HSO4_free_D90a(coeffs_pk_HSO4, temperature, salinity, ionic_strength):
    """Bisulfate dissociation constant in mol/kg-sw on the free scale following
    D90a.  Used when opt_k_HSO4 = 1.

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
    cf = coeffs_pk_HSO4
    TempK = convert.celsius_to_kelvin(temperature)
    logTempK = np.log(TempK)
    lnk_HSO4 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * logTempK
        + (cf[3] / TempK + cf[4] + cf[5] * logTempK) * np.sqrt(ionic_strength)
        + (cf[6] / TempK + cf[7] + cf[8] * logTempK) * ionic_strength
        + (cf[9] / TempK) * np.sqrt(ionic_strength) * ionic_strength
        + (cf[10] / TempK) * ionic_strength**2
    )
    return cf[11] - np.log10(np.exp(lnk_HSO4) * (1 - 0.001005 * salinity))


def coeffs_pk_HSO4_free_KRCB77():
    return np.array([647.59, -6.3451, 0.019085, -0.5208, 0.0])


@valid(
    temperature=[5, 40],
    salinity=[20, 45],
    ionic_strength=[0.40665374, 0.93904847],
)
def pk_HSO4_free_KRCB77(coeffs_pk_HSO4, temperature, salinity, ionic_strength):
    """Bisulfate dissociation constant in mol/kg-sw on the free scale following
    KRCB77.  Used when opt_k_HSO4 = 2.

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
    cf = coeffs_pk_HSO4
    TempK = temperature + 273.15
    pk_HSO4 = (
        cf[0] / TempK + cf[1] + cf[2] * TempK + cf[3] * np.sqrt(ionic_strength)
    )
    return cf[4] + pk_HSO4 - np.log10(1 - 0.001005 * salinity)


def coeffs_pk_HSO4_free_WM13():
    return np.array(
        [
            562.69486,
            -102.5154,
            -0.0001117033,
            0.2477538,
            -13273.76,
            4.24666,
            -0.152671,
            0.0267059,
            -0.000042128,
            0.2542181,
            -0.00509534,
            0.00071589,
            -0.00291179,
            0.0000209968,
            -0.0000403724,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[5, 45])
def pk_HSO4_free_WM13(coeffs_pk_HSO4, temperature, salinity):
    """Bisulfate dissociation constant in mol/kg-sw on the free scale following
    WM13, with the corrections of WMW14.  Used when opt_k_HSO4 = 3.

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
    cf = coeffs_pk_HSO4
    TempK = convert.celsius_to_kelvin(temperature)
    logKS0 = (
        cf[0]
        + cf[1] * np.log(TempK)
        + cf[2] * TempK**2
        + cf[3] * TempK
        + cf[4] / TempK
    )
    logKSK0 = (
        (
            cf[5]
            + cf[6] * TempK
            + cf[7] * TempK * np.log(TempK)
            + cf[8] * TempK**2
        )
        * salinity**0.5
        + (cf[9] + cf[10] * TempK + cf[11] * TempK * np.log(TempK)) * salinity
        + (cf[12] + cf[13] * TempK) * salinity**1.5
        + cf[14] * salinity**2
    )
    k_HSO4 = (1 - 0.001005 * salinity) * 10.0 ** (logKSK0 + logKS0)
    return cf[15] - np.log10(k_HSO4)


def coeffs_pk_HF_free_DR79():
    return np.array([1590.2, -12.641, 1.525, 0.0])


@valid(
    temperature=[5, 35],
    salinity=[10.43, 47.78],
    ionic_strength=[0.21000866, 0.999987],
)
def pk_HF_free_DR79(coeffs_pk_HF, temperature, salinity, ionic_strength):
    """Hydrogen fluoride dissociation constant on the free scale following
    DR79a.  Used when opt_k_HF = 1.

    Note that the validity range given for this function is given as the ranges
    of temperature and salinity that DR79 applied it to when computing pk_H2O,
    rather than being a true validity range for the pk_HF expression itself.

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
    cf = coeffs_pk_HF
    lnKF = cf[0] / (temperature + 273.15) + cf[1] + cf[2] * ionic_strength**0.5
    return cf[3] - np.log10(np.exp(lnKF) * (1 - 0.001005 * salinity))


def coeffs_pk_HF_free_PF87():
    return np.array([874, -9.68, 0.111, 0.0])


@valid(temperature=[9, 33], salinity=[10, 40])
def pk_HF_free_PF87(coeffs_pk_HF, temperature, salinity):
    """Hydrogen fluoride dissociation constant on the free scale following
    PF87.  Used when opt_k_HF = 2.

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
    # since ill defined for low salinityinity.
    # (to be used for S: 10-40, T: 9-33)
    # Nonetheless, P&F87 might actually be better than the fit of D&R79 above,
    # which is based on only three salinityinities: [0 26.7 34.6]
    # Output is on the free pH scale in mol/kg-SW.
    cf = coeffs_pk_HF
    lnKF = cf[0] / (temperature + 273.15) + cf[1] + cf[2] * salinity**0.5
    return cf[3] - lnKF / np.log(10)


def coeffs_pk_BOH3_total_D90b():
    return np.array(
        [
            -8966.9,
            -2890.53,
            -77.942,
            1.728,
            -0.0996,
            148.0248,
            137.1942,
            1.62142,
            -24.4344,
            -25.085,
            -0.2474,
            0.053105,
            0,
        ]
    )


@valid(temperature=[0, 45], salinity=[5, 45])
def pk_BOH3_total_D90b(coeffs_pk_BOH3, temperature, salinity):
    """Boric acid dissociation constant following D90b.  Used when
    opt_k_BOH3 = 1.

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
    cf = coeffs_pk_BOH3
    sqrsalinity = np.sqrt(salinity)
    TempK = convert.celsius_to_kelvin(temperature)
    lnKBtop = (
        cf[0]
        + cf[1] * sqrsalinity
        + cf[2] * salinity
        + cf[3] * sqrsalinity * salinity
        + cf[4] * salinity**2
    )
    lnKB = (
        lnKBtop / TempK
        + cf[5]
        + cf[6] * sqrsalinity
        + cf[7] * salinity
        + (cf[8] + cf[9] * sqrsalinity + cf[10] * salinity) * np.log(TempK)
        + cf[11] * sqrsalinity * TempK
    )
    return cf[12] - lnKB / np.log(10)


def coeffs_pk_BOH3_nbs_LTB69():
    return np.array([-9.26, 0.00886, 0.01, 0.0])


@valid(temperature=[0, 25], salinity=[29, 38])
def pk_BOH3_nbs_LTB69(coeffs_pk_BOH3, temperature, salinity):
    """Boric acid dissociation constant following LTB69.  Used when
    opt_k_BOH3 = 2.

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
    cf = coeffs_pk_BOH3
    logKB = cf[0] + cf[1] * salinity + cf[2] * temperature
    return cf[3] - logKB


def coeffs_pk_H2O_sws_M95():
    return np.array(
        [
            148.9802,
            -13847.26,
            -23.6521,
            -5.977,
            118.67,
            1.0495,
            -0.01615,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[0, 45])
def pk_H2O_sws_M95(coeffs_pk_H2O, temperature, salinity):
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
    cf = coeffs_pk_H2O
    TempK = convert.celsius_to_kelvin(temperature)
    return cf[7] - (
        cf[0]
        + cf[1] / TempK
        + cf[2] * np.log(TempK)
        + (cf[3] + cf[4] / TempK + cf[5] * np.log(TempK)) * np.sqrt(salinity)
        + cf[6] * salinity
    ) / np.log(10)


def coeffs_pk_H2O_sws_M79():
    return np.array(
        [
            148.9802,
            -13847.26,
            -23.6521,
            -79.2447,
            3298.72,
            12.0408,
            -0.019813,
            0.0,
        ]
    )


@valid(temperature=[0, 50], salinity=[0, 40])
def pk_H2O_sws_M79(coeffs_pk_H2O, temperature, salinity):
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
    cf = coeffs_pk_H2O
    TempK = convert.celsius_to_kelvin(temperature)
    return cf[7] - (
        cf[0]
        + cf[1] / TempK
        + cf[2] * np.log(TempK)
        + (cf[3] + cf[4] / TempK + cf[5] * np.log(TempK)) * np.sqrt(salinity)
        + cf[6] * salinity
    ) / np.log(10)


def coeffs_pk_H2O_sws_HO58_M79():
    return np.array([148.9802, -13847.26, -23.6521, 0.0])


@valid(temperature=[0, 50])
def pk_H2O_sws_HO58_M79(coeffs_pk_H2O, temperature):
    """Water dissociation constant on the seawater scale following HO58 refit
    by M79, for freshwater.  Used when opt_k_H2O = 3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        H2O dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    # refit data of Harned and Owen, The Physical Chemistry of
    # Electrolyte Solutions, 1958
    cf = coeffs_pk_H2O
    TempK = convert.celsius_to_kelvin(temperature)
    return cf[3] - (cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)) / np.log(10)


def coeffs_pk_H3PO4_sws_KP67():
    return np.array([0.02, 0.0])


def pk_H3PO4_sws_KP67(coeffs_pk_H3PO4):
    """First phosphate dissociation constant on the seawater scale following
    KP67.  Used when opt_k_phosphate = 2.

    Returns
    -------
    float
        H3PO4 dissociation constant.
    """
    # === CO2SYS.m comments: =======
    # Peng et al don't include the contribution from the KP1 term,
    # but it is so small it doesn't contribute. It needs to be
    # kept so that the routines work ok.
    cf = coeffs_pk_H3PO4
    return cf[1] - np.log10(cf[0])  # This is already on the seawater scale!


def coeffs_pk_H2PO4_nbs_KP67():
    return np.array([-9.039, -1450, 0.0])


def pk_H2PO4_nbs_KP67(coeffs_pk_H2PO4, temperature):
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
    cf = coeffs_pk_H2PO4
    return cf[2] - (cf[0] + cf[1] / (temperature + 273.15)) / np.log(10)


def coeffs_pk_HPO4_nbs_KP67():
    return np.array([4.466, -7276, 0.0])


def pk_HPO4_nbs_KP67(coeffs_pk_HPO4, temperature):
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
    cf = coeffs_pk_HPO4
    return cf[2] - (cf[0] + cf[1] / (temperature + 273.15)) / np.log(10)


def coeffs_pk_H3PO4_sws_YM95():
    return np.array(
        [
            -4576.752,
            115.54,
            -18.453,
            -106.736,
            +0.69171,
            -0.65643,
            -0.01844,
            0.0,
        ]
    )


def pk_H3PO4_sws_YM95(coeffs_pk_H3PO4, temperature, salinity):
    """First phosphate dissociation constant on the seawater scale following
    YM95.  Used when opt_k_phosphate = 1.

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
    cf = coeffs_pk_H3PO4
    TempK = convert.celsius_to_kelvin(temperature)
    lnKP1 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + (cf[3] / TempK + cf[4]) * np.sqrt(salinity)
        + (cf[5] / TempK + cf[6]) * salinity
    )
    return cf[7] - lnKP1 / np.log(10)


def coeffs_pk_H2PO4_sws_YM95():
    return np.array(
        [
            -8814.715,
            +172.1033,
            -27.927,
            -160.34,
            1.3566,
            0.37335,
            -0.05778,
            0.0,
        ]
    )


def pk_H2PO4_sws_YM95(coeffs_pk_H2PO4, temperature, salinity):
    """Second phosphate dissociation constant on the seawater scale following
    YM95.  Used when opt_k_phosphate = 1.

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
    cf = coeffs_pk_H2PO4
    TempK = convert.celsius_to_kelvin(temperature)
    lnKP2 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + (cf[3] / TempK + cf[4]) * np.sqrt(salinity)
        + (cf[5] / TempK + cf[6]) * salinity
    )
    return cf[7] - lnKP2 / np.log(10)


def coeffs_pk_HPO4_sws_YM95():
    return np.array(
        [
            -3070.75,
            -18.126,
            17.27039,
            2.81197,
            -44.99486,
            -0.09984,
            0.0,
        ]
    )


def pk_HPO4_sws_YM95(coeffs_pk_HPO4, temperature, salinity):
    """Third phosphate dissociation constant on the seawater scale following
    YM95.  Used when opt_k_phosphate = 1.

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
    cf = coeffs_pk_HPO4
    TempK = convert.celsius_to_kelvin(temperature)
    lnKP3 = (
        cf[0] / TempK
        + cf[1]
        + (cf[2] / TempK + cf[3]) * np.sqrt(salinity)
        + (cf[4] / TempK + cf[5]) * salinity
    )
    return cf[6] - lnKP3 / np.log(10)


def coeffs_pk_Si_nbs_SMB64():
    return np.array([0.0000000004, 0.0])


def pk_Si_nbs_SMB64(coeffs_pk_Si):
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
    cf = coeffs_pk_Si
    return cf[1] - np.log10(cf[0])


def coeffs_pk_Si_sws_YM95():
    return np.array(
        [
            -8904.2,
            117.4,
            -19.334,
            -458.79,
            3.5913,
            188.74,
            -1.5998,
            -12.1652,
            +0.07871,
            0.0,
        ]
    )


def pk_Si_sws_YM95(coeffs_pk_Si, temperature, salinity, ionic_strength):
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
    cf = coeffs_pk_Si
    TempK = convert.celsius_to_kelvin(temperature)
    lnKSi = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + (cf[3] / TempK + cf[4]) * np.sqrt(ionic_strength)
        + (cf[5] / TempK + cf[6]) * ionic_strength
        + (cf[7] / TempK + cf[8]) * ionic_strength**2
    )
    return cf[9] - np.log10(np.exp(lnKSi) * (1 - 0.001005 * salinity))


def coeffs_pk_H2CO3_total_RRV93():
    return np.array(
        [
            2.83655,
            -2307.1266,
            -1.5529413,
            -0.20760841,
            -4.0484,
            0.08468345,
            -0.00654208,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[5, 45])
def pk_H2CO3_total_RRV93(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is eq. 29 on p. 254 and what they use in their abstract:
    lnK1 = (
        cf[0]
        + cf[1] / TempK
        + cf[2] * np.log(TempK)
        + (cf[3] + cf[4] / TempK) * np.sqrt(salinity)
        + cf[5] * salinity
        + cf[6] * np.sqrt(salinity) * salinity
    )
    return cf[7] - np.log10(
        np.exp(lnK1)  # this is on the total pH scale in mol/kg-H2O
        * (
            1 - 0.001005 * salinity  # convert to mol/kg-SW
        )
    )


def coeffs_pk_HCO3_total_RRV93():
    return np.array(
        [
            -9.226508,
            -3351.6106,
            -0.2005743,
            -0.106901773,
            -23.9722,
            0.1130822,
            -0.00846934,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[5, 45])
def pk_HCO3_total_RRV93(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is eq. 30 on p. 254 and what they use in their abstract:
    lnK2 = (
        cf[0]
        + cf[1] / TempK
        + cf[2] * np.log(TempK)
        + (cf[3] + cf[4] / TempK) * np.sqrt(salinity)
        + cf[5] * salinity
        + cf[6] * np.sqrt(salinity) * salinity
    )
    return cf[7] - np.log10(
        np.exp(lnK2)  # this is on the total pH scale in mol/kg-H2O
        * (
            1 - 0.001005 * salinity  # convert to mol/kg-SW
        )
    )


def coeffs_pk_H2CO3_sws_GP89():
    return np.array([812.27, 3.356, -0.00171, 0.000091, 0.0])


@valid(temperature=[-1, 40], salinity=[10, 50])
def pk_H2CO3_sws_GP89(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1652 and what they use in the abstract:
    pK1 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * salinity * np.log(TempK)
        + cf[3] * salinity**2
    )
    return cf[4] + pK1  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_HCO3_sws_GP89():
    return np.array([1450.87, 4.604, -0.00385, 0.000182, 0.0])


@valid(temperature=[-1, 40], salinity=[10, 50])
def pk_HCO3_sws_GP89(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1652 and what they use in the abstract:
    pK2 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * salinity * np.log(TempK)
        + cf[3] * salinity**2
    )
    return cf[4] + pK2  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_H2CO3_sws_H73_DM87():
    return np.array([851.4, 3.237, -0.0106, 0.000105, 0.0])


@valid(temperature=[2, 35], salinity=[20, 40])
def pk_H2CO3_sws_H73_DM87(coeffs_pk_H2CO3, temperature, salinity):
    """First carbonic acid dissociation constant following DM87 refit of H73a
    and H73b.  Used when opt_k_carbonic = 3.

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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from Table 4 on p. 1739.
    pK1 = cf[0] / TempK + cf[1] + cf[2] * salinity + cf[3] * salinity**2
    return cf[4] + pK1  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_HCO3_sws_H73_DM87():
    return np.array([-3885.4, 125.844, -18.141, -0.0192, 0.000132, 0.0])


@valid(temperature=[2, 35], salinity=[20, 40])
def pk_HCO3_sws_H73_DM87(coeffs_pk_HCO3, temperature, salinity):
    """Second carbonic acid dissociation constant following DM87 refit of H73a
    and H73b.  Used when opt_k_carbonic = 3.

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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from Table 4 on p. 1739.
    pK2 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + cf[3] * salinity
        + cf[4] * salinity**2
    )
    return cf[5] + pK2  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_H2CO3_sws_MCHP73_DM87():
    return np.array([3670.7, -62.008, 9.7944, -0.0118, 0.000116, 0.0])


@valid(temperature=[2, 35], salinity=[20, 40])
def pk_H2CO3_sws_MCHP73_DM87(coeffs_pk_H2CO3, temperature, salinity):
    """First carbonic acid dissociation constant following DM87 refit of
    MCHP73.  Used when opt_k_carbonic = 4.

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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 4 on p. 1739.
    pK1 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + cf[3] * salinity
        + cf[4] * salinity**2
    )
    return cf[5] + pK1  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_HCO3_sws_MCHP73_DM87():
    return np.array([1394.7, 4.777, -0.0184, 0.000118, 0.0])


@valid(temperature=[2, 35], salinity=[20, 40])
def pk_HCO3_sws_MCHP73_DM87(coeffs_pk_HCO3, temperature, salinity):
    """Second carbonic acid dissociation constant following DM87 refit of
    MCHP73.  Used when opt_k_carbonic = 4.

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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 4 on p. 1739.
    pK2 = cf[0] / TempK + cf[1] + cf[2] * salinity + cf[3] * salinity**2
    return cf[4] + pK2  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_H2CO3_sws_HM_DM87():
    return np.array([845, +3.248, -0.0098, 0.000087, 0.0])


@valid(temperature=[2, 35], salinity=[20, 40])
def pk_H2CO3_sws_HM_DM87(coeffs_pk_H2CO3, temperature, salinity):
    """First carbonic acid dissociation constant following DM87 refit of MCHP73
    plus Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.

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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1740.
    pK1 = cf[0] / TempK + cf[1] + cf[2] * salinity + cf[3] * salinity**2
    return cf[4] + pK1  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_HCO3_sws_HM_DM87():
    return np.array([1377.3, 4.824, -0.0185, 0.000122, 0.0])


@valid(temperature=[2, 35], salinity=[20, 40])
def pk_HCO3_sws_HM_DM87(coeffs_pk_HCO3, temperature, salinity):
    """Second carbonic acid dissociation constant following DM87 refit of
    MCHP73 plus Hansson [H73a, H73b].  Used when opt_k_carbonic = 5.

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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is in Table 5 on p. 1740.
    pK2 = cf[0] / TempK + cf[1] + cf[2] * salinity + cf[3] * salinity**2
    return cf[4] + pK2  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_H2CO3_nbs_MCHP73():
    return np.array([-13.7201, 0.031334, 3235.76, 1.3e-5, -0.1032, 0.0])


@valid(temperature=[2, 35], salinity=[19, 43])
def pk_H2CO3_nbs_MCHP73(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        cf[0]
        + cf[1] * TempK
        + cf[2] / TempK
        + cf[3] * salinity * TempK
        + cf[4] * salinity**0.5
    )
    return cf[5] + pK1  # this is on the NBS scale


def coeffs_pk_HCO3_nbs_MCHP73():
    return np.array(
        [
            5371.9645,
            1.671221,
            0.22913,
            18.3802,
            -128375.28,
            -2194.3055,
            -8.0944e-4,
            -5617.11,
            2.136,
            0.0,
        ]
    )


@valid(temperature=[2, 35], salinity=[19, 43])
def pk_HCO3_nbs_MCHP73(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    salinity = np.where(salinity < 1e-16, 1e-16, salinity)
    # ^ added in v1.8.3, because salinity=0 gives log10(salinity)=-inf
    # pK2 is not defined for salinity=0, since log10(0)=-inf, but since v1.8.3
    # we return the value for salinity=1e-16 instead (this option shouldn't be
    # used in such low salinities anyway; it's only valid above 19!)
    pK2 = (
        cf[0]
        + cf[1] * TempK
        + cf[2] * salinity
        + cf[3] * np.log10(salinity)
        + cf[4] / TempK
        + cf[5] * np.log10(TempK)
        + cf[6] * salinity * TempK
        + cf[7] * np.log10(salinity) / TempK
        + cf[8] * salinity / TempK
    )
    return cf[9] + pK2  # this is on the NBS scale


def coeffs_pk_H2CO3_sws_M79():
    return np.array([290.9097, -14554.21, -45.0575, 0.0])


@valid(temperature=[0, 50])
def pk_H2CO3_sws_M79(coeffs_pk_H2CO3, temperature):
    """First carbonic acid dissociation constant following M79, pure water
    case.  Used when opt_k_carbonic = 8.

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
    # This is only to be used for salinity=0 water (note the absence of S in
    # the below formulations).
    # These are the thermodynamic Constants:
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    lnK1 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    return cf[3] - lnK1 / np.log(10)


def coeffs_pk_HCO3_sws_M79():
    return np.array([207.6548, -11843.79, -33.6485, 0.0])


@valid(temperature=[0, 50])
def pk_HCO3_sws_M79(coeffs_pk_HCO3, temperature):
    """Second carbonic acid dissociation constant following M79, pure water
    case.  Used when opt_k_carbonic = 8.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        HCO3 dissociation constant.
    """
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    lnK2 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    return cf[3] - lnK2 / np.log(10)


def coeffs_pk_H2CO3_nbs_CW98():
    return np.array(
        [
            200.1,
            0.3220,
            3404.71,
            0.032786,
            -14.8435,
            -0.071692,
            0.0021487,
            0.0,
        ]
    )


@valid(temperature=[0.2, 30], salinity=[0, 40])
def pk_H2CO3_nbs_CW98(coeffs_pk_H2CO3, temperature, salinity):
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
    # Conversion to SWS scale by division by fH is uncertain at low salinity
    # due to junction potential.
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    F1 = cf[0] / TempK + cf[1]
    pK1 = (
        cf[2] / TempK
        + cf[3] * TempK
        + cf[4]
        + cf[5] * F1 * salinity**0.5
        + cf[6] * salinity
    )
    return cf[7] + pK1  # this is on the NBS scale


def coeffs_pk_HCO3_nbs_CW98():
    return np.array(
        [
            -129.24,
            1.4381,
            2902.39,
            0.02379,
            -6.4980,
            -0.3191,
            0.0198,
            0.0,
        ]
    )


@valid(temperature=[0.2, 30], salinity=[0, 40])
def pk_HCO3_nbs_CW98(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    F2 = cf[0] / TempK + cf[1]
    pK2 = (
        cf[2] / TempK
        + cf[3] * TempK
        + cf[4]
        + cf[5] * F2 * salinity**0.5
        + cf[6] * salinity
    )
    return cf[7] + pK2  # this is on the NBS scale


def coeffs_pk_H2CO3_total_LDK00():
    return np.array([3633.86, -61.2172, 9.6777, -0.011555, 0.0001152, 0.0])


@valid(temperature=[2, 35], salinity=[19, 43])
def pk_H2CO3_total_LDK00(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + cf[3] * salinity
        + cf[4] * salinity**2
    )
    return cf[5] + pK1  # this is on the Total pH scale in mol/kg-SW


def coeffs_pk_HCO3_total_LDK00():
    return np.array([471.78, 25.929, -3.16967, -0.01781, 0.0001122, 0.0])


@valid(temperature=[2, 35], salinity=[19, 43])
def pk_HCO3_total_LDK00(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + cf[3] * salinity
        + cf[4] * salinity**2
    )
    return cf[5] + pK2  # this is on the Total pH scale in mol/kg-SW


def coeffs_pk_H2CO3_sws_MM02():
    return np.array([-43.6977, -0.0129037, 1.364e-4, 2885.378, 7.045159, 0.0])


@valid(temperature=[0, 45], salinity=[5, 42])
def pk_H2CO3_sws_MM02(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        cf[0]
        + cf[1] * salinity
        + cf[2] * salinity**2
        + cf[3] / TempK
        + cf[4] * np.log(TempK)
    )
    return cf[5] + pK1  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_HCO3_sws_MM02():
    return np.array(
        [
            -452.0940,
            13.142162,
            -8.101e-4,
            21263.61,
            68.483143,
            -581.4428,
            0.259601,
            -1.967035,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[5, 42])
def pk_HCO3_sws_MM02(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        cf[0]
        + cf[1] * salinity
        + cf[2] * salinity**2
        + cf[3] / TempK
        + cf[4] * np.log(TempK)
        + (cf[5] * salinity + cf[6] * salinity**2) / TempK
        + cf[7] * salinity * np.log(TempK)
    )
    return cf[8] + pK2  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_H2CO3_sws_MPL02():
    return np.array([6.359, -0.00664, -0.01322, 4.989e-5, 0.0])


@valid(temperature=[-1.6, 35], salinity=[34, 37])
def pk_H2CO3_sws_MPL02(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    pK1 = (
        cf[0] + cf[1] * salinity + cf[2] * temperature + cf[3] * temperature**2
    )
    return cf[4] + pK1  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_HCO3_sws_MPL02():
    return np.array([9.867, -0.01314, -0.01904, 2.448e-5, 0.0])


@valid(temperature=[-1.6, 35], salinity=[34, 37])
def pk_HCO3_sws_MPL02(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    pK2 = (
        cf[0] + cf[1] * salinity + cf[2] * temperature + cf[3] * temperature**2
    )
    return cf[4] + pK2  # this is on the SWS pH scale in mol/kg-SW


def coeffs_pk_H2CO3_sws_MGH06():
    return np.array(
        [
            -126.34048,
            6320.813,
            19.568224,
            13.4191,
            0.0331,
            -5.33e-5,
            -530.123,
            -6.103,
            -2.06950,
            0.0,
        ]
    )


@valid(temperature=[0, 50], salinity=[1, 50])
def pk_H2CO3_sws_MGH06(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK1_0 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    A_1 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B_1 = cf[6] * salinity**0.5 + cf[7] * salinity
    C_1 = cf[8] * salinity**0.5
    pK1 = A_1 + B_1 / TempK + C_1 * np.log(TempK) + pK1_0  # pK1 sigma = 0.0054
    return cf[9] + pK1


def coeffs_pk_HCO3_sws_MGH06():
    return np.array(
        [
            -90.18333,
            5143.692,
            14.613358,
            21.0894,
            0.1248,
            -3.687e-4,
            -772.483,
            -20.051,
            -3.3336,
            0.0,
        ]
    )


@valid(temperature=[0, 50], salinity=[1, 50])
def pk_HCO3_sws_MGH06(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK2_0 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    A_2 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B_2 = cf[6] * salinity**0.5 + cf[7] * salinity
    C_2 = cf[8] * salinity**0.5
    pK2 = A_2 + B_2 / TempK + C_2 * np.log(TempK) + pK2_0  # pK2 sigma = 0.011
    return cf[9] + pK2


def coeffs_pk_H2CO3_sws_M10():
    return np.array(
        [
            -126.34048,
            6320.813,
            19.568224,
            13.4038,
            0.03206,
            -5.242e-5,
            -530.659,
            -5.8210,
            -2.0664,
            0.0,
        ]
    )


@valid(temperature=[0, 50], salinity=[1, 50])
def pk_H2CO3_sws_M10(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from page 141
    pK10 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    # This is from their table 2, page 140.
    A1 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B1 = cf[6] * salinity**0.5 + cf[7] * salinity
    C1 = cf[8] * salinity**0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    return cf[9] + pK1


def coeffs_pk_HCO3_sws_M10():
    return np.array(
        [
            -90.18333,
            5143.692,
            14.613358,
            21.3728,
            0.1218,
            -3.688e-4,
            -788.289,
            -19.189,
            -3.374,
            0.0,
        ]
    )


@valid(temperature=[0, 50], salinity=[1, 50])
def pk_HCO3_sws_M10(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    # This is from page 141
    pK20 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    # This is from their table 3, page 140.
    A2 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B2 = cf[6] * salinity**0.5 + cf[7] * salinity
    C2 = cf[8] * salinity**0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    return cf[9] + pK2


def coeffs_pk_H2CO3_sws_WMW14():
    return np.array(
        [
            -126.34048,
            6320.813,
            19.568224,
            13.409160,
            0.031646,
            -5.1895e-5,
            -531.3642,
            -5.713,
            -2.0669166,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[0, 45])
def pk_H2CO3_sws_WMW14(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK10 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    A1 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B1 = cf[6] * salinity**0.5 + cf[7] * salinity
    C1 = cf[8] * salinity**0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    return cf[9] + pK1


def coeffs_pk_HCO3_sws_WMW14():
    return np.array(
        [
            -90.18333,
            5143.692,
            14.613358,
            21.225890,
            0.12450870,
            -3.7243e-4,
            -779.3444,
            -19.91739,
            -3.3534679,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[0, 45])
def pk_HCO3_sws_WMW14(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK20 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    A2 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B2 = cf[6] * salinity**0.5 + cf[7] * salinity
    C2 = cf[8] * salinity**0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    return cf[9] + pK2


def coeffs_pk_H2CO3_total_WMW14():
    return np.array(
        [
            -126.34048,
            6320.813,
            19.568224,
            13.568513,
            +0.031645,
            -5.3834e-5,
            -539.2304,
            -5.635,
            -2.0901396,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[0, 45])
def pk_H2CO3_total_WMW14(coeffs_pk_H2CO3, temperature, salinity):
    """First carbonic acid dissociation constant following WM13/WMW14.
    Used when opt_k_carbonic = 17.

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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    # Coefficients from the corrigendum document [WMW14]
    pK10 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    A1 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B1 = cf[6] * salinity**0.5 + cf[7] * salinity
    C1 = cf[8] * salinity**0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    return cf[9] + pK1


def coeffs_pk_HCO3_total_WMW14():
    return np.array(
        [
            -90.18333,
            5143.692,
            14.613358,
            21.389248,
            0.12452358,
            -3.7447e-4,
            -787.3736,
            -19.84233,
            -3.3773006,
            0.0,
        ]
    )


@valid(temperature=[0, 45], salinity=[0, 45])
def pk_HCO3_total_WMW14(coeffs_pk_HCO3, temperature, salinity):
    """Second carbonic acid dissociation constant following WM13/WMW14.

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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    # Coefficients from the corrigendum document [WMW14]
    pK20 = cf[0] + cf[1] / TempK + cf[2] * np.log(TempK)
    A2 = cf[3] * salinity**0.5 + cf[4] * salinity + cf[5] * salinity**2
    B2 = cf[6] * salinity**0.5 + cf[7] * salinity
    C2 = cf[8] * salinity**0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    return cf[9] + pK2


def coeffs_pk_H2CO3_total_SLH20():
    return np.array(
        [
            8510.63,  # ±1139.8
            -172.4493,  # ±26.131
            26.32996,  # ±3.9161
            -0.011555,
            0.0001152,
            0.0,
        ]
    )


@valid(temperature=[-1.67, 31.8], salinity=[30.73, 37.57])
def pk_H2CO3_total_SLH20(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        cf[0] / TempK
        + cf[1]
        + cf[2] * np.log(TempK)
        + cf[3] * salinity
        + cf[4] * salinity**2
    )
    return cf[5] + pK1  # this is on the Total pH scale in mol/kg-SW


def coeffs_pk_HCO3_total_SLH20():
    return np.array([4226.23, -59.4636, 9.60817, -0.01781, 0.0001122, 0.0])


@valid(temperature=[-1.67, 31.8], salinity=[30.73, 37.57])
def pk_HCO3_total_SLH20(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        cf[0] / TempK  # ±1050.8
        + cf[1]  # ±24.016
        + cf[2] * np.log(TempK)  # ±3.5966
        + cf[3] * salinity
        + cf[4] * salinity**2
    )
    return cf[5] + pK2  # this is on the Total pH scale in mol/kg-SW


def coeffs_pk_HCO3_total_SB21():
    return np.array(
        [
            116.8067,
            -3655.02,
            -16.45817,
            0.04523,
            -0.615,
            -0.0002799,
            4.969,
            0.0,
        ]
    )


@valid(temperature=[15, 35], salinity=[19.6, 41])
def pk_HCO3_total_SB21(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        cf[0]
        + cf[1] / TempK
        + cf[2] * np.log(TempK)
        + cf[3] * salinity
        + cf[4] * np.sqrt(salinity)
        + cf[5] * salinity**2
        + cf[6] * salinity / TempK
    )
    return cf[7] + pK2


def coeffs_pk_H2CO3_total_PLR18():
    return np.array(
        [
            -176.48,
            6.14528,
            -0.127714,
            7.396e-5,
            9914.37,
            -622.886,
            29.714,
            26.05129,
            -0.666812,
            0.0,
        ]
    )


@valid(temperature=[-6, 25], salinity=[33, 100])
def pk_H2CO3_total_PLR18(coeffs_pk_H2CO3, temperature, salinity):
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
    cf = coeffs_pk_H2CO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK1 = (
        cf[0]
        + cf[1] * salinity**0.5
        + cf[2] * salinity
        + cf[3] * salinity**2
        + (cf[4] + cf[5] * salinity**0.5 + cf[6] * salinity) / TempK
        + (cf[7] + cf[8] * salinity**0.5) * np.log(TempK)
    )
    return cf[9] + pK1


def coeffs_pk_HCO3_total_PLR18():
    return np.array(
        [
            -323.52692,
            27.557655,
            0.154922,
            -2.48396e-4,
            14763.287,
            -1014.819,
            -14.35223,
            50.385807,
            -4.4630415,
            0.0,
        ]
    )


@valid(temperature=[-6, 25], salinity=[33, 100])
def pk_HCO3_total_PLR18(coeffs_pk_HCO3, temperature, salinity):
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
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    pK2 = (
        cf[0]
        + cf[1] * salinity**0.5
        + cf[2] * salinity
        + cf[3] * salinity**2
        + (cf[4] + cf[5] * salinity**0.5 + cf[6] * salinity) / TempK
        + (cf[7] + cf[8] * salinity**0.5) * np.log(TempK)
    )
    return cf[9] + pK2


def coeffs_pk_HCO3_total_MMB25():
    return np.array(
        [
            5.1703,
            2136.77,
            -177788,
            -0.4457,
            +0.0674,
            -0.0008238,
            0.0,
        ]
    )


@valid(temperature=[0, 35], salinity=[0, 41])
def pk_HCO3_total_MMB25(coeffs_pk_HCO3, temperature, salinity):
    """Carbonic acid dissociation constants with K2 following MMB25.
    K1 should come from WMW14.
    Used when opt_k_carbonic = 19.
    """
    cf = coeffs_pk_HCO3
    TempK = convert.celsius_to_kelvin(temperature)
    Sal = salinity
    pK2 = (
        cf[0]
        + cf[1] / TempK
        + cf[2] / TempK**2
        + cf[3] * np.sqrt(Sal) / (1 + 1.11 * np.sqrt(Sal))
        + cf[4] * Sal / np.log(TempK)
        + cf[5] * np.sqrt(Sal) * TempK
    )
    return cf[6] + pK2


def coeffs_pk_H2S_total_YM95():
    return np.array([225.838, -13275.3, -34.6435, 0.3449, -0.0274, 0.0])


def pk_H2S_total_YM95(coeffs_pk_H2S, temperature, salinity):
    """Hydrogen sulfide dissociation constant on the total scale following
    YM95.

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
    cf = coeffs_pk_H2S
    TempK = convert.celsius_to_kelvin(temperature)
    lnkH2S = (
        cf[0]
        + cf[1] / TempK
        + cf[2] * np.log(TempK)
        + cf[3] * np.sqrt(salinity)
        + cf[4] * salinity
    )
    return cf[5] - lnkH2S / np.log(10)


def coeffs_pk_NH3_sws_YM95():
    return np.array(
        [
            -6285.33,
            +0.0001635,
            -0.25444,
            0.46532,
            -123.7184,
            -0.01992,
            3.17556,
            0.0,
        ]
    )


def pk_NH3_sws_YM95(coeffs_pk_NH3, temperature, salinity):
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
    cf = coeffs_pk_NH3
    TempK = convert.celsius_to_kelvin(temperature)
    lnkNH3 = (
        cf[0] / TempK
        + cf[1] * TempK
        - cf[2]
        + (cf[3] + cf[4] / TempK) * np.sqrt(salinity)
        + (cf[5] + cf[6] / TempK) * salinity
    )
    return cf[7] - lnkNH3 / np.log(10)


def coeffs_pk_NH3_total_CW95():
    return np.array(
        [
            9.244605,
            -2729.33,
            0.04203362,
            -11.24742,
            -13.6416,
            1.176949,
            -0.02860785,
            545.4834,
            -0.1462507,
            0.0090226468,
            -0.0001471361,
            10.5425,
            0.004669309,
            -0.0001691742,
            -0.5677934,
            -2.354039e-05,
            0.009698623,
            0.0,
        ]
    )


@valid(temperature=[-2, 40], salinity=[0, 40])
def pk_NH3_total_CW95(coeffs_pk_NH3, temperature, salinity):
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
    cf = coeffs_pk_NH3
    TempK = convert.celsius_to_kelvin(temperature)
    PKNH3expCW = cf[0] + cf[1] * (1 / 298.15 - 1 / TempK)
    PKNH3expCW = PKNH3expCW + (cf[2] + cf[3] / TempK) * salinity**0.25
    PKNH3expCW = (
        PKNH3expCW
        + (cf[4] + cf[5] * TempK**0.5 + cf[6] * TempK + cf[7] / TempK)
        * salinity**0.5
    )
    PKNH3expCW = (
        PKNH3expCW
        + (cf[8] + cf[9] * TempK**0.5 + cf[10] * TempK + cf[11] / TempK)
        * salinity**1.5
    )
    PKNH3expCW = (
        PKNH3expCW
        + (cf[12] + cf[13] * TempK**0.5 + cf[14] / TempK) * salinity**2
    )
    PKNH3expCW = PKNH3expCW + (cf[15] + cf[16] / TempK) * salinity**2.5
    KNH3 = 10.0**-PKNH3expCW  # this is on the total pH scale in mol/kg-H2O
    KNH3 = KNH3 * (1 - 0.001005 * salinity)  # convert to mol/kg-SW
    return cf[17] - np.log10(KNH3)


def coeffs_pk_HNO2_total_BBWB24():
    return np.array([16084.01, 50.17, -336.92, 0.0])


@valid(temperature=[5, 35])
def pk_HNO2_total_BBWB24(coeffs_pk_HNO2, temperature):
    """Nitrous acid dissociation constant in artificial seawater following
    BBWB24.

    Used when opt_k_HNO2 = 1 (default).  Valid from 5 to 35 °C.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        HNO2 dissociation constant.
    """
    cf = coeffs_pk_HNO2
    T = convert.celsius_to_kelvin(temperature)
    pk_HNO2 = cf[0] / T + cf[1] * np.log(T) + cf[2]
    return cf[3] + pk_HNO2


def coeffs_pk_HNO2_nbs_BBWB24_freshwater():
    return np.array([16437.31, 53.61, -357.43, 0.0])


@valid(temperature=[5, 35])
def pk_HNO2_nbs_BBWB24_freshwater(coeffs_pk_HNO2, temperature):
    """Nitrous acid dissociation constant in freshwater following BBWB24.

    Used when opt_k_HNO2 = 2.  Valid from 5 to 35 °C.

    Parameters
    ----------
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        HNO2 dissociation constant.
    """
    cf = coeffs_pk_HNO2
    T = convert.celsius_to_kelvin(temperature)
    pk_HNO2 = cf[0] / T + cf[1] * np.log(T) + cf[2]
    return cf[3] + pk_HNO2
