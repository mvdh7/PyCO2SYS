# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Estimate stoichiometric equilibrium constants at atmospheric pressure."""

from autograd import numpy as np
from .. import salts


def kCO2_W74(TempK, Sal):
    """Henry's constant for CO2 solubility in mol/kg-sw/atm following W74."""
    # === CO2SYS.m comments: =======
    # Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    # This is in mol/kg-SW/atm.
    TempK100 = TempK / 100
    lnK0 = (
        -60.2409
        + 93.4517 / TempK100
        + 23.3585 * np.log(TempK100)
        + Sal * (0.023517 - 0.023656 * TempK100 + 0.0047036 * TempK100 ** 2)
    )
    return np.exp(lnK0)


def kHSO4_FREE_D90a(TempK, Sal):
    """Bisulfate dissociation constant following D90a."""
    # === CO2SYS.m comments: =======
    # Dickson, A. G., J. Chemical Thermodynamics, 22:113-127, 1990
    # The goodness of fit is .021.
    # It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    # TYPO on p. 121: the constant e9 should be e8.
    # Output KS is on the free pH scale in mol/kg-sw.
    # This is from eqs 22 and 23 on p. 123, and Table 4 on p 121:
    logTempK = np.log(TempK)
    IonS = salts.ionstr_DOE94(Sal)
    lnKSO4 = (
        -4276.1 / TempK
        + 141.328
        - 23.093 * logTempK
        + (-13856 / TempK + 324.57 - 47.986 * logTempK) * np.sqrt(IonS)
        + (35474 / TempK - 771.54 + 114.723 * logTempK) * IonS
        + (-2698 / TempK) * np.sqrt(IonS) * IonS
        + (1776 / TempK) * IonS ** 2
    )
    return np.exp(lnKSO4) * (1 - 0.001005 * Sal)


def kHSO4_FREE_KRCB77(TempK, Sal):
    """Bisulfate dissociation constant following KRCB77."""
    # === CO2SYS.m comments: =======
    # Khoo, et al, Analytical Chemistry, 49(1):29-34, 1977
    # KS was found by titrations with a hydrogen electrode
    # of artificial seawater containing sulfate (but without F)
    # at 3 Salinities from 20 to 45 and artificial seawater NOT
    # containing sulfate (nor F) at 16 Salinities from 15 to 45,
    # both at temperatures from 5 to 40 deg C.
    # KS is on the Free pH scale (inherently so).
    # It was given in mol/kg-H2O. I convert it to mol/kg-SW.
    # He finds log(beta) which = my pKS;
    # his beta is an association constant.
    # The rms error is .0021 in pKS, or about .5% in KS.
    # This is equation 20 on p. 33:
    # Output KS is on the free pH scale in mol/kg-sw.
    IonS = salts.ionstr_DOE94(Sal)
    pKSO4 = 647.59 / TempK - 6.3451 + 0.019085 * TempK - 0.5208 * np.sqrt(IonS)
    return 10.0 ** -pKSO4 * (1 - 0.001005 * Sal)


def kHSO4_FREE_WM13(TempK, Sal):
    """Bisulfate dissociation constant following WM13/WMW14."""
    logKS0 = (
        562.69486
        - 102.5154 * np.log(TempK)
        - 0.0001117033 * TempK ** 2
        + 0.2477538 * TempK
        - 13273.76 / TempK
    )
    logKSK0 = (
        (
            4.24666
            - 0.152671 * TempK
            + 0.0267059 * TempK * np.log(TempK)
            - 0.000042128 * TempK ** 2
        )
        * Sal ** 0.5
        + (0.2542181 - 0.00509534 * TempK + 0.00071589 * TempK * np.log(TempK)) * Sal
        + (-0.00291179 + 0.0000209968 * TempK) * Sal ** 1.5
        + -0.0000403724 * Sal ** 2
    )
    kSO4 = (1 - 0.001005 * Sal) * 10.0 ** (logKSK0 + logKS0)
    return kSO4


def kHF_FREE_DR79(TempK, Sal):
    """Hydrogen fluoride dissociation constant following DR79."""
    # === CO2SYS.m comments: =======
    # Dickson, A. G. and Riley, J. P., Marine Chemistry 7:89-99, 1979:
    # this is on the free pH scale in mol/kg-sw
    IonS = salts.ionstr_DOE94(Sal)
    lnKF = 1590.2 / TempK - 12.641 + 1.525 * IonS ** 0.5
    return np.exp(lnKF) * (1 - 0.001005 * Sal)


def kHF_FREE_PF87(TempK, Sal):
    """Hydrogen fluoride dissociation constant following PF87."""
    # Note that this is not currently used or an option in CO2SYS,
    # despite the equations below appearing in CO2SYS.m (commented out).
    # === CO2SYS.m comments: =======
    # Another expression exists for KF: Perez and Fraga 1987. Not used here
    # since ill defined for low Salinity. (to be used for S: 10-40, T: 9-33)
    # Nonetheless, P&F87 might actually be better than the fit of D&R79 above,
    # which is based on only three Salinities: [0 26.7 34.6]
    # Output is on the free pH scale in mol/kg-SW.
    lnKF = 874 / TempK - 9.68 + 0.111 * Sal ** 0.5
    return np.exp(lnKF)


def kBOH3_NBS_LTB69(TempK, Sal):
    """Boric acid dissociation constant following LTB69."""
    # === CO2SYS.m comments: =======
    # This is for GEOSECS and Peng et al.
    # Lyman, John, UCLA Thesis, 1957
    # fit by Li et al, JGR 74:5507-5525, 1969.
    # logKB is on NBS pH scale
    TempC = TempK - 273.15
    logKB = -9.26 + 0.00886 * Sal + 0.01 * TempC
    return 10.0 ** logKB


def kBOH3_TOT_D90b(TempK, Sal):
    """Boric acid dissociation constant following D90b."""
    # === CO2SYS.m comments: =======
    # Dickson, A. G., Deep-Sea Research 37:755-766, 1990.
    # lnKB is on Total pH scale
    sqrSal = np.sqrt(Sal)
    lnKBtop = (
        -8966.9
        - 2890.53 * sqrSal
        - 77.942 * Sal
        + 1.728 * sqrSal * Sal
        - 0.0996 * Sal ** 2
    )
    lnKB = (
        lnKBtop / TempK
        + 148.0248
        + 137.1942 * sqrSal
        + 1.62142 * Sal
        + (-24.4344 - 25.085 * sqrSal - 0.2474 * Sal) * np.log(TempK)
        + 0.053105 * sqrSal * TempK
    )
    return np.exp(lnKB)


def kH2O_SWS_M79(TempK, Sal):
    """Water dissociation constant following M79."""
    # === CO2SYS.m comments: =======
    # Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    return np.exp(
        148.9802
        - 13847.26 / TempK
        - 23.6521 * np.log(TempK)
        + (-79.2447 + 3298.72 / TempK + 12.0408 * np.log(TempK)) * np.sqrt(Sal)
        - 0.019813 * Sal
    )


def kH2O_SWS_HO58_M79(TempK, Sal):
    """Water dissociation constant following HO58 refit by M79."""
    # === CO2SYS.m comments: =======
    # Millero, Geochemica et Cosmochemica Acta 43:1651-1661, 1979
    # refit data of Harned and Owen, The Physical Chemistry of
    # Electrolyte Solutions, 1958
    return np.exp(148.9802 - 13847.26 / TempK - 23.6521 * np.log(TempK))


def kH2O_SWS_M95(TempK, Sal):
    """Water dissociation constant following M95."""
    # === CO2SYS.m comments: =======
    # Millero, Geochemica et Cosmochemica Acta 59:661-677, 1995.
    # his check value of 1.6 umol/kg-SW should be 6.2 (for ln(k))
    return np.exp(
        148.9802
        - 13847.26 / TempK
        - 23.6521 * np.log(TempK)
        + (-5.977 + 118.67 / TempK + 1.0495 * np.log(TempK)) * np.sqrt(Sal)
        - 0.01615 * Sal
    )


def kH3PO4_NBS_KP67(TempK, Sal):
    """Phosphate dissociation constants following KP67."""
    # === CO2SYS.m comments: =======
    # Peng et al don't include the contribution from the KP1 term,
    # but it is so small it doesn't contribute. It needs to be
    # kept so that the routines work ok.
    # KP2, KP3 from Kester, D. R., and Pytkowicz, R. M.,
    # Limnology and Oceanography 12:243-252, 1967:
    # these are only for sals 33 to 36 and are on the NBS scale.
    KP1 = 0.02  # This is already on the seawater scale!
    KP2 = np.exp(-9.039 - 1450 / TempK)
    KP3 = np.exp(4.466 - 7276 / TempK)
    return KP1, KP2, KP3


def kSi_NBS_SMB64(TempK, Sal):
    """Silicate dissociation constant following SMB64."""
    # === CO2SYS.m comments: =======
    # Sillen, Martell, and Bjerrum,  Stability Constants of metal-ion
    # complexes, The Chemical Society (London), Special Publ. 17:751, 1964.
    return 0.0000000004


def kH3PO4_SWS_YM95(TempK, Sal):
    """Phosphate dissociation constants following YM95."""
    # === CO2SYS.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
    # KP1, KP2, KP3 are on the SWS pH scale in mol/kg-SW.
    lnKP1 = (
        -4576.752 / TempK
        + 115.54
        - 18.453 * np.log(TempK)
        + (-106.736 / TempK + 0.69171) * np.sqrt(Sal)
        + (-0.65643 / TempK - 0.01844) * Sal
    )
    KP1 = np.exp(lnKP1)
    lnKP2 = (
        -8814.715 / TempK
        + 172.1033
        - 27.927 * np.log(TempK)
        + (-160.34 / TempK + 1.3566) * np.sqrt(Sal)
        + (0.37335 / TempK - 0.05778) * Sal
    )
    KP2 = np.exp(lnKP2)
    lnKP3 = (
        -3070.75 / TempK
        - 18.126
        + (17.27039 / TempK + 2.81197) * np.sqrt(Sal)
        + (-44.99486 / TempK - 0.09984) * Sal
    )
    KP3 = np.exp(lnKP3)
    return KP1, KP2, KP3


def kSi_SWS_YM95(TempK, Sal):
    """Silicate dissociation constant following YM95."""
    # === CO2SYS.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995
    # KSi was given on the SWS pH scale in mol/kg-H2O, but is converted here
    # to mol/kg-sw.
    IonS = salts.ionstr_DOE94(Sal)
    lnKSi = (
        -8904.2 / TempK
        + 117.4
        - 19.334 * np.log(TempK)
        + (-458.79 / TempK + 3.5913) * np.sqrt(IonS)
        + (188.74 / TempK - 1.5998) * IonS
        + (-12.1652 / TempK + 0.07871) * IonS ** 2
    )
    return np.exp(lnKSi) * (1 - 0.001005 * Sal)


def kH2CO3_TOT_RRV93(TempK, Sal):
    """Carbonic acid dissociation constants following RRV93."""
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
    # This is eq. 29 on p. 254 and what they use in their abstract:
    lnK1 = (
        2.83655
        - 2307.1266 / TempK
        - 1.5529413 * np.log(TempK)
        + (-0.20760841 - 4.0484 / TempK) * np.sqrt(Sal)
        + 0.08468345 * Sal
        - 0.00654208 * np.sqrt(Sal) * Sal
    )
    K1 = np.exp(lnK1) * (  # this is on the total pH scale in mol/kg-H2O
        1 - 0.001005 * Sal
    )  # convert to mol/kg-SW
    # This is eq. 30 on p. 254 and what they use in their abstract:
    lnK2 = (
        -9.226508
        - 3351.6106 / TempK
        - 0.2005743 * np.log(TempK)
        + (-0.106901773 - 23.9722 / TempK) * np.sqrt(Sal)
        + 0.1130822 * Sal
        - 0.00846934 * np.sqrt(Sal) * Sal
    )
    K2 = np.exp(lnK2) * (  # this is on the total pH scale in mol/kg-H2O
        1 - 0.001005 * Sal
    )  # convert to mol/kg-SW
    return K1, K2


def kH2CO3_SWS_GP89(TempK, Sal):
    """Carbonic acid dissociation constants following GP89."""
    # === CO2SYS.m comments: =======
    # GOYET AND POISSON, Deep-Sea Research, 36(11):1635-1654, 1989
    # The 2s precision in pK1 is .011, or 2.5% in K1.
    # The 2s precision in pK2 is .02, or 4.5% in K2.
    # This is in Table 5 on p. 1652 and what they use in the abstract:
    pK1 = 812.27 / TempK + 3.356 - 0.00171 * Sal * np.log(TempK) + 0.000091 * Sal ** 2
    K1 = 10.0 ** -pK1  # this is on the SWS pH scale in mol/kg-SW
    # This is in Table 5 on p. 1652 and what they use in the abstract:
    pK2 = 1450.87 / TempK + 4.604 - 0.00385 * Sal * np.log(TempK) + 0.000182 * Sal ** 2
    K2 = 10.0 ** -pK2  # this is on the SWS pH scale in mol/kg-SW
    return K1, K2


def kH2CO3_SWS_H73_DM87(TempK, Sal):
    """Carbonic acid dissociation constants following DM87 refit of H73a and H73b."""
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
    # This is from Table 4 on p. 1739.
    pK1 = 851.4 / TempK + 3.237 - 0.0106 * Sal + 0.000105 * Sal ** 2
    K1 = 10.0 ** -pK1  # this is on the SWS pH scale in mol/kg-SW
    # This is from Table 4 on p. 1739.
    pK2 = (
        -3885.4 / TempK
        + 125.844
        - 18.141 * np.log(TempK)
        - 0.0192 * Sal
        + 0.000132 * Sal ** 2
    )
    K2 = 10.0 ** -pK2  # this is on the SWS pH scale in mol/kg-SW
    return K1, K2


def kH2CO3_SWS_MCHP73_DM87(TempK, Sal):
    """Carbonic acid dissociation constants following DM87 refit of MCHP73."""
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
    # This is in Table 4 on p. 1739.
    pK1 = (
        3670.7 / TempK
        - 62.008
        + 9.7944 * np.log(TempK)
        - 0.0118 * Sal
        + 0.000116 * Sal ** 2
    )
    K1 = 10.0 ** -pK1  # this is on the SWS pH scale in mol/kg-SW
    # This is in Table 4 on p. 1739.
    pK2 = 1394.7 / TempK + 4.777 - 0.0184 * Sal + 0.000118 * Sal ** 2
    K2 = 10.0 ** -pK2  # this is on the SWS pH scale in mol/kg-SW
    return K1, K2


def kH2CO3_SWS_HM_DM87(TempK, Sal):
    """Carbonic acid dissociation constants following DM87 refit of MCHP73 plus studies
    by Hansson [H73a, H73b].
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
    # This is in Table 5 on p. 1740.
    pK1 = 845 / TempK + 3.248 - 0.0098 * Sal + 0.000087 * Sal ** 2
    K1 = 10.0 ** -pK1  # this is on the SWS pH scale in mol/kg-SW
    # This is in Table 5 on p. 1740.
    pK2 = 1377.3 / TempK + 4.824 - 0.0185 * Sal + 0.000122 * Sal ** 2
    K2 = 10.0 ** -pK2  # this is on the SWS pH scale in mol/kg-SW
    return K1, K2


@np.errstate(divide="ignore", invalid="ignore")  # because Sal=0 gives log10(Sal)=-inf
def kH2CO3_NBS_MCHP73(TempK, Sal):
    """Carbonic acid dissociation constants following MCHP73."""
    # === CO2SYS.m comments: =======
    # GEOSECS and Peng et al use K1, K2 from Mehrbach et al,
    # Limnology and Oceanography, 18(6):897-907, 1973.
    # I.e., these are the original Mehrbach dissociation constants.
    # The 2s precision in pK1 is .005, or 1.2% in K1.
    # The 2s precision in pK2 is .008, or 2% in K2.
    pK1 = (
        -13.7201
        + 0.031334 * TempK
        + 3235.76 / TempK
        + 1.3e-5 * Sal * TempK
        - 0.1032 * Sal ** 0.5
    )
    K1 = 10.0 ** -pK1  # this is on the NBS scale
    pK2 = (
        5371.9645
        + 1.671221 * TempK
        + 0.22913 * Sal
        + 18.3802 * np.log10(Sal)
        - 128375.28 / TempK
        - 2194.3055 * np.log10(TempK)
        - 8.0944e-4 * Sal * TempK
        - 5617.11 * np.log10(Sal) / TempK
        + 2.136 * Sal / TempK
    )  # pK2 is not defined for Sal=0, since log10(0)=-inf
    K2 = 10.0 ** -pK2  # this is on the NBS scale
    return K1, K2


def kH2CO3_SWS_M79(TempK, Sal):
    """Carbonic acid dissociation constants following M79, pure water case."""
    # === CO2SYS.m comments: =======
    # PURE WATER CASE
    # Millero, F. J., Geochemica et Cosmochemica Acta 43:1651-1661, 1979:
    # K1 from refit data from Harned and Davis,
    # J American Chemical Society, 65:2030-2037, 1943.
    # K2 from refit data from Harned and Scholes,
    # J American Chemical Society, 43:1706-1709, 1941.
    # This is only to be used for Sal=0 water (note the absence of S in the
    # below formulations).
    # These are the thermodynamic Constants:
    logTempK = np.log(TempK)
    lnK1 = 290.9097 - 14554.21 / TempK - 45.0575 * logTempK
    K1 = np.exp(lnK1)
    lnK2 = 207.6548 - 11843.79 / TempK - 33.6485 * logTempK
    K2 = np.exp(lnK2)
    return K1, K2


def kH2CO3_NBS_CW98(TempK, Sal):
    """Carbonic acid dissociation constants following CW98."""
    # === CO2SYS.m comments: =======
    # From Cai and Wang 1998, for estuarine use.
    # Data used in this work is from:
    # K1: Merhback (1973) for S>15, for S<15: Mook and Keone (1975)
    # K2: Merhback (1973) for S>20, for S<20: Edmond and Gieskes (1970)
    # Sigma of residuals between fits and above data: Â±0.015, +0.040 for K1
    # and K2, respectively.
    # Sal 0-40, Temp 0.2-30
    # Limnol. Oceanogr. 43(4) (1998) 657-668
    # On the NBS scale
    # Their check values for F1 don't work out, not sure if this was correctly
    # published...
    # Conversion to SWS scale by division by fH is uncertain at low Sal due to
    # junction potential.
    F1 = 200.1 / TempK + 0.3220
    pK1 = (
        3404.71 / TempK
        + 0.032786 * TempK
        - 14.8435
        - 0.071692 * F1 * Sal ** 0.5
        + 0.0021487 * Sal
    )
    K1 = 10.0 ** -pK1  # this is on the NBS scale
    F2 = -129.24 / TempK + 1.4381
    pK2 = (
        2902.39 / TempK
        + 0.02379 * TempK
        - 6.4980
        - 0.3191 * F2 * Sal ** 0.5
        + 0.0198 * Sal
    )
    K2 = 10.0 ** -pK2  # this is on the NBS scale
    return K1, K2


def kH2CO3_TOT_LDK00(TempK, Sal):
    """Carbonic acid dissociation constants following LDK00."""
    # === CO2SYS.m comments: =======
    # From Lueker, Dickson, Keeling, 2000
    # This is Mehrbach's data refit after conversion to the Total scale, for
    # comparison with their equilibrator work.
    # Mar. Chem. 70 (2000) 105-119
    # Total scale and kg-sw
    pK1 = (
        3633.86 / TempK
        - 61.2172
        + 9.6777 * np.log(TempK)
        - 0.011555 * Sal
        + 0.0001152 * Sal ** 2
    )
    K1 = 10.0 ** -pK1  # this is on the Total pH scale in mol/kg-SW
    pK2 = (
        471.78 / TempK
        + 25.929
        - 3.16967 * np.log(TempK)
        - 0.01781 * Sal
        + 0.0001122 * Sal ** 2
    )
    K2 = 10.0 ** -pK2  # this is on the Total pH scale in mol/kg-SW
    return K1, K2


def kH2CO3_SWS_MM02(TempK, Sal):
    """Carbonic acid dissociation constants following MM02."""
    # === CO2SYS.m comments: =======
    # Mojica Prieto and Millero 2002. Geochim. et Cosmochim. Acta. 66(14),
    # 2529-2540.
    # sigma for pK1 is reported to be 0.0056
    # sigma for pK2 is reported to be 0.010
    # This is from the abstract and pages 2536-2537
    pK1 = (
        -43.6977
        - 0.0129037 * Sal
        + 1.364e-4 * Sal ** 2
        + 2885.378 / TempK
        + 7.045159 * np.log(TempK)
    )
    pK2 = (
        -452.0940
        + 13.142162 * Sal
        - 8.101e-4 * Sal ** 2
        + 21263.61 / TempK
        + 68.483143 * np.log(TempK)
        + (-581.4428 * Sal + 0.259601 * Sal ** 2) / TempK
        - 1.967035 * Sal * np.log(TempK)
    )
    K1 = 10.0 ** -pK1  # this is on the SWS pH scale in mol/kg-SW
    K2 = 10.0 ** -pK2  # this is on the SWS pH scale in mol/kg-SW
    return K1, K2


def kH2CO3_SWS_MPL02(TempK, Sal):
    """Carbonic acid dissociation constants following MPL02."""
    # === CO2SYS.m comments: =======
    # Millero et al., 2002. Deep-Sea Res. I (49) 1705-1723.
    # Calculated from overdetermined WOCE-era field measurements
    # sigma for pK1 is reported to be 0.005
    # sigma for pK2 is reported to be 0.008
    # This is from page 1715
    TempC = TempK - 273.15
    pK1 = 6.359 - 0.00664 * Sal - 0.01322 * TempC + 4.989e-5 * TempC ** 2
    pK2 = 9.867 - 0.01314 * Sal - 0.01904 * TempC + 2.448e-5 * TempC ** 2
    K1 = 10.0 ** -pK1  # this is on the SWS pH scale in mol/kg-SW
    K2 = 10.0 ** -pK2  # this is on the SWS pH scale in mol/kg-SW
    return K1, K2


def kH2CO3_SWS_MGH06(TempK, Sal):
    """Carbonic acid dissociation constants following MGH06."""
    # === CO2SYS.m comments: =======
    # From Millero 2006 work on pK1 and pK2 from titrations
    # Millero, Graham, Huang, Bustos-Serrano, Pierrot. Mar.Chem. 100 (2006)
    # 80-94.
    # S=1 to 50, T=0 to 50. On seawater scale (SWS). From titrations in Gulf
    # Stream seawater.
    pK1_0 = -126.34048 + 6320.813 / TempK + 19.568224 * np.log(TempK)
    A_1 = 13.4191 * Sal ** 0.5 + 0.0331 * Sal - 5.33e-5 * Sal ** 2
    B_1 = -530.123 * Sal ** 0.5 - 6.103 * Sal
    C_1 = -2.06950 * Sal ** 0.5
    pK1 = A_1 + B_1 / TempK + C_1 * np.log(TempK) + pK1_0  # pK1 sigma = 0.0054
    K1 = 10.0 ** -(pK1)
    pK2_0 = -90.18333 + 5143.692 / TempK + 14.613358 * np.log(TempK)
    A_2 = 21.0894 * Sal ** 0.5 + 0.1248 * Sal - 3.687e-4 * Sal ** 2
    B_2 = -772.483 * Sal ** 0.5 - 20.051 * Sal
    C_2 = -3.3336 * Sal ** 0.5
    pK2 = A_2 + B_2 / TempK + C_2 * np.log(TempK) + pK2_0  # pK2 sigma = 0.011
    K2 = 10.0 ** -(pK2)
    return K1, K2


def kH2CO3_SWS_M10(TempK, Sal):
    """Carbonic acid dissociation constants following M10."""
    # === CO2SYS.m comments: =======
    # From Millero, 2010, also for estuarine use.
    # Marine and Freshwater Research, v. 61, p. 139-142.
    # Fits through compilation of real seawater titration results:
    # Mehrbach et al. (1973), Mojica-Prieto & Millero (2002), Millero et al.
    # (2006)
    # Constants for K's on the SWS;
    # This is from page 141
    pK10 = -126.34048 + 6320.813 / TempK + 19.568224 * np.log(TempK)
    # This is from their table 2, page 140.
    A1 = 13.4038 * Sal ** 0.5 + 0.03206 * Sal - 5.242e-5 * Sal ** 2
    B1 = -530.659 * Sal ** 0.5 - 5.8210 * Sal
    C1 = -2.0664 * Sal ** 0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    K1 = 10.0 ** -pK1
    # This is from page 141
    pK20 = -90.18333 + 5143.692 / TempK + 14.613358 * np.log(TempK)
    # This is from their table 3, page 140.
    A2 = 21.3728 * Sal ** 0.5 + 0.1218 * Sal - 3.688e-4 * Sal ** 2
    B2 = -788.289 * Sal ** 0.5 - 19.189 * Sal
    C2 = -3.374 * Sal ** 0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    K2 = 10.0 ** -pK2
    return K1, K2


def _kH2CO3_WMW14(TempK):
    """Scale-independent part of WMW14 carbonic acid dissociation constants."""
    pK1_0 = -126.34048 + 6320.813 / TempK + 19.568224 * np.log(TempK)
    pK2_0 = -90.18333 + 5143.692 / TempK + 14.613358 * np.log(TempK)
    return pK1_0, pK2_0


def kH2CO3_SWS_WMW14(TempK, Sal):
    """Carbonic acid dissociation constants, Seawater scale, following WMW14."""
    # === CO2SYS.m comments: =======
    # From Waters, Millero, Woosley 2014
    # Mar. Chem., 165, 66-67, 2014
    # Corrigendum to "The free proton concentration scale for seawater pH".
    # Effectively, this is an update of Millero (2010) formulation
    # (WhichKs==14)
    # Constants for K's on the SWS;
    pK10, pK20 = _kH2CO3_WMW14(TempK)
    A1 = 13.409160 * Sal ** 0.5 + 0.031646 * Sal - 5.1895e-5 * Sal ** 2
    B1 = -531.3642 * Sal ** 0.5 - 5.713 * Sal
    C1 = -2.0669166 * Sal ** 0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    K1 = 10.0 ** -pK1
    A2 = 21.225890 * Sal ** 0.5 + 0.12450870 * Sal - 3.7243e-4 * Sal ** 2
    B2 = -779.3444 * Sal ** 0.5 - 19.91739 * Sal
    C2 = -3.3534679 * Sal ** 0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    K2 = 10.0 ** -pK2
    return K1, K2


def k_H2CO3_TOT_WMW14(TempK, Sal):
    """Carbonic acid dissociation constants, Total scale, following WM13/WMW14."""
    # Coefficients from the corrigendum document [WMW14]
    pK10, pK20 = _kH2CO3_WMW14(TempK)
    A1 = 13.568513 * Sal ** 0.5 + 0.031645 * Sal - 5.3834e-5 * Sal ** 2
    B1 = -539.2304 * Sal ** 0.5 - 5.635 * Sal
    C1 = -2.0901396 * Sal ** 0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    K1 = 10.0 ** -pK1
    A2 = 21.389248 * Sal ** 0.5 + 0.12452358 * Sal - 3.7447e-4 * Sal ** 2
    B2 = -787.3736 * Sal ** 0.5 - 19.84233 * Sal
    C2 = -3.3773006 * Sal ** 0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    K2 = 10.0 ** -pK2
    return K1, K2


def k_H2CO3_FREE_WMW14(TempK, Sal):
    """Carbonic acid dissociation constants, Free scale, following WM13/WMW14."""
    # Coefficients from the corrigendum document [WMW14]
    pK10, pK20 = _kH2CO3_WMW14(TempK)
    A1 = 5.592953 * Sal ** 0.5 + 0.028845 * Sal - 6.388e-5 * Sal ** 2
    B1 = -225.7489 * Sal ** 0.5 - 4.761 * Sal
    C1 = -0.8715109 * Sal ** 0.5
    pK1 = pK10 + A1 + B1 / TempK + C1 * np.log(TempK)
    K1 = 10.0 ** -pK1
    A2 = 13.396949 * Sal ** 0.5 + 0.12193009 * Sal - 3.8362e-4 * Sal ** 2
    B2 = -472.8633 * Sal ** 0.5 - 19.03634 * Sal
    C2 = -2.1563270 * Sal ** 0.5
    pK2 = pK20 + A2 + B2 / TempK + C2 * np.log(TempK)
    K2 = 10.0 ** -pK2
    return K1, K2


def kH2CO3_TOT_SLH20(TempK, Sal):
    """Carbonic acid dissociation constants following SLH20."""
    # Coefficients and their 95% confidence intervals from SLH20 Table 1.
    pK1 = (
        8510.63 / TempK  # ±1139.8
        - 172.4493  # ±26.131
        + 26.32996 * np.log(TempK)  # ±3.9161
        - 0.011555 * Sal
        + 0.0001152 * Sal ** 2
    )
    K1 = 10.0 ** -pK1  # this is on the Total pH scale in mol/kg-SW
    pK2 = (
        4226.23 / TempK  # ±1050.8
        - 59.4636  # ±24.016
        + 9.60817 * np.log(TempK)  # ±3.5966
        - 0.01781 * Sal
        + 0.0001122 * Sal ** 2
    )
    K2 = 10.0 ** -pK2  # this is on the Total pH scale in mol/kg-SW
    return K1, K2


def kH2CO3_TOT_SB21(TempK, Sal):
    """Carbonic acid dissociation constants with K2 following SB21.
    K1 comes from WMW14.
    """
    K1 = k_H2CO3_TOT_WMW14(TempK, Sal)[0]
    pK2 = (
        116.8067
        - 3655.02 / TempK
        - 16.45817 * np.log(TempK)
        + 0.04523 * Sal
        - 0.615 * np.sqrt(Sal)
        - 0.0002799 * Sal ** 2
        + 4.969 * Sal / TempK
    )
    K2 = 10.0 ** -pK2
    return K1, K2


def kH2S_TOT_YM95(TempK, Sal):
    """Hydrogen sulfide dissociation constant following YM95."""
    # === CO2SYS_v1_21.m comments: =======
    # H2S  Millero et. al.( 1988)  Limnol. Oceanogr. 33,269-274.
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995. Total Scale.
    # Yao Millero say equations have been refitted to SWS scale but not true as
    # they agree with Millero 1988 which are on Total Scale.
    # Also, calculations agree at high H2S with AquaEnv when assuming it is on
    # Total Scale.
    lnkH2S = (
        225.838
        - 13275.3 / TempK
        - 34.6435 * np.log(TempK)
        + 0.3449 * np.sqrt(Sal)
        - 0.0274 * Sal
    )
    return np.exp(lnkH2S)


def kNH3_SWS_YM95(TempK, Sal):
    """Ammonium association constant following YM95."""
    # === CO2SYS_v1_21.m comments: =======
    # Yao and Millero, Aquatic Geochemistry 1:53-88, 1995   SWS
    lnkNH3 = (
        -6285.33 / TempK
        + 0.0001635 * TempK
        - 0.25444
        + (0.46532 - 123.7184 / TempK) * np.sqrt(Sal)
        + (-0.01992 + 3.17556 / TempK) * Sal
    )
    return np.exp(lnkNH3)


def kNH3_TOT_CW95(TempK, Sal):
    """Ammonium association constant following CW95."""
    # === CO2SYS_v1_21.m comments: =======
    # Clegg Whitfield 1995
    # Geochimica et Cosmochimica Acta, Vol. 59, No. 12. pp. 2403-2421
    # eq (18)  Total scale   t=[-2 to 40 oC]  S=[0 to 40 ppt]   pK=+-0.00015
    PKNH3expCW = 9.244605 - 2729.33 * (1 / 298.15 - 1 / TempK)
    PKNH3expCW += (0.04203362 - 11.24742 / TempK) * Sal ** 0.25
    PKNH3expCW += (
        -13.6416 + 1.176949 * TempK ** 0.5 - 0.02860785 * TempK + 545.4834 / TempK
    ) * Sal ** 0.5
    PKNH3expCW += (
        -0.1462507
        + 0.0090226468 * TempK ** 0.5
        - 0.0001471361 * TempK
        + 10.5425 / TempK
    ) * Sal ** 1.5
    PKNH3expCW += (
        0.004669309 - 0.0001691742 * TempK ** 0.5 - 0.5677934 / TempK
    ) * Sal ** 2
    PKNH3expCW += (-2.354039e-05 + 0.009698623 / TempK) * Sal ** 2.5
    KNH3 = 10.0 ** -PKNH3expCW  # this is on the total pH scale in mol/kg-H2O
    KNH3 = KNH3 * (1 - 0.001005 * Sal)  # convert to mol/kg-SW
    return KNH3
