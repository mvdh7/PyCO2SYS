# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
"""
PyCO2SYS.equilibria.pcx
=======================
Calculate presure-correction factors for equilibrium constants.  These should be
multiplied by the raw K value (not pK) to convert.

Functions
---------
pressure_factor
    Calculate pressure-correction factor for a particular equilibrium constant using
    the deltaV / kappa formulation.
factor_k_BOH3_M79
    Calculate pressure-correction factor for k_BOH3 following M79.
    Used when opt_factor_pk_BOH3 = 1.
factor_k_BOH3_GEOSECS
    Calculate pressure-correction factor for k_BOH3 following the GEOSECS approach.
    Used when opt_factor_pk_BOH3 = 2.
factor_k_H2O
    Calculate pressure-correction factor for k_H2O.  Used when opt_factor_pk_H2O = 1.
factor_k_H2O_fw
    Calculate pressure-correction factor for k_H2O in freshwater.
    Used when opt_factor_pk_H2O = 2.
factor_k_H2S
    Calculate pressure-correction factor for k_H2S.
factor_k_HSO4
    Calculate pressure-correction factor for k_HSO4.
factor_k_H3PO4
    Calculate pressure-correction factor for k_H3PO4.
factor_k_H2PO4
    Calculate pressure-correction factor for k_H2PO4.
factor_k_HPO4
    Calculate pressure-correction factor for k_HPO4.
factor_k_Si
    Calculate pressure-correction factor for k_Si.
factor_k_NH3
    Calculate pressure-correction factor for k_NH3.
factor_k_H2CO3
    Calculate pressure-correction factor for k_H2CO3.
    Used when opt_factor_pk_H2CO3 = 1.
factor_k_H2CO3_GEOSECS
    Calculate pressure-correction factor for k_H2CO3 following GEOSECS.
    Used when opt_factor_pk_H2CO3 = 2.
factor_k_H2CO3_fw
    Calculate pressure-correction factor for k_H2CO3 in freshwater.
    Used when opt_factor_pk_H2CO3 = 3.
factor_k_H2CO3
    Calculate pressure-correction factor for k_HCO3.
    Used when opt_factor_pk_HCO3 = 1.
factor_k_HCO3_GEOSECS
    Calculate pressure-correction factor for k_HCO3 following GEOSECS.
    Used when opt_factor_pk_HCO3 = 2.
factor_k_HCO3_fw
    Calculate pressure-correction factor for k_HCO3 in freshwater.
    Used when opt_factor_pk_HCO3 = 3.
factor_k_CO2
    Calculate the pressure-correction factor for k_CO2 following W74 eq. 5.
"""

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
from jax import numpy as np

from .. import convert
from ..meta import valid


def pressure_factor(deltaV, kappa, pressure, temperature, gas_constant):
    """Calculate pressure-correction factor for a particular equilibrium constant using
    the deltaV / kappa formulation.

    Parameters
    ----------
    deltaV : float
        The deltaV constant for the pressure correction.
    kappa : float
        The kappa constant for the pressure correction.
    pressure : float
        Hydrostatic pressure in dbar.
    temperature : float
        Temperature in °C.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    Pbar = convert.decibar_to_bar(pressure)
    TempK = convert.celsius_to_kelvin(temperature)
    return np.exp((-deltaV + 0.5 * kappa * Pbar) * Pbar / (gas_constant * TempK))


def factor_k_H2S(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H2S.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # === CO2SYS.m comments: =======
    # Millero 1995 gives values for deltaV in fresh water instead of SW.
    # Millero 1995 gives -b0 as -2.89 instead of 2.89.
    # Millero 1983 is correct for both.
    deltaV = -11.07 - 0.009 * temperature - 0.000942 * temperature**2
    kappa = (-2.89 + 0.054 * temperature) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_HSO4(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_HSO4.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # === CO2SYS.m comments: =======
    # This is from Millero, 1995, which is the same as Millero, 1983.
    # It is assumed that KS is on the free pH scale.
    deltaV = -18.03 + 0.0466 * temperature + 0.000316 * temperature**2
    kappa = (-4.53 + 0.09 * temperature) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_HF(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_HF.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # === CO2SYS.m comments: =======
    # This is from Millero, 1995, which is the same as Millero, 1983.
    # It is assumed that KF is on the free pH scale.
    deltaV = -9.78 - 0.009 * temperature - 0.000942 * temperature**2
    Kappa = (-3.91 + 0.054 * temperature) / 1000
    return pressure_factor(deltaV, Kappa, pressure, temperature, gas_constant)


@valid(temperature=[2, 22], pressure=[0, 10132.5])
def factor_k_BOH3_M79(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_BOH3 following M79.
    Used when opt_factor_pk_BOH3 = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # Below: this is from Millero, 1979.
    # It is from data of Culberson and Pytkowicz, 1968.
    deltaV = -29.48 + 0.1622 * temperature - 0.002608 * temperature**2
    # Millero, 1983 has:
    #   deltaV = -28.56 + .1211*TempCi - .000321*TempCi*TempCi
    # Millero, 1992 has:
    #   deltaV = -29.48 + .1622*TempCi + .295*(Sali - 34.8)
    # Millero, 1995 has:
    #   deltaV = -29.48 - .1622*TempCi - .002608*TempCi*TempCi
    #   deltaV = deltaV + .295*(Sali - 34.8) # Millero, 1979
    kappa = -2.84 / 1000  # Millero, 1979
    # Millero, 1992 and Millero, 1995 also have this.
    #   Kappa = Kappa + .354*(Sali - 34.8)/1000: # Millero,1979
    # Millero, 1983 has:
    #   Kappa = (-3 + .0427*TempCi)/1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_BOH3_GEOSECS(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_BOH3 following the GEOSECS approach.
    Used when opt_factor_pk_BOH3 = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
    # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
    # Culberson and Pytkowicz, L and O 13:403-417, 1968:
    # but the fits are the same as those in Edmond and Gieskes, GCA, 34:1261-1291, 1970
    # who in turn quote Li, personal communication
    TempK = convert.celsius_to_kelvin(temperature)
    Pbar = convert.decibar_to_bar(pressure)
    # This one is handled differently, because the equation doesn't fit the
    # standard deltaV & Kappa form of pressure_factor.
    return np.exp((27.5 - 0.095 * temperature) * Pbar / (gas_constant * TempK))


def factor_k_H2O_fw(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H2O in freshwater.
    Used when opt_factor_pk_H2O = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # This is from Millero, 1983.
    deltaV = -25.6 + 0.2324 * temperature - 0.0036246 * temperature**2
    kappa = (-7.33 + 0.1368 * temperature - 0.001233 * temperature**2) / 1000
    # Note: the temperature dependence of KappaK1 and KappaKW for freshwater
    # in Millero, 1983 are the same.
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_H2O(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H2O.  Used when opt_factor_pk_H2O = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # GEOSECS doesn't include OH term, so this won't matter.
    # Peng et al didn't include pressure, but here I assume that the KW
    # correction is the same as for the other seawater cases.
    # This is from Millero, 1983 and his programs CO2ROY(T).BAS.
    deltaV = -20.02 + 0.1119 * temperature - 0.001409 * temperature**2
    # Millero, 1992 and Millero, 1995 have:
    kappa = (-5.13 + 0.0794 * temperature) / 1000  # Millero, 1983
    # Millero, 1995 has this too, but Millero, 1992 is different.
    # Millero, 1979 does not list values for these.
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_H3PO4(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H3PO4.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    deltaV = -14.51 + 0.1211 * temperature - 0.000321 * temperature**2
    kappa = (-2.67 + 0.0427 * temperature) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_H2PO4(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H2PO4.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    deltaV = -23.12 + 0.1758 * temperature - 0.002647 * temperature**2
    kappa = (-5.15 + 0.09 * temperature) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_HPO4(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_HPO4.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    deltaV = -26.57 + 0.202 * temperature - 0.003042 * temperature**2
    kappa = (-4.08 + 0.0714 * temperature) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_Si(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_Si.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # === CO2SYS.m comments: =======
    # The only mention of this is Millero, 1995 where it is stated that the
    # values have been estimated from the values of boric acid. HOWEVER,
    # there is no listing of the values in the table.
    # Here we use the values for boric acid.
    deltaV = -29.48 + 0.1622 * temperature - 0.002608 * temperature**2
    kappa = -2.84 / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_NH3(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_NH3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # === CO2SYS.m comments: =======
    # The corrections are from Millero, 1995, which are the same as Millero, 1983.
    deltaV = -26.43 + 0.0889 * temperature - 0.000905 * temperature**2
    kappa = (-5.03 + 0.0814 * temperature) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


@valid(temperature=[2, 22], pressure=[0, 10132.5])
def factor_k_H2CO3(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H2CO3.
    Used when opt_factor_pk_H2CO3 = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # These are from Millero, 1995.
    # They are the same as Millero, 1979 and Millero, 1992.
    # They are from data of Culberson and Pytkowicz, 1968.
    deltaV = -25.5 + 0.1271 * temperature
    # deltaV = deltaV - .151*(Sali - 34.8) # Millero, 1979
    kappa = (-3.08 + 0.0877 * temperature) / 1000
    # Kappa = Kappa - .578*(Sali - 34.8)/1000 # Millero, 1979
    # The fits given in Millero, 1983 are somewhat different.
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_H2CO3_fw(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H2CO3 in freshwater.
    Used when opt_factor_pk_H2CO3 = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # Pressure effects on K1 in freshwater: this is from Millero, 1983.
    deltaV = -30.54 + 0.1849 * temperature - 0.0023366 * temperature**2
    kappa = (-6.22 + 0.1368 * temperature - 0.001233 * temperature**2) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_H2CO3_GEOSECS(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_H2CO3 following GEOSECS.
    Used when opt_factor_pk_H2CO3 = 3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
    # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
    # Culberson and Pytkowicz, L and O 13:403-417, 1968:
    # but the fits are the same as those in
    # Edmond and Gieskes, GCA, 34:1261-1291, 1970
    # who in turn quote Li, personal communication
    # This one is handled differently because the equation doesn't fit the
    # standard deltaV & kappa form of pressure_factor.
    Pbar = convert.decibar_to_bar(pressure)
    return np.exp(
        (24.2 - 0.085 * temperature) * Pbar / (gas_constant * (temperature + 273.15))
    )


@valid(temperature=[2, 22], pressure=[0, 10132.5])
def factor_k_HCO3(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_HCO3.
    Used when opt_factor_pk_HCO3 = 1.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # These are from Millero, 1995.
    # They are the same as Millero, 1979 and Millero, 1992.
    # They are from data of Culberson and Pytkowicz, 1968.
    deltaV = -15.82 - 0.0219 * temperature
    # deltaV = deltaV + .321*(Sali - 34.8) # Millero, 1979
    kappa = (1.13 - 0.1475 * temperature) / 1000
    # Kappa = Kappa - .314*(Sali - 34.8)/1000 # Millero, 1979
    # The fit given in Millero, 1983 is different.
    # Not by a lot for deltaV, but by much for Kappa.
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_HCO3_fw(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_HCO3 in freshwater.
    Used when opt_factor_pk_HCO3 = 2.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # Pressure effects on K2 in freshwater: this is from Millero, 1983.
    deltaV = -29.81 + 0.115 * temperature - 0.001816 * temperature**2
    kappa = (-5.74 + 0.093 * temperature - 0.001896 * temperature**2) / 1000
    return pressure_factor(deltaV, kappa, pressure, temperature, gas_constant)


def factor_k_HCO3_GEOSECS(temperature, pressure, gas_constant):
    """Calculate pressure-correction factor for k_HCO3 following GEOSECS.
    Used when opt_factor_pk_HCO3 = 3.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        The universal gas constant in ml / (bar * K * mol).

    Returns
    -------
    float
        The correction factor, to be multiplied by the K value to correct it.
    """
    # GEOSECS Pressure Effects On K1, K2, KB (on the NBS scale)
    # Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982 quotes
    # Culberson and Pytkowicz, L and O 13:403-417, 1968:
    # but the fits are the same as those in
    # Edmond and Gieskes, GCA, 34:1261-1291, 1970
    # who in turn quote Li, personal communication
    # Takahashi et al had 26.4, but 16.4 is from Edmond and Gieskes
    # and matches the GEOSECS results
    # This one is handled differently because the equation doesn't fit the
    # standard deltaV & Kappa form of pressure_factor.
    Pbar = convert.decibar_to_bar(pressure)
    return np.exp(
        (16.4 - 0.04 * temperature) * Pbar / (gas_constant * (temperature + 273.15))
    )


@valid(temperature=[-1, 40], pressure=[0, 5066.25])
def factor_k_CO2(temperature, pressure, gas_constant, pressure_atmosphere):
    """Calculate the pressure-correction factor for k_CO2 following W74 eq. 5.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    gas_constant : float
        Universal gas constant in ml / (bar * K * mol).
    pressure_atmosphere : float
        Atmospheric pressure in atm.

    Returns
    -------
    float
        pressure-correction factor for kCO2.
    """
    pressure_bar = convert.decibar_to_bar(pressure)
    vCO2 = 32.3  # partial molar volume of CO2 in ml / mol
    # Note that PyCO2SYS's gas_constant R is in bar units, not atm, so the pressures
    # and equation here are all converted into bar, unlike in the original W74.
    return np.exp(
        (1.01325 - (pressure_bar + pressure_atmosphere * 1.01325))
        * vCO2
        / (gas_constant * (temperature + 273.15))
    )
