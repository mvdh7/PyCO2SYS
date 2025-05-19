# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
from jax import numpy as np

from . import constants

bh_TOG93_H24 = 28995  # J / mol
bh_enthalpy_H24 = 25288  # J / mol
bl_TOG93 = 42.3e-3  # 1 / °C
aq_TOG93 = -43.5e-6  # 1 / °C ** 2
bq_TOG93 = 43.3e-3  # 1 / °C


def inverse(temperature, gas_constant, bh):
    """Calculate υ using the van 't Hoff form of Humphreys (2024) with a variable
    bh coefficient.

    Parameters
    ----------
    temperature : array-like
        Temperature in °C.
    gas_constant : float
        The universal gas constant in ml / (bar mol K).

    Returns
    -------
        υ in % / °C.
    """
    return 100 * bh / (gas_constant * 0.1 * (temperature + constants.Tzero) ** 2)


def expUps_Hoff_H24(temperature__pre, temperature, gas_constant, bh):
    """Calculate exp(Υ) using the van 't Hoff form of Humphreys (2024) with a variable
    bh coefficient.

    Parameters
    ----------
    temperature__pre : array-like
        Starting temperature (t0) in °C.
    temperature : array-like
        Adjusted temperature (t1) in °C.
    gas_constant : float
        The universal gas constant in ml / (bar mol K).

    Returns
    -------
    array-like
        The adjustment factor exp(Υ).
    """
    return np.exp(
        (1 / (temperature__pre + constants.Tzero) - 1 / (temperature + constants.Tzero))
        * bh
        / (gas_constant * 0.1)
    )


def get_bh_H24(temperature, salinity, fCO2):
    """Calculate bh based on the parameterisation of Humphreys (2024) to the OceanSODA-
    ETZH data product.

    Parameters
    ----------
    temperature : array-like
        Temperature in °C.
    salinity : array-like
        Practical salinity.
    fCO2 : array-like
        Seawater fugacity of CO2 in µatm.

    Returns
    -------
    bh : array-like
        The coefficient bh in J / mol.
    """
    c, t, tt, s, ss, f, ff, ts, tf, sf = (
        3.13184463e04,
        1.39487529e02,
        -1.21087624e00,
        -4.22484243e00,
        -6.52212406e-01,
        -1.69522191e01,
        -5.47585838e-04,
        -3.02071783e00,
        1.66972942e-01,
        3.09654019e-01,
    )
    return (
        c
        + t * temperature
        + tt * temperature**2
        + s * salinity
        + ss * salinity**2
        + f * fCO2
        + ff * fCO2**2
        + ts * temperature * salinity
        + tf * temperature * fCO2
        + sf * salinity * fCO2
    )


def ups_parameterised_H24(temperature, salinity, fCO2, gas_constant):
    """Calculate υ using the van 't Hoff form of Humphreys (2024) with a variable bh
    coefficient based on a parameterisation with the OceanSODA-ETZH data product.

    Parameters
    ----------
    temperature : array-like
        Temperature in °C.
    salinity : array-like
        Practical salinity.
    fCO2 : array-like
        Seawater fugacity of CO2 in µatm.
    gas_constant : float
        The universal gas constant in ml / (bar mol K).

    Returns
    -------
        υ in % / °C.
    """
    bh = get_bh_H24(temperature, salinity, fCO2)
    return inverse(temperature, gas_constant, bh)


def expUps_parameterised_H24_t0_insitu(
    temperature__pre,
    temperature,
    salinity,
    fCO2__pre,
    gas_constant,
):
    """Calculate adjustment factor exp(Υ) using the van 't Hoff form of Humphreys (2024)
    with a constant bh coefficient based on a parameterisation with the OceanSODA-ETZH
    data product.

    Parameters
    ----------
    temperature__pre : array-like
        Starting temperature (t0) in °C.
    temperature : array-like
        Adjusted temperature (t1) in °C.
    salinity : array-like
        Practical salinity.
    fCO2 : array-like
        Seawater CO2 fugacity at the input condition (t0).
    gas_constant : array-like
        Universal gas constant.

    Returns
    -------
    array-like
        The adjustment factor exp(Υ).
    """
    # Calculate bh from the input conditions
    bh = get_bh_H24(temperature__pre, salinity, fCO2__pre)
    return expUps_Hoff_H24(temperature__pre, temperature, gas_constant, bh)


def expUps_parameterised_H24(
    temperature,
    temperature_out,
    salinity,
    fCO2,
    gas_constant,
    opt_which_fCO2_insitu=1,
):
    """Calculate adjustment factor exp(Υ) using the van 't Hoff form of Humphreys (2024)
    with a constant bh coefficient based on a parameterisation with the OceanSODA-ETZH
    data product.

    Parameters
    ----------
    temperature : array-like
        Starting temperature (t0) in °C.
    temperature_out : array-like
        Adjusted temperature (t1) in °C.
    salinity : array-like
        Practical salinity.
    fCO2 : array-like
        Seawater CO2 fugacity at the condition corresponding to opt_which_fCO2_insitu.
    gas_constant : array-like
        Universal gas constant.
    opt_which_fCO2_insitu : int, optional
        Whether (1) the input condition (starting) or (2) output condition (adjusted)
        temperature should be used to calculate bh, by default 1 (i.e., input).  If
        using (2), then the user needs to make sure the fCO2 input already approximately
        represents output conditions, e.g. by using expUps_TOG93_H24.

    Returns
    -------
    array-like
        The adjustment factor exp(Υ).
    """
    if opt_which_fCO2_insitu == 1:
        # Calculate bh from the input conditions
        bh = get_bh_H24(temperature, salinity, fCO2)
    elif opt_which_fCO2_insitu == 2:
        # Calculate bh from the output conditions (fCO2 should also be estimated there)
        bh = get_bh_H24(temperature_out, salinity, fCO2)
    return expUps_Hoff_H24(temperature, temperature_out, gas_constant, bh)


def ups_enthalpy_H24(temperature, gas_constant):
    """Calculate υ using the van 't Hoff form of Humphreys (2024) with a constant bh
    coefficient based on the approximation with standard enthalpies of reaction.

    Parameters
    ----------
    temperature : array-like
        Temperature in °C.
    gas_constant : float
        The universal gas constant in ml / (bar mol K).

    Returns
    -------
        υ in % / °C.
    """
    return inverse(temperature, gas_constant, bh_enthalpy_H24)


def expUps_enthalpy_H24(temperature__pre, temperature, gas_constant):
    """Calculate adjustment factor exp(Υ) using the van 't Hoff form of Humphreys (2024)
    with a constant bh coefficient based on the approximation with standard enthalpies
    of reaction.

    Parameters
    ----------
    temperature : array-like
        Starting temperature (t0) in °C.
    temperature_out : array-like
        Adjusted temperature (t1) in °C.

    Returns
    -------
    array-like
        The adjustment factor exp(Υ).
    """
    return expUps_Hoff_H24(temperature__pre, temperature, gas_constant, bh_enthalpy_H24)


def ups_TOG93_H24(temperature, gas_constant):
    """Calculate υ using the van 't Hoff form of Humphreys (2024) with a constant bh
    coefficient fitted to the Takahashi et al. (1993) dataset.

    Parameters
    ----------
    temperature : array-like
        Temperature in °C.
    gas_constant : float
        The universal gas constant in ml / (bar mol K).

    Returns
    -------
        υ in % / °C.
    """
    return inverse(temperature, gas_constant, bh_TOG93_H24)


def expUps_TOG93_H24(temperature__pre, temperature, gas_constant):
    """Calculate adjustment factor exp(Υ) using the van 't Hoff form of Humphreys (2024)
    with a constant bh coefficient fitted to the Takahashi et al. (1993) dataset.

    Parameters
    ----------
    temperature : array-like
        Starting temperature (t0) in °C.
    temperature_out : array-like
        Adjusted temperature (t1) in °C.

    Returns
    -------
    array-like
        The adjustment factor exp(Υ).
    """
    return expUps_Hoff_H24(temperature__pre, temperature, gas_constant, bh_TOG93_H24)


def ups_linear_TOG93():
    """Return υ from the linear fit of Takahashi et al. (1993).

    Returns
    -------
    float
        υ in %/ °C.
    """
    return bl_TOG93


def linear(bl):
    return bl * 100


def quadratic(temperature, aq, bq):
    return 100 * (2 * aq * temperature + bq)


def expUps_linear_TOG93(temperature__pre, temperature):
    """Calculate adjustment factor exp(Υ) with the linear fit of Takahashi et al.
    (1993).

    Parameters
    ----------
    temperature__pre : array-like
        Starting temperature (t0) in °C or K.
    temperature : array-like
        Adjusted temperature (t1) in °C or K.

    Returns
    -------
    array-like
        The adjustment factor exp(Υ).
    """
    return np.exp(bl_TOG93 * (temperature - temperature__pre))


def ups_quadratic_TOG93(temperature):
    """Calculate υ from the quadratic fit of Takahashi et al. (1993).

    Parameters
    ----------
    temperature : array-like
        Temperature in °C.

    Returns
    -------
        υ in % / °C.
    """
    return 2 * aq_TOG93 * temperature + bq_TOG93


def expUps_quadratic_TOG93(temperature__pre, temperature):
    """Calculate adjustment factor exp(Υ) with the quadratic fit of Takahashi et al.
    (1993).

    Parameters
    ----------
    temperature__pre : array-like
        Starting temperature (t0) in °C.
    temperature : array-like
        Adjusted temperature (t1) in °C.

    Returns
    -------
    array-like
        The adjustment factor exp(Υ).
    """
    return np.exp(
        aq_TOG93 * (temperature**2 - temperature__pre**2)
        + bq_TOG93 * (temperature - temperature__pre)
    )
