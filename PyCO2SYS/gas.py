# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate gas properties."""

from autograd import numpy as np
from . import convert


def fugacity_factor(
    temperature,
    opt_k_carbonic,
    gas_constant,
    pressure_bar,
    pressure_atmosphere=1.0,
    opt_pressured_kCO2=0,
):
    """Calculate the fugacity factor following W74.

    Parameters
    ----------
    temperature : float
        Temperature in °C.
    opt_k_carbonic : int
        Which carbonic acid dissociation constants to use.
    gas_constant : float
        The universal gas constant in .
    pressure_bar : float
        Hydrostatic pressure in bar.
    pressure_atmosphere : float, optional
        Atmospheric pressure in atm, by default 1.0.
    opt_pressured_kCO2 : int, optional
        Whether to include the hydrostatic pressure effect, by default 0 (i.e., no).

    Returns
    -------
    float
        The fugacity factor; multiply this by pCO2 to get fCO2.
    """
    # Unless opt_pressured_kCO2 is set to 1, this assumes that the pressure is at one
    # atmosphere, or close to it.
    # Otherwise, the pressure term in the exponent affects the results.
    # Following Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    # Delta and B are in cm**3/mol.
    TempK = convert.celsius_to_kelvin(temperature)
    RT = gas_constant * TempK
    Delta = 57.7 - 0.118 * TempK
    b = (
        -1636.75
        + 12.0408 * TempK
        - 0.0327957 * TempK**2
        + 3.16528 * 0.00001 * TempK**3
    )
    # # For a mixture of CO2 and air at 1 atm (at low CO2 concentrations):
    # P1atm = 1.01325  # in bar
    p_bar = pressure_atmosphere * 1.01325  # convert atm to bar
    # If requested (opt_pressured_kCO2 == 1), account for hydrostatic pressure (new in
    # v1.8.2) - otherwise, just use atmospheric pressure like in previous versions
    p_total = np.where(opt_pressured_kCO2 == 1, p_bar + pressure_bar, p_bar)
    # Note that the x2**2 term below is ignored because it is almost equal to 1
    # (x2 = (1 - xCO2); xCO2 << 1)
    FugFac = np.exp((b + 2 * Delta) * p_total / RT)
    # GEOSECS and Peng assume pCO2 = fCO2, or FugFac = 1
    FugFac = np.where(np.isin(opt_k_carbonic, [6, 7]), 1.0, FugFac)
    return FugFac


def vpfactor(temperature, salinity, pressure_atmosphere=1.0):
    """Calculate the vapour pressure factor.

    Parameters
    ----------
    temperature : float
        Seawater temperature in degrees C.
    salinity : float
        Practical salinity.
    pressure_atmosphere : float, optional
        Barometric pressure in atmospheres.  Default = 1.0 atm.
    """
    # Weiss, R. F., and Price, B. A., Nitrous oxide solubility in water and
    #       seawater, Marine Chemistry 8:347-359, 1980.
    # They fit the data of Goff and Gratch (1946) with the vapor pressure
    #       lowering by sea salt as given by Robinson (1954).
    # This fits the more complicated Goff and Gratch, and Robinson equations
    #       from 273 to 313 deg K and 0 to 40 Sali with a standard error
    #       of .015#, about 5 uatm over this range.
    # This may be on IPTS-29 since they didn't mention the temperature scale,
    #       and the data of Goff and Gratch came before IPTS-48.
    # The references are:
    # Goff, J. A. and Gratch, S., Low pressure properties of water from -160 deg
    #       to 212 deg F, Transactions of the American Society of Heating and
    #       Ventilating Engineers 52:95-122, 1946.
    # Robinson, Journal of the Marine Biological Association of the U. K.
    #       33:449-455, 1954.
    #       This is eq. 10 on p. 350.
    #       This is in atmospheres.
    tempK = convert.celsius_to_kelvin(temperature)
    # WP80 eq. (10)
    VPWP = np.exp(24.4543 - 67.4509 * (100 / tempK) - 4.8489 * np.log(tempK / 100))
    VPCorrWP = np.exp(-0.000544 * salinity)
    VPSWWP = VPWP * VPCorrWP
    VPFac = pressure_atmosphere - VPSWWP
    return VPFac
