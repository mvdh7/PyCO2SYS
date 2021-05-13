# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate gas properties."""

from autograd import numpy as np
from . import convert


def fugacityfactor(TempC, WhichKs, RGas):
    """Calculate the fugacity factor."""
    # This assumes that the pressure is at one atmosphere, or close to it.
    # Otherwise, the Pres term in the exponent affects the results.
    # Following Weiss, R. F., Marine Chemistry 2:203-215, 1974.
    # Delta and B are in cm**3/mol.
    TempK = convert.TempC2K(TempC)
    RT = RGas * TempK
    Delta = 57.7 - 0.118 * TempK
    b = (
        -1636.75
        + 12.0408 * TempK
        - 0.0327957 * TempK ** 2
        + 3.16528 * 0.00001 * TempK ** 3
    )
    # For a mixture of CO2 and air at 1 atm (at low CO2 concentrations):
    P1atm = 1.01325  # in bar
    FugFac = np.exp((b + 2 * Delta) * P1atm / RT)
    # GEOSECS and Peng assume pCO2 = fCO2, or FugFac = 1
    FugFac = np.where((WhichKs == 6) | (WhichKs == 7), 1.0, FugFac)
    return FugFac


def vpfactor(TempC, Sal):
    """Calculate the vapour pressure factor."""
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
    TempK = convert.TempC2K(TempC)
    VPWP = np.exp(24.4543 - 67.4509 * (100 / TempK) - 4.8489 * np.log(TempK / 100))
    VPCorrWP = np.exp(-0.000544 * Sal)
    VPSWWP = VPWP * VPCorrWP
    VPFac = 1.0 - VPSWWP  # this assumes 1 atmosphere
    return VPFac
