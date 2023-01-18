# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate equilibrium constants from temperature, salinity and pressure."""

from autograd import numpy as np
from . import p1atm, pcx, pressured
from .. import constants, convert, gas, solubility

__all__ = ["p1atm", "pcx", "pressured"]


def prepare(temperature, pressure, equilibria):
    """Initialise equilibria dict if needed and convert temperature/pressure units to
    Kelvin/bar."""
    temperature_K = convert.celsius_to_kelvin(temperature)
    pressure_bar = convert.decibar_to_bar(pressure)
    if equilibria is None:
        equilibria = {}
    return temperature_K, pressure_bar, equilibria


def assemble(
    temperature,
    pressure,
    totals,
    opt_pH_scale,
    opt_k_carbonic,
    opt_k_bisulfate,
    opt_k_fluoride,
    opt_gas_constant,
    Ks=None,
    pressure_atmosphere=1.0,
    opt_pressured_kCO2=0,
):
    """Evaluate all stoichiometric equilibrium constants, converted to the chosen pH
    scale, and corrected for pressure.

    Inputs must first be conditioned.

    This finds the Constants of the CO2 system in seawater or freshwater,
    corrects them for pressure, and reports them on the chosen pH scale.
    The process is as follows: the Constants (except k_constants, KF which stay on the
    free scale - these are only corrected for pressure) are:
          1) evaluated as they are given in the literature,
          2) converted to the SWS scale in mol/kg-SW or to the NBS scale,
          3) corrected for pressure,
          4) converted to the SWS pH scale in mol/kg-SW,
          5) converted to the chosen pH scale.

    Based on a subset of Constants, version 04.01, 10-13-97, by Ernie Lewis.
    """
    k_constants = Ks  # temporary fix
    temperature_K, pressure_bar, k_constants = prepare(
        temperature, pressure, k_constants
    )
    salinity = totals["Sal"]
    # Set ideal gas constant
    if "gas_constant" not in k_constants:
        k_constants["RGas"] = constants.RGasConstant(opt_gas_constant)
    gas_constant = k_constants["RGas"]
    # Get KSO4 and KF, at pressure, and always on the Free pH scale
    if "KSO4" not in k_constants:
        k_constants["KSO4"] = pressured.KSO4(
            temperature_K, salinity, pressure_bar, gas_constant, opt_k_bisulfate
        )
    if "KF" not in k_constants:
        k_constants["KF"] = pressured.KF(
            temperature_K, salinity, pressure_bar, gas_constant, opt_k_fluoride
        )
    # Depressurise KSO4 and KF for pH scale conversions
    if "KSO4_P0" not in k_constants:
        k_constants["KSO4_P0"] = k_constants["KSO4"] / pcx.KSO4fac(
            temperature_K, pressure_bar, gas_constant
        )
    if "KF_P0" not in k_constants:
        k_constants["KF_P0"] = k_constants["KF"] / pcx.KFfac(
            temperature_K, pressure_bar, gas_constant
        )
    # Correct pH scale conversion factors for pressure.
    # Note that fH has been assumed to be independent of pressure.
    # The values KSO4 and KF are already now pressure-corrected, so the pH scale
    # conversions are now valid at pressure.
    # Find pH scale conversion factor: this is the scale they will be put on
    k_constants = convert.get_pHfactor_from_SWS(
        temperature_K, salinity, totals, k_constants, opt_pH_scale, opt_k_carbonic
    )
    pHfactor = k_constants["pHfactor_from_SWS"]  # for convenience
    SWStoTOT_P0 = convert.pH_sws_to_total_P0(
        temperature_K, totals, k_constants, opt_k_bisulfate, opt_k_fluoride
    )
    # Borate
    if "KB" not in k_constants:
        k_constants["KB"] = (
            pressured.KB(
                temperature_K,
                salinity,
                pressure_bar,
                gas_constant,
                opt_k_carbonic,
                k_constants["fH"],
                SWStoTOT_P0,
            )
            * pHfactor
        )
    # Water
    if "KW" not in k_constants:
        k_constants["KW"] = (
            pressured.KW(
                temperature_K, salinity, pressure_bar, gas_constant, opt_k_carbonic
            )
            * pHfactor
        )
    # Phosphate
    if (
        ("KP1" not in k_constants)
        or ("KP2" not in k_constants)
        or ("KP3" not in k_constants)
    ):
        KP1, KP2, KP3 = pressured.KP(
            temperature_K,
            salinity,
            pressure_bar,
            gas_constant,
            opt_k_carbonic,
            k_constants["fH"],
        )
        if "KP1" not in k_constants:
            k_constants["KP1"] = KP1 * pHfactor
        if "KP2" not in k_constants:
            k_constants["KP2"] = KP2 * pHfactor
        if "KP3" not in k_constants:
            k_constants["KP3"] = KP3 * pHfactor
    # Silicate
    if "KSi" not in k_constants:
        k_constants["KSi"] = (
            pressured.KSi(
                temperature_K,
                salinity,
                pressure_bar,
                gas_constant,
                opt_k_carbonic,
                k_constants["fH"],
            )
            * pHfactor
        )
    # Carbonate
    if ("K1" not in k_constants) or ("K2" not in k_constants):
        K1, K2 = pressured.KC(
            temperature_K,
            salinity,
            pressure_bar,
            gas_constant,
            opt_k_carbonic,
            k_constants["fH"],
            SWStoTOT_P0,
        )
        if "K1" not in k_constants:
            k_constants["K1"] = K1 * pHfactor
        if "K2" not in k_constants:
            k_constants["K2"] = K2 * pHfactor
    # Sulfide
    if "KH2S" not in k_constants:
        k_constants["KH2S"] = (
            pressured.KH2S(
                temperature_K,
                salinity,
                pressure_bar,
                gas_constant,
                opt_k_carbonic,
                SWStoTOT_P0,
            )
            * pHfactor
        )
    # Ammonium
    if "KNH3" not in k_constants:
        k_constants["KNH3"] = (
            pressured.KNH3(
                temperature_K,
                salinity,
                pressure_bar,
                gas_constant,
                opt_k_carbonic,
                SWStoTOT_P0,
            )
            * pHfactor
        )
    # K0 for CO2 dissolution
    if "K0" not in k_constants:
        k_constants["K0"] = np.where(
            opt_pressured_kCO2 == 1,
            pressured.kCO2_W74(
                temperature_K, salinity, pressure_bar, gas_constant, pressure_atmosphere
            ),
            p1atm.kCO2_W74(temperature_K, salinity),
        )
    if "FugFac" not in k_constants:
        k_constants["FugFac"] = gas.fugacity_factor(
            temperature,
            opt_k_carbonic,
            gas_constant,
            pressure_bar,
            pressure_atmosphere=pressure_atmosphere,
            opt_pressured_kCO2=opt_pressured_kCO2,
        )
    if "VPFac" not in k_constants:
        k_constants["VPFac"] = gas.vpfactor(
            temperature, salinity, pressure_atmosphere=pressure_atmosphere
        )
    k_constants = convert.get_pHfactor_to_Free(
        temperature_K, salinity, totals, k_constants, opt_pH_scale, opt_k_carbonic
    )
    # Aragonite and calcite solubility products
    if "KAr" not in k_constants:
        k_constants["KAr"] = np.where(
            (opt_k_carbonic == 6) | (opt_k_carbonic == 7),  # GEOSECS values
            solubility.k_aragonite_GEOSECS(
                temperature_K, salinity, pressure_bar, gas_constant
            ),
            solubility.k_aragonite_M83(
                temperature_K, salinity, pressure_bar, gas_constant
            ),
        )
    if "KCa" not in k_constants:
        k_constants["KCa"] = np.where(
            (opt_k_carbonic == 6) | (opt_k_carbonic == 7),  # GEOSECS values
            solubility.k_calcite_I75(
                temperature_K, salinity, pressure_bar, gas_constant
            ),
            solubility.k_calcite_M83(
                temperature_K, salinity, pressure_bar, gas_constant
            ),
        )
    # Extra alkalinity components
    if "k_alpha" not in k_constants:
        k_constants["k_alpha"] = 1e-7
    if "k_beta" not in k_constants:
        k_constants["k_beta"] = 1e-7
    return k_constants
