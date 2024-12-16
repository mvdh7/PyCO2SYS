# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate saturation states of soluble solids."""

from jax import numpy as np

from . import convert


def _deltaKappaCalcite_I75(temperature):
    """Delta and kappa terms for calcite solubility [I75]."""
    # Note that Millero, GCA 1995 has typos:
    #   (-.5304, -.3692, and 10^3 for Kappa factor)
    deltaVKCa = -48.76 + 0.5304 * temperature
    KappaKCa = (-11.76 + 0.3692 * temperature) / 1000
    return deltaVKCa, KappaKCa


def k_calcite_M83(temperature, salinity, pressure, gas_constant):
    """Calcite solubility following M83."""
    TempK = convert.celsius_to_kelvin(temperature)
    Pbar = convert.decibar_to_bar(pressure)
    logKCa = -171.9065 - 0.077993 * TempK + 2839.319 / TempK
    logKCa = logKCa + 71.595 * np.log10(TempK)
    logKCa = logKCa + (-0.77712 + 0.0028426 * TempK + 178.34 / TempK) * np.sqrt(
        salinity
    )
    logKCa = logKCa - 0.07711 * salinity + 0.0041249 * np.sqrt(salinity) * salinity
    # sd fit = .01 (for salinity part, not part independent of salinity)
    KCa = 10.0**logKCa  # this is in (mol/kg-SW)^2 at zero pressure
    # Add pressure correction for calcite [I75, M79]
    deltaVKCa, KappaKCa = _deltaKappaCalcite_I75(temperature)
    lnKCafac = (-deltaVKCa + 0.5 * KappaKCa * Pbar) * Pbar / (gas_constant * TempK)
    KCa = KCa * np.exp(lnKCafac)
    return KCa


def k_aragonite_M83(temperature, salinity, pressure, gas_constant):
    """Aragonite solubility following M83 with pressure correction of I75."""
    TempK = convert.celsius_to_kelvin(temperature)
    Pbar = convert.decibar_to_bar(pressure)
    logKAr = -171.945 - 0.077993 * TempK + 2903.293 / TempK
    logKAr = logKAr + 71.595 * np.log10(TempK)
    logKAr = logKAr + (-0.068393 + 0.0017276 * TempK + 88.135 / TempK) * np.sqrt(
        salinity
    )
    logKAr = logKAr - 0.10018 * salinity + 0.0059415 * np.sqrt(salinity) * salinity
    # sd fit = .009 (for salinity part, not part independent of salinity)
    KAr = 10.0**logKAr  # this is in (mol/kg-SW)^2
    # Add pressure correction for aragonite [M79]:
    deltaVKCa, KappaKCa = _deltaKappaCalcite_I75(temperature)
    # Same as Millero, GCA 1995 except for typos (-.5304, -.3692,
    #   and 10^3 for Kappa factor)
    deltaVKAr = deltaVKCa + 2.8
    KappaKAr = KappaKCa
    lnKArfac = (-deltaVKAr + 0.5 * KappaKAr * Pbar) * Pbar / (gas_constant * TempK)
    KAr = KAr * np.exp(lnKArfac)
    return KAr


@np.errstate(divide="ignore")
def k_calcite_P0_I75(temperature, salinity):
    """Calcite solubility constant following ICHP73/I75 with no pressure correction.
    For use with GEOSECS constants.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    return 0.0000001 * (
        -34.452
        - 39.866 * salinity ** (1 / 3)
        + 110.21 * np.log10(salinity)
        - 0.0000075752 * TempK**2
    )


def k_calcite_I75(temperature, salinity, pressure, gas_constant):
    """Calcite solubility constant following ICHP73/I75 with pressure correction.
    For use with GEOSECS constants.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    Pbar = convert.decibar_to_bar(pressure)
    # === CO2SYS.m comments: =======
    # *** CalculateKCaforGEOSECS:
    # Ingle et al, Marine Chemistry 1:295-307, 1973 is referenced in
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
    # but the fit is actually from Ingle, Marine Chemistry 3:301-319, 1975).
    # This is in (mol/kg-SW)^2
    # ==============================
    KCa = k_calcite_P0_I75(temperature, salinity)
    # Now add pressure correction
    # === CO2SYS.m comments: =======
    # Culberson and Pytkowicz, Limnology and Oceanography 13:403-417, 1968
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
    # but their paper is not even on this topic).
    # The fits appears to be new in the GEOSECS report.
    # I can't find them anywhere else.
    # ==============================
    KCa = KCa * np.exp((36 - 0.2 * temperature) * Pbar / (gas_constant * TempK))
    return KCa


def k_aragonite_GEOSECS(temperature, salinity, pressure, gas_constant):
    """Aragonite solubility following ICHP73 with no pressure correction.
    For use with GEOSECS constants.
    """
    TempK = convert.celsius_to_kelvin(temperature)
    Pbar = convert.decibar_to_bar(pressure)
    # === CO2SYS.m comments: =======
    # *** CalculateKArforGEOSECS:
    # Berner, R. A., American Journal of Science 276:713-730, 1976:
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
    KCa = k_calcite_P0_I75(temperature, salinity)
    KAr = 1.45 * KCa  # this is in (mol/kg-SW)^2
    # Berner (p. 722) states that he uses 1.48.
    # It appears that 1.45 was used in the GEOSECS calculations
    # Now add pressure correction
    # === CO2SYS.m comments: =======
    # Culberson and Pytkowicz, Limnology and Oceanography 13:403-417, 1968
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
    # but their paper is not even on this topic).
    # The fits appears to be new in the GEOSECS report.
    # I can't find them anywhere else.
    KAr = KAr * np.exp((33.3 - 0.22 * temperature) * Pbar / (gas_constant * TempK))
    return KAr


def OC_from_CO3(CO3, Ca, k_calcite):
    """Calculate [CO3] given saturation state w.r.t. calcite."""
    return 1e-12 * CO3 * Ca / k_calcite


def OA_from_CO3(CO3, Ca, k_aragonite):
    """Calculate [CO3] given saturation state w.r.t. aragonite."""
    return 1e-12 * CO3 * Ca / k_aragonite


def CO3_from_OC(saturation_calcite, Ca, k_calcite):
    """Calculate [CO3] given saturation state w.r.t. calcite."""
    return 1e12 * saturation_calcite * k_calcite / Ca


def CO3_from_OA(saturation_aragonite, Ca, k_aragonite):
    """Calculate [CO3] given saturation state w.r.t. aragonite."""
    return 1e12 * saturation_aragonite * k_aragonite / Ca
