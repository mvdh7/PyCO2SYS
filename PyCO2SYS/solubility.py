# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
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

@valid(Mg_percent=[0,22])
def get_kt_Mg_calcite_25C_1atm_synthetic(Mg_percent):
    # curve 3
    a, b, c, d = -1.04700900e-05,  8.41626295e-04, -1.06073211e-03, -8.50230163e+00
    return 10 ** (a * Mg_percent**3 + b*Mg_percent**2 + c * Mg_percent + d)

@valid(Mg_percent=[0,45])
def get_kt_Mg_calcite_25C_1atm_biogenic(Mg_percent):
    # curve 2
    a, b, c, d = -9.56525687e-06,  6.34456760e-04,  2.18346079e-03, -8.37659138e+00
    return 10 ** (a * Mg_percent**3 + b*Mg_percent**2 + c * Mg_percent + d)


@valid(Mg_percent=[0,27])
def get_kt_Mg_calcite_25C_1atm_minprep(Mg_percent):
    # curve 1
    a, b, c, d = -2.34131949e-04,  8.57477879e-03, -1.61786329e-02, -8.51219119e+00
    return 10 ** (a * Mg_percent**3 + b*Mg_percent**2 + c * Mg_percent + d)



def _get_deltaH_Mg_calcite(Mg_percent):
    # enthalpy deltaH based on Mg content
    # ideal solid solution line between dH_calcite (0% Mg) and dH_magnesite (100% Mg)
    deltaH_c_25 = -13.07  # NBS
    deltaH_mag_25 = -28.9  # Robie1995
    return deltaH_c_25 - (deltaH_c_25 - deltaH_mag_25) / 100 * Mg_percent


def _get_Cp_Mg_calcite(Mg_percent):
    # heat capacity Cp based on Mg content
    # I think it could potentially be deleted, makes little difference
    # linear line between Cp_25 of calcite and Cp_25 of Magnesite
    cp_c_25 = 81.88  # NBS
    cp_mag_25 = 75.52  # NBS
    return cp_c_25 - (cp_c_25 - cp_mag_25) / 100 * Mg_percent


def _get_kt_calcite_1atm_PB82(temperature):
    TempK = convert.celsius_to_kelvin(temperature)
    # temperature dependence of K_calcite according to PB82
    return 10 ** (
        -171.9065 - 0.077993 * TempK + 2839.319 / TempK + 71.595 * np.log10(TempK)
    )

def _get_kt_magnesite_1atm_B18(temperature):
    TempK = convert.celsius_to_kelvin(temperature)
    # temperature dependence of K_calcite according to B18
    return 10 ** (
        7.267 - 1476.604 / TempK - 0.033918 * TempK
    )

def _get_kt_calcite_magnesite_idealmix_1atm(temperature, Mg_percent):
    ideal_mix = np.log10(_get_kt_magnesite_1atm_B18(temperature))*Mg_percent/100 + (
        np.log10(_get_kt_calcite_1atm_PB82(temperature))*(1-(Mg_percent/100)))
    return 10 ** (ideal_mix)


def get_kt_Mg_calcite_1atm_vantHoff(
    temperature, Mg_percent, gas_constant, kt_Mg_calcite_25C_1atm
):
    GasR = gas_constant / 10
    TempK = convert.celsius_to_kelvin(temperature)
    T0 = 298.15  # standard temperature is 25C
    kt_Mg_calcite_1atm = (
        np.exp(
            (
                (-_get_deltaH_Mg_calcite(Mg_percent) * 1000)
                / GasR
                * (1 / (TempK) - 1 / (T0))
            )
            - _get_Cp_Mg_calcite(Mg_percent)
            / GasR
            * (np.log(TempK / T0) + T0 / TempK - 1)
        )
        * kt_Mg_calcite_25C_1atm
    )
    return kt_Mg_calcite_1atm


def get_kt_Mg_calcite_1atm_idealmix(temperature, kt_Mg_calcite_25C_1atm, Mg_percent):
    # uses temperature dependence of logK(calcite) from PB82
    # uses temperature dependence of logK(magnesite) from B18
    # assumes ideal solid solution
    delta = np.log10(_get_kt_calcite_magnesite_idealmix_1atm(25, Mg_percent)) - (
        np.log10(kt_Mg_calcite_25C_1atm))
    kt_Mg_calcite_1atm = 10 ** (np.log10(_get_kt_calcite_magnesite_idealmix_1atm(temperature, Mg_percent)) 
                                - delta )
    return kt_Mg_calcite_1atm


def get_kt_Mg_calcite_1atm_PB82(temperature, kt_Mg_calcite_25C_1atm):
    # uses temperature dependence of logK(calcite) from PB82
    delta = np.log10(_get_kt_calcite_1atm_PB82(25)) - (
        np.log10(kt_Mg_calcite_25C_1atm))
    kt_Mg_calcite_1atm = 10 ** (np.log10(_get_kt_calcite_1atm_PB82(temperature)) 
                                - delta)
    return kt_Mg_calcite_1atm


# parameters for different ions,
# acf_caco3 = 1
acf_params_Ca = [
    7.76951341e-02,
    2.28137488e+00,
    1.08980433e+01,
    2.69207962e+02,
    -5.82525447e+01,
    1.27866112e-01,
    7.54130862e-03,
    8.44820348e-02,
    2.27443209e+00,
    4.67107098e-01,
    3.70905684e+02,
    1.50378527e+02,
    7.52749248e-03,
   -2.52416954e-02,
    3.08064065e+00,
]
acf_params_Mg = [
    -8.99520485e+00,
    2.94298452e+04,
    3.41365638e+03,
    4.46735197e+02,
    4.10581581e+04,
   -1.61855534e+02,
    4.72310071e-02,
    1.46211671e-01,
    4.46262145e+00,
   -5.14957487e+01,
    3.31316128e+05,
    6.10039215e+03,
    1.09121032e-03,
   -1.98622574e-04,
   -8.11411910e-01
]
acf_params_CO3 = [
    -1.56682788e-02,
    7.71409406e+01,
    7.76810062e+00,
    -5.07318950e+03,
    -9.51872681e+08,
    1.36830277e+09,
    -1.85870465e-07,
    1.91965700e-01,
    6.41576790e+00,
    1.37088945e+00,
    -1.62853568e+03,
    -5.20524907e+03,
    2.00256489e-03,
    -5.67468044e-03,
    7.98885950e+00
]


def _get_activity_coefficient(salinity, temperature, params):
    TempK = convert.celsius_to_kelvin(temperature)
    # get activity coefficients
    a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2, e0, e1, e2 = params
    A = a0 + a1 * (1 / (salinity + a2))
    B = b0 + b1 * (1 / (salinity + b2))
    C = c0 + c1 * (1 / (salinity + c2))
    D = d0 + d1 * (1 / (salinity + d2))
    E = e0 + e1 * (1 / (salinity + e2))
    return A - C * (TempK - B) ** (1 / D) * np.log(E * (TempK - B))


def get_activity_coefficient_Ca(salinity, temperature):
    return _get_activity_coefficient(salinity, temperature, acf_params_Ca)


def get_activity_coefficient_Mg(salinity, temperature):
    return _get_activity_coefficient(salinity, temperature, acf_params_Mg)


def get_activity_coefficient_CO3(salinity, temperature):
    return _get_activity_coefficient(salinity, temperature, acf_params_CO3)


def get_k_Mg_calcite_1atm(acf_Ca, acf_Mg, acf_CO3, Mg_percent, kt_Mg_calcite_1atm, gas_constant):
    # calculate stoichiometric K*
    Mg_fraction = Mg_percent / 100
    k_Mg_calcite_1atm = kt_Mg_calcite_1atm / (
        acf_Ca ** (1 - Mg_fraction) * acf_Mg**Mg_fraction * acf_CO3
    )
    return k_Mg_calcite_1atm


def get_k_Mg_calcite(
    pressure, temperature, Mg_percent, gas_constant, k_Mg_calcite_1atm
):
    TempK = convert.celsius_to_kelvin(temperature)
    Pbar = convert.decibar_to_bar(pressure)
    deltaV_Ca, deltaK_Ca = _deltaKappaCalcite_I75(temperature)  # from PyCO2Sys
    # get ∆V for Mg content
    deltaV_Mg_calcite = deltaV_Ca + 0.1022 * Mg_percent  # sources for this fit: RB62, PR90, A77
    # get ∆K
    deltaK_Mg_calcite = deltaK_Ca
    # calculate pressure dependence
    ln_K_K = (-deltaV_Mg_calcite + 0.5 * deltaK_Mg_calcite * Pbar) * Pbar / (gas_constant * TempK)
    k_Mg_calcite = k_Mg_calcite_1atm * np.exp(ln_K_K)
    return k_Mg_calcite


def OMgCaCO3_from_CO3(Ca, Mg, CO3, Mg_percent, k_Mg_calcite):
    Mg_fraction = Mg_percent / 100
    return 1e-12 * (CO3 * Ca ** (1 - Mg_fraction) * Mg ** Mg_fraction) / k_Mg_calcite
