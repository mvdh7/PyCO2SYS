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


def get_kt_Mg_calcite_25C_1atm_synthetic(Mg_content):
    # curve 3
    a, b, c, d = -1.047e-05, 0.0008416, -0.001061, -8.502
    return 10 ** (a * Mg_content**3 + b**Mg_content**2 + c * Mg_content + d)


def get_kt_Mg_calcite_25C_1atm_biogenic(Mg_content):
    # curve 2
    a, b, c, d = -8.968e-06, 0.0006153, 0.002346, -8.377
    return 10 ** (a * Mg_content**3 + b**Mg_content**2 + c * Mg_content + d)


def get_kt_Mg_calcite_25C_1atm_minprep(Mg_content):
    # curve 1
    a, b, c, d = -0.0002341, 0.008575, -0.01618, -8.512
    return 10 ** (a * Mg_content**3 + b**Mg_content**2 + c * Mg_content + d)


def get_deltaH_Mg_calcite(Mg_content):
    # enthalpy deltaH based on Mg content
    # linear line between dH = -13.07 calcite (0% Mg) and dH = -48.19 of magnesite (100% Mg)
    deltaH_c_25 = -13.07  # NBS
    deltaH_mag_25 = -48.19  # NBS
    return deltaH_c_25 - (deltaH_c_25 - deltaH_mag_25) / 100 * Mg_content


def get_Cp_Mg_calcite(Mg_content):
    # heat capacity Cp based on Mg content
    # I think it could potentially be deleted, makes little difference
    # linear line between Cp_25 of calcite and Cp_25 of Magnesite
    cp_c_25 = 81.88  # NBS
    cp_mag_25 = 75.52  # NBS
    return cp_c_25 - (cp_c_25 - cp_mag_25) / 100 * Mg_content


def get_kt_calcite_1atm_PB82(temperature):
    TempK = convert.celsius_to_kelvin(temperature)
    # temperature dependence of K_calcite according to PB82
    return 10 ** (
        -171.9065 - 0.077993 * TempK + 2839.319 / TempK + 71.595 * np.log10(TempK)
    )


def get_kt_Mg_calcite_1atm_vantHoff(
    temperature, Mg_content, gas_constant, kt_Mg_calcite_25C_1atm
):
    TempK = convert.celsius_to_kelvin(temperature)
    T0 = 298.15  # standard temperature is 25C
    kt_Mg_calcite_1atm = (
        np.exp(
            (
                (-get_deltaH_Mg_calcite(Mg_content) * 1000)
                / gas_constant
                * (1 / (TempK) - 1 / (T0))
            )
            - get_Cp_Mg_calcite(Mg_content)
            / gas_constant
            * (np.log(TempK / T0) + T0 / TempK - 1)
        )
        * kt_Mg_calcite_25C_1atm
    )
    return kt_Mg_calcite_1atm


def get_kt_Mg_calcite_1atm_PB82(temperature, kt_Mg_calcite_25C_1atm):
    # uses temperature dependence of logK(calcite from PB82)
    # logK_c_t needs to be calculated somewhere
    # logK_c_25_0 = Ksp of calcite at 25C, constant
    np.log10(kt_Mg_calcite_25C_1atm)
    logK_c_25_0 = -8.479
    logK_mg_t_0 = (np.log10(kt_Mg_calcite_25C_1atm) - logK_c_25_0) + np.log10(
        get_kt_calcite_1atm_PB82(temperature)
    )
    kt_Mg_calcite_1atm = 10 ** (logK_mg_t_0)
    return kt_Mg_calcite_1atm


# parameters for different ions,
# gamma_caco3 = 1
gamma_params_Ca = [
    7.85297464e-02,
    2.27643150e00,
    1.08708941e01,
    2.69256372e02,
    -6.00600118e01,
    2.34393707e-01,
    7.47893672e-03,
    8.44162753e-02,
    2.28471350e00,
    3.39037059e-01,
    4.22637079e02,
    1.64492453e02,
    7.54535042e-03,
    -2.49687587e-02,
    3.05849724e00,
]
gamma_params_Mg = [
    5.75686680e-02,
    3.10718154e00,
    2.50402952e01,
    -5.72765166e01,
    -1.19322731e05,
    -1.85585651e05,
    3.77915772e-04,
    2.96865403e-03,
    3.50767071e00,
    -2.49157790e-05,
    5.79697618e02,
    6.29759954e02,
    2.34258794e-03,
    -6.00823112e-03,
    7.08741293e00,
]
gamma_params_CO3 = [
    -6.60736728e-02,
    1.93338610e00,
    3.45352367e00,
    2.03940328e02,
    -4.73054165e03,
    1.04868555e02,
    1.69091249e-04,
    1.46377440e-02,
    3.89712262e00,
    7.83948823e-01,
    2.41731438e00,
    5.39490714e00,
    4.08298567e-03,
    -5.80998630e-03,
    1.07123040e01,
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
    return _get_activity_coefficient(salinity, temperature, gamma_params_Ca)


def get_activity_coefficient_Mg(salinity, temperature):
    return _get_activity_coefficient(salinity, temperature, gamma_params_Mg)


def get_activity_coefficient_CO3(salinity, temperature):
    return _get_activity_coefficient(salinity, temperature, gamma_params_CO3)


def get_k_Mg_calcite_1atm(
    gamma_Ca, gamma_Mg, gamma_CO3, Mg_content, kt_Mg_calcite_1atm
):
    # calculate stoichiometric K*
    k_Mg_calcite_1atm = kt_Mg_calcite_1atm / (
        gamma_Ca ** (1 - Mg_content) * gamma_Mg**Mg_content * gamma_CO3
    )
    return k_Mg_calcite_1atm


def get_k_Mg_calcite(
    pressure, temperature, Mg_content, gas_constant, k_Mg_calcite_1atm
):
    TempK = convert.celsius_to_kelvin(temperature)
    deltaV_ca, deltaK_ca = _deltaKappaCalcite_I75(temperature)  # from PyCO2Sys
    # get ∆V for Mg content
    deltaV_mg = deltaV_ca + 10.22 * Mg_content  # sources for this fit: RB62, PR90, A77
    # get ∆K
    deltaK_mg = deltaK_ca
    # calculate pressure dependence
    ln_K_K = (
        (-deltaV_mg + 0.5 * deltaK_mg * pressure) * pressure / (gas_constant * TempK)
    )
    k_Mg_calcite = k_Mg_calcite_1atm * np.exp(ln_K_K)
    return k_Mg_calcite
