# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate saturation states of soluble solids."""

from autograd import numpy as np
from . import convert


def _deltaKappaCalcite_I75(TempC):
    """Delta and kappa terms for calcite solubility [I75]."""
    # Note that Millero, GCA 1995 has typos:
    #   (-.5304, -.3692, and 10^3 for Kappa factor)
    deltaVKCa = -48.76 + 0.5304 * TempC
    KappaKCa = (-11.76 + 0.3692 * TempC) / 1000
    return deltaVKCa, KappaKCa


def logk_calcite_M83_ts(TempK, Sal):
    """Calcite solubility following M83 but excluding pressure correction."""
    logKCa = -171.9065 - 0.077993 * TempK + 2839.319 / TempK
    logKCa = logKCa + 71.595 * np.log10(TempK)
    logKCa = logKCa + (-0.77712 + 0.0028426 * TempK + 178.34 / TempK) * np.sqrt(Sal)
    logKCa = logKCa - 0.07711 * Sal + 0.0041249 * np.sqrt(Sal) * Sal
    return logKCa


def k_calcite_M83(TempK, Sal, Pbar, RGas):
    """Calcite solubility following M83."""
    logKCa = logk_calcite_M83_ts(TempK, Sal)
    # sd fit = .01 (for Sal part, not part independent of Sal)
    KCa = 10.0**logKCa  # this is in (mol/kg-SW)^2 at zero pressure
    # Add pressure correction for calcite [I75, M79]
    TempC = convert.kelvin_to_celsius(TempK)
    deltaVKCa, KappaKCa = _deltaKappaCalcite_I75(TempC)
    lnKCafac = (-deltaVKCa + 0.5 * KappaKCa * Pbar) * Pbar / (RGas * TempK)
    KCa = KCa * np.exp(lnKCafac)
    return KCa


def k_mgcalcite(TempK, Sal, Pbar, RGas, calcite_Mg_percent):
    """Magnesian calcite solubility for varying compositions."""
    logKCa = logk_calcite_M83_ts(TempK, Sal)
    logKCa_25_0 = logk_calcite_M83_ts(273.15 + 25, 0)
    MgContent = calcite_Mg_percent / 100
    # calculate the correction factor (offset of magnesian calcite compared to calcite)
    # curve 1, biogenic, minimally treated based on [PM74, recalculated TP77]
    coef_bio = [-234.13194855, 85.74778794, -1.61786329, -8.51219119]
    logKMgCa_25_0_bio = (
        coef_bio[0] * MgContent**3
        + coef_bio[1] * MgContent**2
        + coef_bio[2] * MgContent**1
        + coef_bio[3]
    )
    corr_bio = logKCa_25_0 - logKMgCa_25_0_bio
    # curve 2, biogenic, based on  [WM74, B87, BP89]
    coef_bio_treat = [7.86286001, 0.04450687, -8.39289072]
    logKMgCa_25_0_bio_treat = (
        coef_bio_treat[0] * MgContent**2
        + coef_bio_treat[1] * MgContent**1
        + coef_bio_treat[2]
    )
    corr_bio_treat = logKCa_25_0 - logKMgCa_25_0_bio_treat
    # curve 3, synthetic, based on [MM84, B87]
    coef_synth = [3.38134966, 0.54870137, -8.52124947]
    logKMgCa_25_0_synth = (
        coef_synth[0] * MgContent**2 + coef_synth[1] * MgContent**1 + coef_synth[2]
    )
    corr_synth = logKCa_25_0 - logKMgCa_25_0_synth
    # fish calicte [WMG12]:
    corr_fish = logk_calcite_M83_ts(273.15 + 25, 36.5) - (-5.89)
    # calculate logKMgCa
    logKMgCa_bio = logKCa - corr_bio
    logKMgCa_bio_treat = logKCa - corr_bio_treat
    logKMgCa_synth = logKCa - corr_synth
    logKMgCa_fish = logKCa - corr_fish
    # calculate  KMgCa
    KMgCa_bio = 10.0**logKMgCa_bio  # this is in (mol/kg-SW)^2 at zero pressure
    KMgCa_bio_treat = (
        10.0**logKMgCa_bio_treat
    )  # this is in (mol/kg-SW)^2 at zero pressure
    KMgCa_synth = 10.0**logKMgCa_synth  # this is in (mol/kg-SW)^2 at zero pressure
    KMgCa_fish = 10.0**logKMgCa_fish  # this is in (mol/kg-SW)^2 at zero pressure
    # Add pressure correction for Mg-calcite [I75, M79]
    TempC = convert.kelvin_to_celsius(TempK)
    deltaVKCa, KappaKCa = _deltaKappaCalcite_I75(TempC)
    # molar volume correction for Mg content, based on [RB62, PR90, A77]
    # derived from doing a linear fit through experimental data
    # for varying Mg content
    deltaVKMgCa = deltaVKCa + 10.22 * MgContent
    KappaKMgCa = KappaKCa
    lnKMgCafac = (-deltaVKMgCa + 0.5 * KappaKMgCa * Pbar) * Pbar / (RGas * TempK)
    KMgCa_bio = KMgCa_bio * np.exp(lnKMgCafac)
    KMgCa_bio_treat = KMgCa_bio_treat * np.exp(lnKMgCafac)
    KMgCa_synth = KMgCa_synth * np.exp(lnKMgCafac)
    # for fish with fixed Mg content
    deltaVKMgCa_fish = deltaVKCa + 10.22 * 0.479
    lnKMgCafac_fish = (
        (-deltaVKMgCa_fish + 0.5 * KappaKMgCa * Pbar) * Pbar / (RGas * TempK)
    )
    KMgCa_fish = KMgCa_fish * np.exp(lnKMgCafac_fish)
    return KMgCa_bio, KMgCa_bio_treat, KMgCa_synth, KMgCa_fish


def k_aragonite_M83(TempK, Sal, Pbar, RGas):
    """Aragonite solubility following M83 with pressure correction of I75."""
    logKAr = -171.945 - 0.077993 * TempK + 2903.293 / TempK
    logKAr = logKAr + 71.595 * np.log10(TempK)
    logKAr = logKAr + (-0.068393 + 0.0017276 * TempK + 88.135 / TempK) * np.sqrt(Sal)
    logKAr = logKAr - 0.10018 * Sal + 0.0059415 * np.sqrt(Sal) * Sal
    # sd fit = .009 (for Sal part, not part independent of Sal)
    KAr = 10.0**logKAr  # this is in (mol/kg-SW)^2
    # Add pressure correction for aragonite [M79]:
    TempC = convert.kelvin_to_celsius(TempK)
    deltaVKCa, KappaKCa = _deltaKappaCalcite_I75(TempC)
    # Same as Millero, GCA 1995 except for typos (-.5304, -.3692,
    #   and 10^3 for Kappa factor)
    deltaVKAr = deltaVKCa + 2.8
    KappaKAr = KappaKCa
    lnKArfac = (-deltaVKAr + 0.5 * KappaKAr * Pbar) * Pbar / (RGas * TempK)
    KAr = KAr * np.exp(lnKArfac)
    return KAr


@np.errstate(divide="ignore")
def k_calcite_P0_I75(TempK, Sal):
    """Calcite solubility constant following ICHP73/I75 with no pressure correction.
    For use with GEOSECS constants.
    """
    return 0.0000001 * (
        -34.452
        - 39.866 * Sal ** (1 / 3)
        + 110.21 * np.log10(Sal)
        - 0.0000075752 * TempK**2
    )


def k_calcite_I75(TempK, Sal, Pbar, RGas):
    """Calcite solubility constant following ICHP73/I75 with pressure correction.
    For use with GEOSECS constants.
    """
    # === CO2SYS.m comments: =======
    # *** CalculateKCaforGEOSECS:
    # Ingle et al, Marine Chemistry 1:295-307, 1973 is referenced in
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
    # but the fit is actually from Ingle, Marine Chemistry 3:301-319, 1975).
    # This is in (mol/kg-SW)^2
    # ==============================
    KCa = k_calcite_P0_I75(TempK, Sal)
    # Now add pressure correction
    # === CO2SYS.m comments: =======
    # Culberson and Pytkowicz, Limnology and Oceanography 13:403-417, 1968
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982
    # but their paper is not even on this topic).
    # The fits appears to be new in the GEOSECS report.
    # I can't find them anywhere else.
    # ==============================
    TempC = convert.kelvin_to_celsius(TempK)
    KCa = KCa * np.exp((36 - 0.2 * TempC) * Pbar / (RGas * TempK))
    return KCa


def k_aragonite_GEOSECS(TempK, Sal, Pbar, RGas):
    """Aragonite solubility following ICHP73 with no pressure correction.
    For use with GEOSECS constants.
    """
    # === CO2SYS.m comments: =======
    # *** CalculateKArforGEOSECS:
    # Berner, R. A., American Journal of Science 276:713-730, 1976:
    # (quoted in Takahashi et al, GEOSECS Pacific Expedition v. 3, 1982)
    KCa = k_calcite_P0_I75(TempK, Sal)
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
    TempC = convert.kelvin_to_celsius(TempK)
    KAr = KAr * np.exp((33.3 - 0.22 * TempC) * Pbar / (RGas * TempK))
    return KAr


def CaCO3(CARB, totals, Ks):
    """Calculate calcite and aragonite solubility.

    This calculates omega, the solubility ratio, for calcite and aragonite.
    This is defined by: Omega = [CO3--]*[Ca++]/Ksp,
          where Ksp is the solubility product (either KCa or KAr).
    These are from: M83, I75, M79, ICHP73, B76, TWB82 and CP68.

    Based on CaSolubility, version 01.05, 05-23-97, written by Ernie Lewis.
    """
    OmegaCa = CARB * totals["TCa"] / Ks["KCa"]
    OmegaAr = CARB * totals["TCa"] / Ks["KAr"]
    return OmegaCa, OmegaAr


def MgCaCO3(CARB, totals, Ks, calcite_Mg_percent):
    """Calculate Mg-calcite solubility.
    This is defined by: Omega = [CO3--]*[Ca++]**(1-x)*[Mg++]**(x)/Ksp,
          where Ksp is the solubility product of Mg-calcite
          and x is the mole fraction of Mg++.
    from Woosley et al. (2012), https://doi.org/10.1029/2011JC007599
    """
    MgContent = calcite_Mg_percent / 100
    OmegaMgCa_bio = (
        CARB
        * totals["TCa"] ** (1 - MgContent)
        * totals["TMg"] ** MgContent
        / Ks["KMgCa_bio"]
    )
    OmegaMgCa_bio_treat = (
        CARB
        * totals["TCa"] ** (1 - MgContent)
        * totals["TMg"] ** MgContent
        / Ks["KMgCa_bio_treat"]
    )
    OmegaMgCa_synth = (
        CARB
        * totals["TCa"] ** (1 - MgContent)
        * totals["TMg"] ** MgContent
        / Ks["KMgCa_synth"]
    )
    OmegaMgCa_fish = (
        CARB * totals["TCa"] ** (1 - 0.479) * totals["TMg"] ** 0.479 / Ks["KMgCa_fish"]
    )
    return OmegaMgCa_bio, OmegaMgCa_bio_treat, OmegaMgCa_synth, OmegaMgCa_fish


def CARB_from_OC(OC, totals, Ks):
    """Calculate [CO3] given saturation state w.r.t. calcite."""
    return OC * Ks["KCa"] / totals["TCa"]


def CARB_from_OA(OA, totals, Ks):
    """Calculate [CO3] given saturation state w.r.t. aragonite."""
    return OA * Ks["KAr"] / totals["TCa"]
