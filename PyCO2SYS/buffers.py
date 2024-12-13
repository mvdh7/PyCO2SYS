# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate various buffer factors of the marine carbonate system."""

from jax import numpy as np

from . import solubility, solve
from .meta import egrad

ilog10e = -1 / np.log10(np.exp(1))  # multiplier to convert pH to ln(H)


def d_lnOmega__d_CO3(CO3, Ca, k_calcite):
    """Function for d[ln(Omega)]/d[CO3].  Identical for calcite and aragonite."""
    return egrad(lambda CO3: np.log(solubility.OC_from_CO3(CO3, Ca, k_calcite)))(CO3)


def d_dic__d_pH__alkalinity(
    alkalinity,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    k_H2O,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
):
    alkalinity, pH = np.broadcast_arrays(
        alkalinity,
        pH,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        k_H2O,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
    )[:2]
    return egrad(
        lambda pH: solve.get.inorganic.dic_from_alkalinity_pH(
            alkalinity,
            pH,
            opt_to_free,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            k_H2O,
            k_H2CO3,
            k_HCO3,
            k_BOH3,
            k_H3PO4,
            k_H2PO4,
            k_HPO4,
            k_Si,
            k_NH3,
            k_H2S,
            k_HSO4_free,
            k_HF_free,
        )
    )(pH)


def d_alkalinity__d_pH__dic(
    dic,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    k_H2O,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
):
    dic, pH = np.broadcast_arrays(
        dic,
        pH,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        k_H2O,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
    )[:2]
    return egrad(
        lambda pH: solve.get.inorganic.alkalinity_from_dic_pH(
            dic,
            pH,
            opt_to_free,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            k_H2O,
            k_H2CO3,
            k_HCO3,
            k_BOH3,
            k_H3PO4,
            k_H2PO4,
            k_HPO4,
            k_Si,
            k_NH3,
            k_H2S,
            k_HSO4_free,
            k_HF_free,
        )
    )(pH)


def d_lnCO2__d_pH__alkalinity(
    alkalinity,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    k_CO2,
    k_H2O,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
):
    alkalinity, pH = np.broadcast_arrays(
        alkalinity,
        pH,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        k_CO2,
        k_H2O,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
    )[:2]
    return egrad(
        lambda pH: np.log(
            k_CO2
            * solve.get.inorganic.fCO2_from_alkalinity_pH(
                alkalinity,
                pH,
                opt_to_free,
                total_borate,
                total_phosphate,
                total_silicate,
                total_ammonia,
                total_sulfide,
                total_sulfate,
                total_fluoride,
                k_CO2,
                k_H2O,
                k_H2CO3,
                k_HCO3,
                k_BOH3,
                k_H3PO4,
                k_H2PO4,
                k_HPO4,
                k_Si,
                k_NH3,
                k_H2S,
                k_HSO4_free,
                k_HF_free,
            )
        )
    )(pH)


def d_lnCO2__d_pH__dic(dic, pH, k_CO2, k_H2CO3, k_HCO3):
    dic, pH = np.broadcast_arrays(dic, pH, k_CO2, k_H2CO3, k_HCO3)[:2]
    return egrad(
        lambda pH: np.log(
            k_CO2
            * solve.get.inorganic.fCO2_from_dic_pH(dic, pH, k_CO2, k_H2CO3, k_HCO3)
        )
    )(pH)


def d_CO3__d_pH__alkalinity(
    alkalinity,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    k_H2O,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
):
    alkalinity, pH = np.broadcast_arrays(
        alkalinity,
        pH,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        k_H2O,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
    )[:2]
    return egrad(
        lambda pH: solve.get.inorganic.CO3_from_alkalinity_pH(
            alkalinity,
            pH,
            opt_to_free,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            k_H2O,
            k_H2CO3,
            k_HCO3,
            k_BOH3,
            k_H3PO4,
            k_H2PO4,
            k_HPO4,
            k_Si,
            k_NH3,
            k_H2S,
            k_HSO4_free,
            k_HF_free,
        )
    )(pH)


def d_CO3__d_pH__dic(dic, pH, k_H2CO3, k_HCO3):
    dic, pH = np.broadcast_arrays(dic, pH, k_H2CO3, k_HCO3)[:2]
    return egrad(
        lambda pH: solve.get.inorganic.CO3_from_dic_pH(dic, pH, k_H2CO3, k_HCO3)
    )(pH)


def gamma_dic(d_dic__d_pH__alkalinity, d_lnCO2__d_pH__alkalinity):
    return d_dic__d_pH__alkalinity / d_lnCO2__d_pH__alkalinity


def gamma_alkalinity(d_alkalinity__d_pH__dic, d_lnCO2__d_pH__dic):
    return d_alkalinity__d_pH__dic / d_lnCO2__d_pH__dic


def beta_dic(d_dic__d_pH__alkalinity):
    return d_dic__d_pH__alkalinity / ilog10e


def beta_alkalinity(d_alkalinity__d_pH__dic):
    return d_alkalinity__d_pH__dic / ilog10e


def omega_dic(d_dic__d_pH__alkalinity, d_CO3__d_pH__alkalinity, d_lnOmega__d_CO3):
    return d_dic__d_pH__alkalinity / (d_lnOmega__d_CO3 * d_CO3__d_pH__alkalinity)


def omega_alkalinity(d_alkalinity__d_pH__dic, d_CO3__d_pH__dic, d_lnOmega__d_CO3):
    return d_alkalinity__d_pH__dic / (d_lnOmega__d_CO3 * d_CO3__d_pH__dic)


def all_ESM10(TA, TC, PH, CARB, Sal, TempK, Pbar, totals, Ks, WhichKs):
    """Get all ESM10 buffer factors with automatic differentiation.

    This is more efficient than calculating each one separately because the number of
    automatic differentiation steps is minimised.
    """
    # Get the pH differentials
    dTC_dPH__TA = egrad(lambda PH: solve.get.TCfromTApH(TA, PH, totals, Ks))(PH)
    dTA_dPH__TC = egrad(lambda PH: solve.get.TAfromTCpH(TC, PH, totals, Ks))(PH)
    # gammaTC is (d[ln(CO2)]/d[TC])^-1 with constant TA, i.e. γ_DIC of ESM10
    dlnCO2_dPH__TA = egrad(
        lambda PH: np.log(Ks["K0"] * solve.get.fCO2fromTApH(TA, PH, totals, Ks))
    )(PH)
    gammaTC = dTC_dPH__TA / dlnCO2_dPH__TA
    # gammaTA is (d[ln(CO2)]/d[TA])^-1 with constant TC, i.e. γ_Alk of ESM10
    dlnCO2_dPH__TC = egrad(
        lambda PH: np.log(Ks["K0"] * solve.get.fCO2fromTCpH(TC, PH, totals, Ks))
    )(PH)
    gammaTA = dTA_dPH__TC / dlnCO2_dPH__TC
    # betaTC is (d[ln(H)]/d[TC])^-1 with constant TA, i.e. β_DIC of ESM10
    betaTC = dTC_dPH__TA / ilog10e
    # betaTA is (d[ln(H)]/d[TA])^-1 with constant TC, i.e. β_Alk of ESM10
    betaTA = dTA_dPH__TC / ilog10e
    # Saturation state differential w.r.t. carbonate ion is used for both TC and TA
    # buffers.  Doesn't matter whether we use aragonite or calcite because of the log.
    dlnOmegaAr_dCARB = d_lnOmega__d_CO3(CARB, totals, Ks)
    # omegaTC is (d[ln(OmegaAr)]/d[TC] with constant TA, i.e. ω_DIC of ESM10
    dCARB_dPH__TA = egrad(lambda PH: solve.get.CarbfromTApH(TA, PH, totals, Ks))(PH)
    omegaTC = dTC_dPH__TA / (dlnOmegaAr_dCARB * dCARB_dPH__TA)
    # omegaTA is (d[ln(OmegaAr)]/d[TA] with constant TC, i.e. ω_Alk of ESM10
    dCARB_dPH__TC = egrad(lambda PH: solve.get.CarbfromTCpH(TC, PH, totals, Ks))(PH)
    omegaTA = dTA_dPH__TC / (dlnOmegaAr_dCARB * dCARB_dPH__TC)
    return {
        "gammaTC": gammaTC,
        "betaTC": betaTC,
        "omegaTC": omegaTC,
        "gammaTA": gammaTA,
        "betaTA": betaTA,
        "omegaTA": omegaTA,
    }


def d_alkalinity__d_pH__fCO2(
    pH,
    fCO2,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    k_CO2,
    k_H2O,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
):
    pH, fCO2 = np.broadcast_arrays(
        pH,
        fCO2,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        k_CO2,
        k_H2O,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
    )[:2]
    return egrad(
        lambda pH: solve.get.inorganic.alkalinity_from_pH_fCO2(
            pH,
            fCO2,
            opt_to_free,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            k_CO2,
            k_H2O,
            k_H2CO3,
            k_HCO3,
            k_BOH3,
            k_H3PO4,
            k_H2PO4,
            k_HPO4,
            k_Si,
            k_NH3,
            k_H2S,
            k_HSO4_free,
            k_HF_free,
        )
    )(pH)


def d_dic__d_pH__fCO2(pH, fCO2, k_CO2, k_H2CO3, k_HCO3):
    pH, fCO2 = np.broadcast_arrays(pH, fCO2, k_CO2, k_H2CO3, k_HCO3)[:2]
    return egrad(
        lambda pH: solve.get.inorganic.dic_from_pH_fCO2(
            pH, fCO2, k_CO2, k_H2CO3, k_HCO3
        )
    )(pH)


def Q_isocap(d_alkalinity__d_pH__fCO2, d_dic__d_pH__fCO2):
    """d[TA]/d[TC] at constant fCO2, i.e., Q of HDW18."""
    return d_alkalinity__d_pH__fCO2 / d_dic__d_pH__fCO2


def Q_isocap_approx(dic, pCO2, k_CO2, k_H2CO3, k_HCO3):
    """Approximate isocapnic quotient of HDW18, Eq. 7."""
    return 1 + 2 * (k_HCO3 / (k_CO2 * k_H2CO3)) * dic / pCO2


def psi(Q_isocap):
    """ψ of FCG94 calculated following HDW18."""
    return -1.0 + 2.0 / Q_isocap


def d_fCO2__d_pH__alkalinity(
    alkalinity,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    k_CO2,
    k_H2O,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
):
    alkalinity, pH = np.broadcast_arrays(
        alkalinity,
        pH,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        k_CO2,
        k_H2O,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
    )[:2]
    return egrad(
        lambda pH: solve.get.inorganic.fCO2_from_alkalinity_pH(
            alkalinity,
            pH,
            opt_to_free,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            k_CO2,
            k_H2O,
            k_H2CO3,
            k_HCO3,
            k_BOH3,
            k_H3PO4,
            k_H2PO4,
            k_HPO4,
            k_Si,
            k_NH3,
            k_H2S,
            k_HSO4_free,
            k_HF_free,
        )
    )(pH)


def revelle_factor(dic, fCO2, d_fCO2__d_pH__alkalinity, d_dic__d_pH__alkalinity):
    """Revelle factor as defined by BTSP79."""
    return (d_fCO2__d_pH__alkalinity / d_dic__d_pH__alkalinity) * (dic / fCO2)


def revelle_factor_ESM10(dic, gamma_dic):
    """Revelle factor following ESM10 eq. (23)."""
    return dic / gamma_dic
