# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Calculate various buffer factors of the marine carbonate system."""

from autograd.numpy import errstate, exp, log, log10
from autograd import elementwise_grad as egrad
from .. import solubility, solve
from . import explicit

__all__ = ["explicit"]

ilog10e = -1 / log10(exp(1))  # multiplier to convert pH to ln(H)


@errstate(all="ignore")
def _dlnOmega_dCARB(Sal, TempK, Pbar, CARB, WhichKs, Ks, totals):
    """Function for d[ln(Omega)]/d[CARB].  Identical for calcite and aragonite."""
    return egrad(
        lambda CARB: log(
            solubility.calcite(Sal, TempK, Pbar, CARB, totals["TCa"], WhichKs)
        )
    )(CARB)


def all_ESM10(TA, TC, PH, CARB, Sal, TempK, Pbar, WhichKs, Ks, totals):
    """Get all ESM10 buffer factors with automatic differentiation.

    This is more efficient than calculating each one separately because the number of
    automatic differentiation steps is minimised.
    """
    # Get the pH differentials
    dTC_dPH__TA = egrad(lambda PH: solve.get.TCfromTApH(TA, PH, Ks, totals))(PH)
    dTA_dPH__TC = egrad(lambda PH: solve.get.TAfromTCpH(TC, PH, Ks, totals))(PH)
    # gammaTC is (d[ln(CO2)]/d[TC])^-1 with constant TA, i.e. γ_DIC of ESM10
    dlnCO2_dPH__TA = egrad(
        lambda PH: log(Ks["K0"] * solve.get.fCO2fromTApH(TA, PH, Ks, totals))
    )(PH)
    gammaTC = dTC_dPH__TA / dlnCO2_dPH__TA
    # gammaTA is (d[ln(CO2)]/d[TA])^-1 with constant TC, i.e. γ_Alk of ESM10
    dlnCO2_dPH__TC = egrad(
        lambda PH: log(
            Ks["K0"] * solve.get.fCO2fromTCpH(TC, PH, Ks["K0"], Ks["K1"], Ks["K2"])
        )
    )(PH)
    gammaTA = dTA_dPH__TC / dlnCO2_dPH__TC
    # betaTC is (d[ln(H)]/d[TC])^-1 with constant TA, i.e. β_DIC of ESM10
    betaTC = dTC_dPH__TA / ilog10e
    # betaTA is (d[ln(H)]/d[TA])^-1 with constant TC, i.e. β_Alk of ESM10
    betaTA = dTA_dPH__TC / ilog10e
    # Saturation state differential w.r.t. carbonate ion is used for both TC and TA
    # buffers.  Doesn't matter whether we use aragonite or calcite because of the log.
    dlnOmegaAr_dCARB = _dlnOmega_dCARB(Sal, TempK, Pbar, CARB, WhichKs, Ks, totals)
    # omegaTC is (d[ln(OmegaAr)]/d[TC] with constant TA, i.e. ω_DIC of ESM10
    dCARB_dPH__TA = egrad(lambda PH: solve.get.CarbfromTApH(TA, PH, Ks, totals))(PH)
    omegaTC = dTC_dPH__TA / (dlnOmegaAr_dCARB * dCARB_dPH__TA)
    # omegaTA is (d[ln(OmegaAr)]/d[TA] with constant TC, i.e. ω_Alk of ESM10
    dCARB_dPH__TC = egrad(
        lambda PH: solve.get.CarbfromTCpH(TC, PH, Ks["K1"], Ks["K2"])
    )(PH)
    omegaTA = dTA_dPH__TC / (dlnOmegaAr_dCARB * dCARB_dPH__TC)
    return {
        "gammaTC": gammaTC,
        "betaTC": betaTC,
        "omegaTC": omegaTC,
        "gammaTA": gammaTA,
        "betaTA": betaTA,
        "omegaTA": omegaTA,
    }


def isocap(TA, TC, PH, FC, Ks, totals):
    """d[TA]/d[TC] at constant fCO2, i.e. Q of HDW18."""
    dTA_dPH__FC = egrad(lambda PH: solve.get.TAfrompHfCO2(PH, FC, Ks, totals))(PH)
    dTC_dPH__FC = egrad(
        lambda PH: solve.get.TCfrompHfCO2(PH, FC, Ks["K0"], Ks["K1"], Ks["K2"])
    )(PH)
    return dTA_dPH__FC / dTC_dPH__FC


def psi(Q):
    """ψ of FCG94 calculated following HDW18."""
    return -1.0 + 2.0 / Q


def RevelleFactor(TA, TC, PH, FC, Ks, totals):
    """Revelle factor as defined by BTSP79."""
    dFC_dPH__TA = egrad(lambda PH: solve.get.fCO2fromTApH(TA, PH, Ks, totals))(PH)
    dTC_dPH__TA = egrad(lambda PH: solve.get.TCfromTApH(TA, PH, Ks, totals))(PH)
    return (dFC_dPH__TA / dTC_dPH__TA) * (TC / FC)


def RevelleFactor_ESM10(TC, gammaTC):
    """Revelle factor following ESM10 eq. (23)."""
    return TC / gammaTC


def gammaTC(TA, PH, Ks, totals):
    """(d[ln(CO2)]/d[TC])^-1 with constant TA, i.e. γ_DIC of ESM10."""
    dTC_dPH__TA = egrad(lambda PH: solve.get.TCfromTApH(TA, PH, Ks, totals))(PH)
    dlnFC_dPH__TA = egrad(lambda PH: log(solve.get.fCO2fromTApH(TA, PH, Ks, totals)))(
        PH
    )
    return dTC_dPH__TA / dlnFC_dPH__TA


def gammaTA(TC, PH, Ks, totals):
    """(d[ln(CO2)]/d[TA])^-1 with constant TC, i.e. γ_Alk of ESM10."""
    dTA_dPH__TC = egrad(lambda PH: solve.get.TAfromTCpH(TC, PH, Ks, totals))(PH)
    dlnFC_dPH__TC = egrad(
        lambda PH: log(solve.get.fCO2fromTCpH(TC, PH, Ks["K0"], Ks["K1"], Ks["K2"]))
    )(PH)
    return dTA_dPH__TC / dlnFC_dPH__TC


def betaTC(TA, PH, Ks, totals):
    """(d[ln(H)]/d[TC])^-1 with constant TA, i.e. β_DIC of ESM10."""
    dTC_dlnH__TA = egrad(
        lambda lnH: solve.get.TCfromTApH(TA, lnH / ilog10e, Ks, totals)
    )(PH * ilog10e)
    return dTC_dlnH__TA


def betaTA(TC, PH, Ks, totals):
    """(d[ln(H)]/d[TA])^-1 with constant TC, i.e. β_Alk of ESM10."""
    dTA_dlnH__TC = egrad(
        lambda lnH: solve.get.TAfromTCpH(TC, lnH / ilog10e, Ks, totals)
    )(PH * ilog10e)
    return dTA_dlnH__TC


def omegaTC(TA, PH, CARB, Sal, TempK, Pbar, WhichKs, Ks, totals):
    """(d[ln(OmegaAr)]/d[TC] with constant TA, i.e. ω_DIC of ESM10."""
    dCARB_dPH__TA = egrad(lambda PH: solve.get.CarbfromTApH(TA, PH, Ks, totals))(PH)
    dTC_dPH__TA = egrad(lambda PH: solve.get.TCfromTApH(TA, PH, Ks, totals))(PH)
    return dTC_dPH__TA / (
        dCARB_dPH__TA * _dlnOmega_dCARB(Sal, TempK, Pbar, CARB, WhichKs, Ks, totals)
    )


def omegaTA(TC, PH, CARB, Sal, TempK, Pbar, WhichKs, Ks, totals):
    """(d[ln(OmegaAr)]/d[TA] with constant TC, i.e. ω_Alk of ESM10."""
    dCARB_dPH__TC = egrad(
        lambda PH: solve.get.CarbfromTCpH(TC, PH, Ks["K1"], Ks["K2"])
    )(PH)
    dTA_dPH__TC = egrad(lambda PH: solve.get.TAfromTCpH(TC, PH, Ks, totals))(PH)
    return dTA_dPH__TC / (
        dCARB_dPH__TC * _dlnOmega_dCARB(Sal, TempK, Pbar, CARB, WhichKs, Ks, totals)
    )
