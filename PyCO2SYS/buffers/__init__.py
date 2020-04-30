# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Calculate various buffer factors of the marine carbonate system."""

from autograd.numpy import exp, log, log10
from autograd import elementwise_grad as egrad
from .. import solubility, solve
from . import explicit

__all__ = ["explicit"]

ilog10e = -1 / log10(exp(1))  # multiplier to convert pH to ln(H)


def all_ESM10(TA, TC, PH, CARB, Sal, TempK, Pbar, WhichKs, Ks, totals):
    """Get all ESM10 buffer factors with automatic differentiation.

    This is more efficient than calculating each one separately because the number of
    automatic differentiation steps is minimised.
    """
    # Get the pH differentials (slowest part - iterative)
    dPH_dTC__TA = egrad(lambda TC: solve.get.pHfromTATC(TA, TC, Ks, totals))(TC)
    dPH_dTA__TC = egrad(lambda TA: solve.get.pHfromTATC(TA, TC, Ks, totals))(TA)
    # gammaTC is (d[ln(CO2)]/d[TC])^-1 with constant TA, i.e. γ_DIC of ESM10
    dlnCO2_dPH__TA = egrad(
        lambda PH: log(Ks["K0"] * solve.get.fCO2fromTApH(TA, PH, Ks, totals))
    )(PH)
    gammaTC = 1.0 / (dPH_dTC__TA * dlnCO2_dPH__TA)
    # gammaTA is (d[ln(CO2)]/d[TA])^-1 with constant TC, i.e. γ_Alk of ESM10
    dlnCO2_dPH__TC = egrad(
        lambda PH: log(
            Ks["K0"] * solve.get.fCO2fromTCpH(TC, PH, Ks["K0"], Ks["K1"], Ks["K2"])
        )
    )(PH)
    gammaTA = 1.0 / (dPH_dTA__TC * dlnCO2_dPH__TC)
    # betaTC is (d[ln(H)]/d[TC])^-1 with constant TA, i.e. β_DIC of ESM10
    betaTC = 1.0 / (dPH_dTC__TA * ilog10e)
    # betaTA is (d[ln(H)]/d[TA])^-1 with constant TC, i.e. β_Alk of ESM10
    betaTA = 1.0 / (dPH_dTA__TC * ilog10e)
    # Saturation state differential w.r.t. carbonate ion is used for both TC and TA
    # buffers.  Doesn't matter whether we use aragonite or calcite because of the log.
    dlnOmegaAr_dCARB = egrad(
        lambda CARB: log(
            solubility.calcite(
                Sal, TempK, Pbar, CARB, totals["TCa"], WhichKs, Ks["K1"], Ks["K2"]
            )
        )
    )(CARB)
    # omegaTC is (d[ln(OmegaAr)]/d[TC] with constant TA, i.e. ω_DIC of ESM10
    dCARB_dPH__TA = egrad(lambda PH: solve.get.CarbfromTApH(TA, PH, Ks, totals))(PH)
    omegaTC = 1.0 / (dlnOmegaAr_dCARB * dCARB_dPH__TA * dPH_dTC__TA)
    # omegaTA is (d[ln(OmegaAr)]/d[TA] with constant TC, i.e. ω_Alk of ESM10
    dCARB_dPH__TC = egrad(
        lambda PH: solve.get.CarbfromTCpH(TC, PH, Ks["K1"], Ks["K2"])
    )(PH)
    omegaTA = 1.0 / (dlnOmegaAr_dCARB * dCARB_dPH__TC * dPH_dTA__TC)
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


def RevelleFactor(TA, TC, PH, Ks, totals):
    """Revelle factor defined by BTSP79."""
    dlnCO2_dPH__TA = egrad(
        lambda PH: log(Ks["K0"] * solve.get.fCO2fromTApH(TA, PH, Ks, totals))
    )(PH)
    dPH_dlnTC__TA = egrad(lambda lnTC: solve.get.pHfromTATC(TA, exp(lnTC), Ks, totals))(
        log(TC)
    )
    return dlnCO2_dPH__TA * dPH_dlnTC__TA


def gammaTC(TA, TC, Ks, totals):
    """(d[ln(CO2)]/d[TC])^-1 with constant TA, i.e. γ_DIC of ESM10."""
    gfunc = lambda TC: log(solve.get.fCO2fromTATC(TA, TC, Ks, totals))
    return 1.0 / egrad(gfunc)(TC)


def gammaTA(TA, TC, Ks, totals):
    """(d[ln(CO2)]/d[TA])^-1 with constant TC, i.e. γ_Alk of ESM10."""
    gfunc = lambda TA: log(solve.get.fCO2fromTATC(TA, TC, Ks, totals))
    return 1.0 / egrad(gfunc)(TA)


def betaTC(TA, TC, Ks, totals):
    """(d[ln(H)]/d[TC])^-1 with constant TA, i.e. β_DIC of ESM10."""
    gfunc = lambda TC: solve.get.pHfromTATC(TA, TC, Ks, totals) * ilog10e
    return 1.0 / egrad(gfunc)(TC)


def betaTA(TA, TC, Ks, totals):
    """(d[ln(H)]/d[TA])^-1 with constant TC, i.e. β_Alk of ESM10."""
    gfunc = lambda TA: solve.get.pHfromTATC(TA, TC, Ks, totals) * ilog10e
    return 1.0 / egrad(gfunc)(TA)


def _gfunc_omega(TA, TC, Sal, TempK, Pbar, WhichKs, Ks, totals):
    """Differential function for omegaTA/omegaTC."""
    CARB = solve.get.CarbfromTATC(TA, TC, Ks, totals)
    return log(
        solubility.calcite(
            Sal, TempK, Pbar, CARB, totals["TCa"], WhichKs, Ks["K1"], Ks["K2"]
        )
    )


def omegaTC(TA, TC, Sal, TempK, Pbar, WhichKs, Ks, totals):
    """(d[ln(OmegaAr)]/d[TC] with constant TA, i.e. ω_DIC of ESM10."""
    gfunc = lambda TC: _gfunc_omega(TA, TC, Sal, TempK, Pbar, WhichKs, Ks, totals)
    return 1.0 / egrad(gfunc)(TC)


def omegaTA(TA, TC, Sal, TempK, Pbar, WhichKs, Ks, totals):
    """(d[ln(OmegaAr)]/d[TA] with constant TC, i.e. ω_Alk of ESM10."""
    gfunc = lambda TA: _gfunc_omega(TA, TC, Sal, TempK, Pbar, WhichKs, Ks, totals)
    return 1.0 / egrad(gfunc)(TA)
