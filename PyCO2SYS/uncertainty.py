# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Uncertainty propagation."""
from autograd.numpy import full, isin, nan, size, where
from autograd.numpy import any as np_any
from autograd import elementwise_grad as egrad

# from . import solve
from .solve import get


def dcore_dparX__parY(parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks):
    """Derivatives of all core MCS variables w.r.t. parX at constant parY."""
    # Alias for convenience
    K0 = Ks["K0"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    # Get necessary derivatives
    # icase = solve.getIcase(parXtype, parYtype, checks=True)
    if np_any((parXtype == 1) & (parYtype == 2)):  # dvar_dTA__TC
        dTA_dPH__TC = egrad(lambda PH: get.TAfromTCpH(TC, PH, totals, Ks))(PH)
        dPH_dTA__TC = 1 / dTA_dPH__TC
        dFC_dPH__TC = egrad(lambda PH: get.fCO2fromTCpH(TC, PH, K0, K1, K2))(PH)
        dFC_dTA__TC = dFC_dPH__TC / dTA_dPH__TC
        dCARB_dPH__TC = egrad(lambda PH: get.CarbfromTCpH(TC, PH, K1, K2))(PH)
        dCARB_dTA__TC = dCARB_dPH__TC / dTA_dPH__TC
        dHCO3_dPH__TC = egrad(lambda PH: get.HCO3fromTCpH(TC, PH, K1, K2))(PH)
        dHCO3_dTA__TC = dHCO3_dPH__TC / dTA_dPH__TC
    if np_any((parXtype == 1) & (parYtype == 3)):  # dvar_dTA__PH
        dTC_dTA__PH = egrad(lambda TA: get.TCfromTApH(TA, PH, totals, Ks))(TA)
        dFC_dTA__PH = egrad(lambda TA: get.fCO2fromTApH(TA, PH, totals, Ks))(TA)
        dCARB_dTA__PH = egrad(lambda TA: get.CarbfromTApH(TA, PH, totals, Ks))(TA)
        dHCO3_dTA__PH = egrad(lambda TA: get.HCO3fromTApH(TA, PH, totals, Ks))(TA)
    if np_any((parXtype == 1) & isin(parYtype, [4, 5, 8])):  # dvar_dTA__FC
        dTC_dPH__FC = egrad(lambda PH: get.TCfrompHfCO2(PH, FC, K0, K1, K2))(PH)
        dTA_dPH__FC = egrad(lambda PH: get.TAfrompHfCO2(PH, FC, totals, Ks))(PH)
        dTC_dTA__FC = dTC_dPH__FC / dTA_dPH__FC
        dPH_dTA__FC = 1 / dTA_dPH__FC
        dCARB_dPH__FC = egrad(lambda PH: get.CarbfrompHfCO2(PH, FC, K0, K1, K2))(PH)
        dCARB_dTA__FC = dCARB_dPH__FC / dTA_dPH__FC
        dHCO3_dPH__FC = egrad(lambda PH: get.HCO3frompHfCO2(PH, FC, K0, K1))(PH)
        dHCO3_dTA__FC = dHCO3_dPH__FC / dTA_dPH__FC
    if np_any((parXtype == 1) & (parYtype == 6)):  # dvar_dTA__CARB
        dTC_dPH__CARB = egrad(lambda PH: get.TCfrompHCarb(PH, CARB, K1, K2))(PH)
        dTA_dPH__CARB = egrad(lambda PH: get.TAfrompHCarb(PH, CARB, totals, Ks))(PH)
        dTC_dTA__CARB = dTC_dPH__CARB / dTA_dPH__CARB
        dPH_dTA__CARB = 1 / dTA_dPH__CARB
        dFC_dPH__CARB = egrad(lambda PH: get.fCO2frompHCarb(PH, CARB, K0, K1, K2))(PH)
        dFC_dTA__CARB = dFC_dPH__CARB / dTA_dPH__CARB
        dHCO3_dPH__CARB = egrad(lambda PH: get.HCO3frompHCarb(PH, CARB, K2))(PH)
        dHCO3_dTA__CARB = dHCO3_dPH__CARB / dTA_dPH__CARB
    if np_any((parXtype == 1) & (parYtype == 7)):  # dvar_dTA__HCO3
        dTC_dPH__HCO3 = egrad(lambda PH: get.TCfrompHHCO3(PH, HCO3, K1, K2))(PH)
        dTA_dPH__HCO3 = egrad(lambda PH: get.TAfrompHHCO3(PH, HCO3, totals, Ks))(PH)
        dTC_dTA__HCO3 = dTC_dPH__HCO3 / dTA_dPH__HCO3
        dPH_dTA__HCO3 = 1 / dTA_dPH__HCO3
        dFC_dPH__HCO3 = egrad(lambda PH: get.fCO2frompHHCO3(PH, HCO3, K0, K1))(PH)
        dFC_dTA__HCO3 = dFC_dPH__HCO3 / dTA_dPH__HCO3
        dCARB_dPH__HCO3 = egrad(lambda PH: get.CarbfrompHHCO3(PH, HCO3, K2))(PH)
        dCARB_dTA__HCO3 = dCARB_dPH__HCO3 / dTA_dPH__HCO3
    # Preallocate empty arrays for derivatives
    dTA_dX__Y = full(size(parXtype), nan)
    dTC_dX__Y = full(size(parXtype), nan)
    dPH_dX__Y = full(size(parXtype), nan)
    dFC_dX__Y = full(size(parXtype), nan)
    dCARB_dX__Y = full(size(parXtype), nan)
    dHCO3_dX__Y = full(size(parXtype), nan)
    # Assign derivatives
    X = parXtype == 1  # TA - total alkalinity
    if np_any(X):
        dTA_dX__Y = where(X, 1.0, dTA_dX__Y)
        XY = X & (parYtype == 2)  # TA, TC
        if np_any(XY):
            dTC_dX__Y = where(XY, 0.0, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTA__TC, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTA__TC, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTA__TC, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTA__TC, dHCO3_dX__Y)
        XY = X & (parYtype == 3)  # TA, PH
        if np_any(XY):
            dTC_dX__Y = where(XY, dTC_dTA__PH, dTC_dX__Y)
            dPH_dX__Y = where(XY, 0.0, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTA__PH, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTA__PH, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTA__PH, dHCO3_dX__Y)
        XY = X & isin(parYtype, [4, 5, 8])  # TA, (PC | FC | CO2)
        if np_any(XY):
            dTC_dX__Y = where(XY, dTC_dTA__FC, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTA__FC, dPH_dX__Y)
            dFC_dX__Y = where(XY, 0.0, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTA__FC, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTA__FC, dHCO3_dX__Y)
        XY = X & (parYtype == 6)  # TA, CARB
        if np_any(XY):
            dTC_dX__Y = where(XY, dTC_dTA__CARB, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTA__CARB, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTA__CARB, dFC_dX__Y)
            dCARB_dX__Y = where(XY, 0.0, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTA__CARB, dHCO3_dX__Y)
        XY = X & (parYtype == 7)  # TA, HCO3
        if np_any(XY):
            dTC_dX__Y = where(XY, dTC_dTA__HCO3, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTA__HCO3, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTA__HCO3, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTA__HCO3, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, 0.0, dHCO3_dX__Y)
    # Get pCO2 and CO2(aq) derivatives from fCO2
    dPC_dX__Y = dFC_dX__Y / Ks["FugFac"]
    dCO2_dX__Y = dFC_dX__Y * K0
    # !!! and also update values where parX or parY were pCO2 or CO2(aq) !!!
    return (
        dTA_dX__Y,
        dTC_dX__Y,
        dPH_dX__Y,
        dPC_dX__Y,
        dFC_dX__Y,
        dCARB_dX__Y,
        dHCO3_dX__Y,
        dCO2_dX__Y,
    )
