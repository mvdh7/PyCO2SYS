# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Automatic derivatives for uncertainty propagation."""

from autograd.numpy import full, isin, nan, size, where
from autograd.numpy import any as np_any
from autograd import elementwise_grad as egrad
from .. import engine
from ..solve import get


def dcore_dparX__parY(parXtype, parYtype, TA, TC, PH, FC, CARB, HCO3, totals, Ks):
    """Efficient automatic derivatives of all core MCS variables w.r.t. parX
    at constant parY.
    """
    # Aliases for convenience
    K0 = Ks["K0"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    K012 = [K0, K1, K2]
    # Get necessary derivatives
    Ucase = 10 * parXtype + parYtype  # like Icase, but not sorted
    # Derivatives that are used by multiple Ucases
    if np_any(isin(Ucase, [12, 32, 42, 52, 62, 72, 82])):
        dTA_dPH__TC = egrad(lambda PH: get.TAfromTCpH(TC, PH, totals, Ks))(PH)
        dFC_dPH__TC = egrad(lambda PH: get.fCO2fromTCpH(TC, PH, *K012))(PH)
        dCARB_dPH__TC = egrad(lambda PH: get.CarbfromTCpH(TC, PH, K1, K2))(PH)
        dHCO3_dPH__TC = egrad(lambda PH: get.HCO3fromTCpH(TC, PH, K1, K2))(PH)
    if np_any(isin(Ucase, [21, 31, 41, 51, 61, 71, 81])):
        dTC_dPH__TA = egrad(lambda PH: get.TCfromTApH(TA, PH, totals, Ks))(PH)
        dFC_dPH__TA = egrad(lambda PH: get.fCO2fromTApH(TA, PH, totals, Ks))(PH)
        dCARB_dPH__TA = egrad(lambda PH: get.CarbfromTApH(TA, PH, totals, Ks))(PH)
        dHCO3_dPH__TA = egrad(lambda PH: get.HCO3fromTApH(TA, PH, totals, Ks))(PH)
    if np_any(isin(Ucase, [16, 26, 36])):
        dTC_dPH__CARB = egrad(lambda PH: get.TCfrompHCarb(PH, CARB, K1, K2))(PH)
        dTA_dPH__CARB = egrad(lambda PH: get.TAfrompHCarb(PH, CARB, totals, Ks))(PH)
        dFC_dPH__CARB = egrad(lambda PH: get.fCO2frompHCarb(PH, CARB, *K012))(PH)
        dHCO3_dPH__CARB = egrad(lambda PH: get.HCO3frompHCarb(PH, CARB, K2))(PH)
    if np_any(isin(Ucase, [17, 27, 37])):
        dTC_dPH__HCO3 = egrad(lambda PH: get.TCfrompHHCO3(PH, HCO3, K1, K2))(PH)
        dTA_dPH__HCO3 = egrad(lambda PH: get.TAfrompHHCO3(PH, HCO3, totals, Ks))(PH)
        dFC_dPH__HCO3 = egrad(lambda PH: get.fCO2frompHHCO3(PH, HCO3, K0, K1))(PH)
        dCARB_dPH__HCO3 = egrad(lambda PH: get.CarbfrompHHCO3(PH, HCO3, K2))(PH)
    if np_any(isin(Ucase, [14, 15, 18, 24, 25, 28, 34, 35, 38])):
        dTA_dPH__FC = egrad(lambda PH: get.TAfrompHfCO2(PH, FC, totals, Ks))(PH)
        dTC_dPH__FC = egrad(lambda PH: get.TCfrompHfCO2(PH, FC, *K012))(PH)
        dCARB_dPH__FC = egrad(lambda PH: get.CarbfrompHfCO2(PH, FC, *K012))(PH)
        dHCO3_dPH__FC = egrad(lambda PH: get.HCO3frompHfCO2(PH, FC, K0, K1))(PH)
    # Derivatives specific to a single Ucase
    if np_any((parXtype == 1) & (parYtype == 2)):  # dvar_dTA__TC
        dPH_dTA__TC = 1 / dTA_dPH__TC
        dFC_dTA__TC = dFC_dPH__TC / dTA_dPH__TC
        dCARB_dTA__TC = dCARB_dPH__TC / dTA_dPH__TC
        dHCO3_dTA__TC = dHCO3_dPH__TC / dTA_dPH__TC
    if np_any((parXtype == 1) & (parYtype == 3)):  # dvar_dTA__PH
        dTC_dTA__PH = egrad(lambda TA: get.TCfromTApH(TA, PH, totals, Ks))(TA)
        dFC_dTA__PH = egrad(lambda TA: get.fCO2fromTApH(TA, PH, totals, Ks))(TA)
        dCARB_dTA__PH = egrad(lambda TA: get.CarbfromTApH(TA, PH, totals, Ks))(TA)
        dHCO3_dTA__PH = egrad(lambda TA: get.HCO3fromTApH(TA, PH, totals, Ks))(TA)
    if np_any((parXtype == 1) & isin(parYtype, [4, 5, 8])):  # dvar_dTA__FC
        dTC_dTA__FC = dTC_dPH__FC / dTA_dPH__FC
        dPH_dTA__FC = 1 / dTA_dPH__FC
        dCARB_dTA__FC = dCARB_dPH__FC / dTA_dPH__FC
        dHCO3_dTA__FC = dHCO3_dPH__FC / dTA_dPH__FC
    if np_any((parXtype == 1) & (parYtype == 6)):  # dvar_dTA__CARB
        dTC_dTA__CARB = dTC_dPH__CARB / dTA_dPH__CARB
        dPH_dTA__CARB = 1 / dTA_dPH__CARB
        dFC_dTA__CARB = dFC_dPH__CARB / dTA_dPH__CARB
        dHCO3_dTA__CARB = dHCO3_dPH__CARB / dTA_dPH__CARB
    if np_any((parXtype == 1) & (parYtype == 7)):  # dvar_dTA__HCO3
        dTC_dTA__HCO3 = dTC_dPH__HCO3 / dTA_dPH__HCO3
        dPH_dTA__HCO3 = 1 / dTA_dPH__HCO3
        dFC_dTA__HCO3 = dFC_dPH__HCO3 / dTA_dPH__HCO3
        dCARB_dTA__HCO3 = dCARB_dPH__HCO3 / dTA_dPH__HCO3
    if np_any((parXtype == 2) & (parYtype == 1)):  # dvar_dTC__TA
        dPH_dTC__TA = 1 / dTC_dPH__TA
        dFC_dTC__TA = dFC_dPH__TA / dTC_dPH__TA
        dCARB_dTC__TA = dCARB_dPH__TA / dTC_dPH__TA
        dHCO3_dTC__TA = dHCO3_dPH__TA / dTC_dPH__TA
    if np_any((parXtype == 2) & (parYtype == 3)):  # dvar_dTC__PH
        dTA_dTC__PH = egrad(lambda TC: get.TAfromTCpH(TC, PH, totals, Ks))(TC)
        dFC_dTC__PH = egrad(lambda TC: get.fCO2fromTCpH(TC, PH, *K012))(TC)
        dCARB_dTC__PH = egrad(lambda TC: get.CarbfromTCpH(TC, PH, K1, K2))(TC)
        dHCO3_dTC__PH = egrad(lambda TC: get.HCO3fromTCpH(TC, PH, K1, K2))(TC)
    if np_any((parXtype == 2) & isin(parYtype, [4, 5, 8])):  # dvar_dTC__FC
        dTA_dTC__FC = dTA_dPH__FC / dTC_dPH__FC
        dPH_dTC__FC = 1 / dTC_dPH__FC
        dCARB_dTC__FC = dCARB_dPH__FC / dTC_dPH__FC
        dHCO3_dTC__FC = dHCO3_dPH__FC / dTC_dPH__FC
    if np_any((parXtype == 2) & (parYtype == 6)):  # dvar_dTC__CARB
        dTA_dTC__CARB = dTA_dPH__CARB / dTC_dPH__CARB
        dPH_dTC__CARB = 1 / dTC_dPH__CARB
        dFC_dTC__CARB = dFC_dPH__CARB / dTC_dPH__CARB
        dHCO3_dTC__CARB = dHCO3_dPH__CARB / dTC_dPH__CARB
    if np_any((parXtype == 2) & (parYtype == 7)):  # dvar_dTC__HCO3
        dTA_dTC__HCO3 = dTA_dPH__HCO3 / dTC_dPH__HCO3
        dPH_dTC__HCO3 = 1 / dTC_dPH__HCO3
        dFC_dTC__HCO3 = dFC_dPH__HCO3 / dTC_dPH__HCO3
        dCARB_dTC__HCO3 = dCARB_dPH__HCO3 / dTC_dPH__HCO3
    if np_any(isin(parXtype, [4, 5, 8]) & (parYtype == 1)):  # dvar_dFC__TA
        dTC_dFC__TA = dTC_dPH__TA / dFC_dPH__TA
        dPH_dFC__TA = 1 / dFC_dPH__TA
        dCARB_dFC__TA = dCARB_dPH__TA / dFC_dPH__TA
        dHCO3_dFC__TA = dHCO3_dPH__TA / dFC_dPH__TA
    if np_any(isin(parXtype, [4, 5, 8]) & (parYtype == 2)):  # dvar_dFC__TC
        dTA_dFC__TC = dTA_dPH__TC / dFC_dPH__TC
        dPH_dFC__TC = 1 / dFC_dPH__TC
        dCARB_dFC__TC = dCARB_dPH__TC / dFC_dPH__TC
        dHCO3_dFC__TC = dHCO3_dPH__TC / dFC_dPH__TC
    if np_any(isin(parXtype, [4, 5, 8]) & (parYtype == 3)):  # dvar_dFC__PH
        dTA_dFC__PH = egrad(lambda FC: get.TAfrompHfCO2(PH, FC, totals, Ks))(FC)
        dTC_dFC__PH = egrad(lambda FC: get.TCfrompHfCO2(PH, FC, *K012))(FC)
        dCARB_dFC__PH = egrad(lambda FC: get.CarbfrompHfCO2(PH, FC, *K012))(FC)
        dHCO3_dFC__PH = egrad(lambda FC: get.HCO3frompHfCO2(PH, FC, K0, K1))(FC)
    if np_any(isin(parXtype, [4, 5, 8]) & (parYtype == 6)):  # dvar_dFC__CARB
        dTA_dFC__CARB = egrad(lambda FC: get.TAfromfCO2Carb(FC, CARB, totals, Ks))(FC)
        dTC_dFC__CARB = egrad(lambda FC: get.TCfromfCO2Carb(FC, CARB, *K012))(FC)
        dPH_dFC__CARB = egrad(lambda FC: get.pHfromfCO2Carb(FC, CARB, *K012))(FC)
        dHCO3_dFC__CARB = egrad(lambda FC: get.HCO3fromfCO2Carb(FC, CARB, *K012))(FC)
    if np_any(isin(parXtype, [4, 5, 8]) & (parYtype == 7)):  # dvar_dFC__HCO3
        dTA_dFC__HCO3 = egrad(lambda FC: get.TAfromfCO2HCO3(FC, HCO3, totals, Ks))(FC)
        dTC_dFC__HCO3 = egrad(lambda FC: get.TCfromfCO2HCO3(FC, HCO3, *K012))(FC)
        dPH_dFC__HCO3 = egrad(lambda FC: get.pHfromfCO2HCO3(FC, HCO3, K0, K1))(FC)
        dCARB_dFC__HCO3 = egrad(lambda FC: get.CarbfromfCO2HCO3(FC, HCO3, *K012))(FC)
    if np_any((parXtype == 6) & (parYtype == 1)):  # dvar_dCARB__TA
        dTC_dCARB__TA = dTC_dPH__TA / dCARB_dPH__TA
        dPH_dCARB__TA = 1 / dCARB_dPH__TA
        dFC_dCARB__TA = dFC_dPH__TA / dCARB_dPH__TA
        dHCO3_dCARB__TA = dHCO3_dPH__TA / dCARB_dPH__TA
    if np_any((parXtype == 6) & (parYtype == 2)):  # dvar_dCARB__TC
        dTA_dCARB__TC = dTA_dPH__TC / dCARB_dPH__TC
        dPH_dCARB__TC = 1 / dCARB_dPH__TC
        dFC_dCARB__TC = dFC_dPH__TC / dCARB_dPH__TC
        dHCO3_dCARB__TC = dHCO3_dPH__TC / dCARB_dPH__TC
    if np_any((parXtype == 6) & (parYtype == 3)):  # dvar_dCARB__PH
        dTA_dCARB__PH = egrad(lambda CARB: get.TAfrompHCarb(PH, CARB, totals, Ks))(CARB)
        dTC_dCARB__PH = egrad(lambda CARB: get.TCfrompHCarb(PH, CARB, K1, K2))(CARB)
        dFC_dCARB__PH = egrad(lambda CARB: get.fCO2frompHCarb(PH, CARB, *K012))(CARB)
        dHCO3_dCARB__PH = egrad(lambda CARB: get.HCO3frompHCarb(PH, CARB, K2))(CARB)
    if np_any((parXtype == 6) & isin(parYtype, [4, 5, 8])):  # dvar_dCARB__FC
        dTA_dCARB__FC = egrad(lambda CARB: get.TAfromfCO2Carb(FC, CARB, totals, Ks))(
            CARB
        )
        dTC_dCARB__FC = egrad(lambda CARB: get.TCfromfCO2Carb(FC, CARB, *K012))(CARB)
        dPH_dCARB__FC = egrad(lambda CARB: get.pHfromfCO2Carb(FC, CARB, *K012))(CARB)
        dHCO3_dCARB__FC = egrad(lambda CARB: get.HCO3fromfCO2Carb(FC, CARB, *K012))(
            CARB
        )
    if np_any((parXtype == 6) & (parYtype == 7)):  # dvar_dCARB__HCO3
        dTA_dCARB__HCO3 = egrad(
            lambda CARB: get.TAfromCarbHCO3(CARB, HCO3, totals, Ks)
        )(CARB)
        dTC_dCARB__HCO3 = egrad(lambda CARB: get.TCfromCarbHCO3(CARB, HCO3, K1, K2))(
            CARB
        )
        dPH_dCARB__HCO3 = egrad(lambda CARB: get.pHfromCarbHCO3(CARB, HCO3, K2))(CARB)
        dFC_dCARB__HCO3 = egrad(lambda CARB: get.fCO2fromCarbHCO3(CARB, HCO3, *K012))(
            CARB
        )
    if np_any((parXtype == 7) & (parYtype == 1)):  # dvar_dHCO3__TA
        dTC_dHCO3__TA = dTC_dPH__TA / dHCO3_dPH__TA
        dPH_dHCO3__TA = 1 / dHCO3_dPH__TA
        dFC_dHCO3__TA = dFC_dPH__TA / dHCO3_dPH__TA
        dCARB_dHCO3__TA = dCARB_dPH__TA / dHCO3_dPH__TA
    if np_any((parXtype == 7) & (parYtype == 2)):  # dvar_dHCO3__TC
        dTA_dHCO3__TC = dTA_dPH__TC / dHCO3_dPH__TC
        dPH_dHCO3__TC = 1 / dHCO3_dPH__TC
        dFC_dHCO3__TC = dFC_dPH__TC / dHCO3_dPH__TC
        dCARB_dHCO3__TC = dCARB_dPH__TC / dHCO3_dPH__TC
    if np_any((parXtype == 7) & (parYtype == 3)):  # dvar_dHCO3__PH
        dTC_dHCO3__PH = egrad(lambda HCO3: get.TCfrompHHCO3(PH, HCO3, K1, K2))(HCO3)
        dTA_dHCO3__PH = egrad(lambda HCO3: get.TAfrompHHCO3(PH, HCO3, totals, Ks))(HCO3)
        dFC_dHCO3__PH = egrad(lambda HCO3: get.fCO2frompHHCO3(PH, HCO3, K0, K1))(HCO3)
        dCARB_dHCO3__PH = egrad(lambda HCO3: get.CarbfrompHHCO3(PH, HCO3, K2))(HCO3)
    if np_any((parXtype == 7) & isin(parYtype, [4, 5, 8])):  # dvar_dHCO3__FC
        dTA_dHCO3__FC = egrad(lambda HCO3: get.TAfromfCO2HCO3(FC, HCO3, totals, Ks))(
            HCO3
        )
        dTC_dHCO3__FC = egrad(lambda HCO3: get.TCfromfCO2HCO3(FC, HCO3, *K012))(HCO3)
        dPH_dHCO3__FC = egrad(lambda HCO3: get.pHfromfCO2HCO3(FC, HCO3, K0, K1))(HCO3)
        dCARB_dHCO3__FC = egrad(lambda HCO3: get.CarbfromfCO2HCO3(FC, HCO3, *K012))(
            HCO3
        )
    if np_any((parXtype == 7) & (parYtype == 6)):  # dvar_dHCO3__CARB
        dTA_dHCO3__CARB = egrad(
            lambda HCO3: get.TAfromCarbHCO3(CARB, HCO3, totals, Ks)
        )(HCO3)
        dTC_dHCO3__CARB = egrad(lambda HCO3: get.TCfromCarbHCO3(CARB, HCO3, K1, K2))(
            HCO3
        )
        dPH_dHCO3__CARB = egrad(lambda HCO3: get.pHfromCarbHCO3(CARB, HCO3, K2))(HCO3)
        dFC_dHCO3__CARB = egrad(lambda HCO3: get.fCO2fromCarbHCO3(CARB, HCO3, *K012))(
            HCO3
        )
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
    X = parXtype == 2  # TC - dissolved inorganic carbon
    if np_any(X):
        dTC_dX__Y = where(X, 1.0, dTC_dX__Y)
        XY = X & (parYtype == 1)  # TC, TA
        if np_any(XY):
            dTA_dX__Y = where(XY, 0.0, dTA_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTC__TA, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTC__TA, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTC__TA, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTC__TA, dHCO3_dX__Y)
        XY = X & (parYtype == 3)  # TC, PH
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dTC__PH, dTA_dX__Y)
            dPH_dX__Y = where(XY, 0.0, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTC__PH, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTC__PH, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTC__PH, dHCO3_dX__Y)
        XY = X & isin(parYtype, [4, 5, 8])  # TC, (PC | FC | CO2)
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dTC__FC, dTA_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTC__FC, dPH_dX__Y)
            dFC_dX__Y = where(XY, 0.0, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTC__FC, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTC__FC, dHCO3_dX__Y)
        XY = X & (parYtype == 6)  # TC, CARB
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dTC__CARB, dTA_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTC__CARB, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTC__CARB, dFC_dX__Y)
            dCARB_dX__Y = where(XY, 0.0, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dTC__CARB, dHCO3_dX__Y)
        XY = X & (parYtype == 7)  # TC, HCO3
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dTC__HCO3, dTA_dX__Y)
            dPH_dX__Y = where(XY, dPH_dTC__HCO3, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dTC__HCO3, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dTC__HCO3, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, 0.0, dHCO3_dX__Y)
    X = parXtype == 3  # PH - seawater pH
    if np_any(X):
        dPH_dX__Y = where(X, 1.0, dPH_dX__Y)
        XY = X & (parYtype == 1)  # PH, TA
        if np_any(XY):
            dTA_dX__Y = where(XY, 0.0, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dPH__TA, dTC_dX__Y)
            dFC_dX__Y = where(XY, dFC_dPH__TA, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dPH__TA, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dPH__TA, dHCO3_dX__Y)
        XY = X & (parYtype == 2)  # PH, TC
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dPH__TC, dTA_dX__Y)
            dTC_dX__Y = where(XY, 0.0, dTC_dX__Y)
            dFC_dX__Y = where(XY, dFC_dPH__TC, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dPH__TC, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dPH__TC, dHCO3_dX__Y)
        XY = X & isin(parYtype, [4, 5, 8])  # PH, (PC | FC | CO2)
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dPH__FC, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dPH__FC, dTC_dX__Y)
            dFC_dX__Y = where(XY, 0.0, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dPH__FC, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dPH__FC, dHCO3_dX__Y)
        XY = X & (parYtype == 6)  # PH, CARB
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dPH__CARB, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dPH__CARB, dTC_dX__Y)
            dFC_dX__Y = where(XY, dFC_dPH__CARB, dFC_dX__Y)
            dCARB_dX__Y = where(XY, 0.0, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dPH__CARB, dHCO3_dX__Y)
        XY = X & (parYtype == 7)  # PH, HCO3
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dPH__HCO3, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dPH__HCO3, dTC_dX__Y)
            dFC_dX__Y = where(XY, dFC_dPH__HCO3, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dPH__HCO3, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, 0.0, dHCO3_dX__Y)
    X = isin(parXtype, [4, 5, 8])  # (PC | FC | CO2) - CO2 fugacity, p.p. or (aq)
    if np_any(X):
        dFC_dX__Y = where(X, 1.0, dFC_dX__Y)
        XY = X & (parYtype == 1)  # (PC | FC | CO2), TA
        if np_any(XY):
            dTA_dX__Y = where(XY, 0.0, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dFC__TA, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dFC__TA, dPH_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dFC__TA, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dFC__TA, dHCO3_dX__Y)
        XY = X & (parYtype == 2)  # (PC | FC | CO2), TC
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dFC__TC, dTA_dX__Y)
            dTC_dX__Y = where(XY, 0.0, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dFC__TC, dPH_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dFC__TC, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dFC__TC, dHCO3_dX__Y)
        XY = X & (parYtype == 3)  # (PC | FC | CO2), PH
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dFC__PH, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dFC__PH, dTC_dX__Y)
            dPH_dX__Y = where(XY, 0.0, dPH_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dFC__PH, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dFC__PH, dHCO3_dX__Y)
        XY = X & (parYtype == 6)  # (PC | FC | CO2), CARB
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dFC__CARB, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dFC__CARB, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dFC__CARB, dPH_dX__Y)
            dCARB_dX__Y = where(XY, 0.0, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dFC__CARB, dHCO3_dX__Y)
        XY = X & (parYtype == 7)  # (PC | FC | CO2), HCO3
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dFC__HCO3, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dFC__HCO3, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dFC__HCO3, dPH_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dFC__HCO3, dCARB_dX__Y)
            dHCO3_dX__Y = where(XY, 0.0, dHCO3_dX__Y)
    X = parXtype == 6  # CARB - carbonate ion
    if np_any(X):
        dCARB_dX__Y = where(X, 1.0, dCARB_dX__Y)
        XY = X & (parYtype == 1)  # CARB, TA
        if np_any(XY):
            dTA_dX__Y = where(XY, 0.0, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dCARB__TA, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dCARB__TA, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dCARB__TA, dFC_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dCARB__TA, dHCO3_dX__Y)
        XY = X & (parYtype == 2)  # CARB, TC
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dCARB__TC, dTA_dX__Y)
            dTC_dX__Y = where(XY, 0.0, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dCARB__TC, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dCARB__TC, dFC_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dCARB__TC, dHCO3_dX__Y)
        XY = X & (parYtype == 3)  # CARB, PH
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dCARB__PH, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dCARB__PH, dTC_dX__Y)
            dPH_dX__Y = where(XY, 0.0, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dCARB__PH, dFC_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dCARB__PH, dHCO3_dX__Y)
        XY = X & isin(parYtype, [4, 5, 8])  # CARB, (PC | FC | CO2)
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dCARB__FC, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dCARB__FC, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dCARB__FC, dPH_dX__Y)
            dFC_dX__Y = where(XY, 0.0, dFC_dX__Y)
            dHCO3_dX__Y = where(XY, dHCO3_dCARB__FC, dHCO3_dX__Y)
        XY = X & (parYtype == 7)  # CARB, HCO3
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dCARB__HCO3, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dCARB__HCO3, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dCARB__HCO3, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dCARB__HCO3, dFC_dX__Y)
            dHCO3_dX__Y = where(XY, 0.0, dHCO3_dX__Y)
    X = parXtype == 7  # HCO3 - bicarbonate ion
    if np_any(X):
        dHCO3_dX__Y = where(X, 1.0, dHCO3_dX__Y)
        XY = X & (parYtype == 1)  # HCO3, TA
        if np_any(XY):
            dTA_dX__Y = where(XY, 0.0, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dHCO3__TA, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dHCO3__TA, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dHCO3__TA, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dHCO3__TA, dCARB_dX__Y)
        XY = X & (parYtype == 2)  # HCO3, TC
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dHCO3__TC, dTA_dX__Y)
            dTC_dX__Y = where(XY, 0.0, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dHCO3__TC, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dHCO3__TC, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dHCO3__TC, dCARB_dX__Y)
        XY = X & (parYtype == 3)  # HCO3, PH
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dHCO3__PH, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dHCO3__PH, dTC_dX__Y)
            dPH_dX__Y = where(XY, 0.0, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dHCO3__PH, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dHCO3__PH, dCARB_dX__Y)
        XY = X & isin(parYtype, [4, 5, 8])  # HCO3, (PC | FC | CO2)
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dHCO3__FC, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dHCO3__FC, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dHCO3__FC, dPH_dX__Y)
            dFC_dX__Y = where(XY, 0.0, dFC_dX__Y)
            dCARB_dX__Y = where(XY, dCARB_dHCO3__FC, dCARB_dX__Y)
        XY = X & (parYtype == 6)  # HCO3, CARB
        if np_any(XY):
            dTA_dX__Y = where(XY, dTA_dHCO3__CARB, dTA_dX__Y)
            dTC_dX__Y = where(XY, dTC_dHCO3__CARB, dTC_dX__Y)
            dPH_dX__Y = where(XY, dPH_dHCO3__CARB, dPH_dX__Y)
            dFC_dX__Y = where(XY, dFC_dHCO3__CARB, dFC_dX__Y)
            dCARB_dX__Y = where(XY, 0.0, dCARB_dX__Y)
    # Get pCO2 and CO2(aq) derivatives from fCO2
    dPC_dX__Y = dFC_dX__Y / Ks["FugFac"]
    dCO2_dX__Y = dFC_dX__Y * K0
    # Update values where parX or parY were pCO2 or CO2(aq)
    X = parXtype == 4  # pCO2
    if any(X):
        dTA_dX__Y = where(X, dTA_dX__Y * Ks["FugFac"], dTA_dX__Y)
        dTC_dX__Y = where(X, dTC_dX__Y * Ks["FugFac"], dTC_dX__Y)
        dPH_dX__Y = where(X, dPH_dX__Y * Ks["FugFac"], dPH_dX__Y)
        dPC_dX__Y = where(X, dPC_dX__Y * Ks["FugFac"], dPC_dX__Y)
        dFC_dX__Y = where(X, dFC_dX__Y * Ks["FugFac"], dFC_dX__Y)
        dCARB_dX__Y = where(X, dCARB_dX__Y * Ks["FugFac"], dCARB_dX__Y)
        dHCO3_dX__Y = where(X, dHCO3_dX__Y * Ks["FugFac"], dHCO3_dX__Y)
        dCO2_dX__Y = where(X, dCO2_dX__Y * Ks["FugFac"], dCO2_dX__Y)
    X = parXtype == 8  # CO2(aq)
    if any(X):
        dTA_dX__Y = where(X, dTA_dX__Y / K0, dTA_dX__Y)
        dTC_dX__Y = where(X, dTC_dX__Y / K0, dTC_dX__Y)
        dPH_dX__Y = where(X, dPH_dX__Y / K0, dPH_dX__Y)
        dPC_dX__Y = where(X, dPC_dX__Y / K0, dPC_dX__Y)
        dFC_dX__Y = where(X, dFC_dX__Y / K0, dFC_dX__Y)
        dCARB_dX__Y = where(X, dCARB_dX__Y / K0, dCARB_dX__Y)
        dHCO3_dX__Y = where(X, dHCO3_dX__Y / K0, dHCO3_dX__Y)
        dCO2_dX__Y = where(X, dCO2_dX__Y / K0, dCO2_dX__Y)
    return {
        "TAlk": dTA_dX__Y,
        "TCO2": dTC_dX__Y,
        "pH": dPH_dX__Y,
        "pCO2": dPC_dX__Y,
        "fCO2": dFC_dX__Y,
        "CO3": dCARB_dX__Y,
        "HCO3": dHCO3_dX__Y,
        "CO2": dCO2_dX__Y,
    }


def pars2core(co2dict, uncertainties):
    """Do uncertainty propagation."""
    # Extract results from the `co2dict` for convenience
    totals = engine.dict2totals(co2dict)
    Kis, Kos = engine.dict2equilibria(co2dict)
    par1type = co2dict["PAR1TYPE"]
    par2type = co2dict["PAR2TYPE"]
    TA = co2dict["TAlk"] * 1e-6
    TC = co2dict["TCO2"] * 1e-6
    PHi = co2dict["pHin"]
    FCi = co2dict["fCO2in"] * 1e-6
    CARBi = co2dict["CO3in"] * 1e-6
    HCO3i = co2dict["HCO3in"] * 1e-6
    PHo = co2dict["pHout"]
    FCo = co2dict["fCO2out"] * 1e-6
    CARBo = co2dict["CO3out"] * 1e-6
    HCO3o = co2dict["HCO3out"] * 1e-6
    # Get par1/part derivatives
    par1type_TA = full(size(par1type), 1)
    par2type_TC = full(size(par2type), 2)
    dcoreo_dTA__TC = dcore_dparX__parY(
        par1type_TA, par2type_TC, TA, TC, PHo, FCo, CARBo, HCO3o, totals, Kos
    )
    dcoreo_dTC__TA = dcore_dparX__parY(
        par2type_TC, par1type_TA, TA, TC, PHo, FCo, CARBo, HCO3o, totals, Kos
    )
    if "PAR1" in uncertainties:
        dcorei_dp1 = dcore_dparX__parY(
            par1type, par2type, TA, TC, PHi, FCi, CARBi, HCO3i, totals, Kis
        )
        dcoreo_dp1 = {
            k: dcorei_dp1["TAlk"] * dcoreo_dTA__TC[k]
            + dcorei_dp1["TCO2"] * dcoreo_dTC__TA[k]
            for k in dcorei_dp1
        }
    if "PAR2" in uncertainties:
        dcorei_dp2 = dcore_dparX__parY(
            par2type, par1type, TA, TC, PHi, FCi, CARBi, HCO3i, totals, Kis
        )
        dcoreo_dp2 = {
            k: dcorei_dp2["TAlk"] * dcoreo_dTA__TC[k]
            + dcorei_dp2["TCO2"] * dcoreo_dTC__TA[k]
            for k in dcorei_dp2
        }
    # Merge everything into output dicts
    iosame = ["TAlk", "TCO2"]
    uout = {}
    if "PAR1" in uncertainties:
        dvars_dp1 = {k: dcorei_dp1[k] for k in iosame}
        dvars_dp1.update(
            {"{}in".format(k): v for k, v in dcorei_dp1.items() if k not in iosame}
        )
        dvars_dp1.update(
            {"{}out".format(k): v for k, v in dcoreo_dp1.items() if k not in iosame}
        )
        uout.update({"PAR1": dvars_dp1})
    if "PAR2" in uncertainties:
        dvars_dp2 = {k: dcorei_dp2[k] for k in iosame}
        dvars_dp2.update(
            {"{}in".format(k): v for k, v in dcorei_dp2.items() if k not in iosame}
        )
        dvars_dp2.update(
            {"{}out".format(k): v for k, v in dcoreo_dp2.items() if k not in iosame}
        )
        uout.update({"PAR2": dvars_dp2})
    return uout
