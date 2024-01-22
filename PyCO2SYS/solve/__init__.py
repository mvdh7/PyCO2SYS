# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Solve the marine carbonate system from any two of its variables."""

from autograd import numpy as np, elementwise_grad as egrad
from . import delta, initialise, get
from .. import bio, buffers, convert, equilibria, gas, solubility

__all__ = ["delta", "initialise", "get"]


@np.errstate(invalid="ignore")
def _core_sanity(TC, PC, FC, CARB, HCO3, CO2):
    """Run basic sanity checks on core marine carbonate system parameter values."""
    assert np.all(
        (TC >= 0) | np.isnan(TC)
    ), "Dissolved inorganic carbon cannot be negative."
    assert np.all((PC >= 0) | np.isnan(PC)), "CO2 partial pressure cannot be negative."
    assert np.all((FC >= 0) | np.isnan(FC)), "CO2 fugacity cannot be negative."
    assert np.all(
        (CARB >= 0) | np.isnan(CARB)
    ), "Carbonate ion content cannot be negative."
    assert np.all(
        (HCO3 >= 0) | np.isnan(HCO3)
    ), "Bicarbonate ion content cannot be negative."
    assert np.all((CO2 >= 0) | np.isnan(CO2)), "Aqueous CO2 content cannot be negative."
    assert np.all(
        (CARB < TC) | np.isnan(CARB) | np.isnan(TC)
    ), "Carbonate ion content must be less than DIC."
    assert np.all(
        (HCO3 < TC) | np.isnan(HCO3) | np.isnan(TC)
    ), "Bicarbonate ion content must be less than DIC."
    assert np.all(
        (CO2 < TC) | np.isnan(CO2) | np.isnan(TC)
    ), "Aqueous CO2 content must be less than DIC."


def pair2core(par1, par2, par1type, par2type, convert_units=False, checks=True):
    """Expand `par1` and `par2` inputs into one array per core variable of the marine
    carbonate system.  Convert units from microX to X if requested with the input
    logical `convertunits`.
    """
    # assert (
    #     np.size(par1) == np.size(par2) == np.size(par1type) == np.size(par2type)
    # ), "`par1`, `par2`, `par1type` and `par2type` must all be the same size."
    ntps = np.broadcast(par1, par2, par1type, par2type).shape
    # Generate empty vectors for...
    TA = np.full(ntps, np.nan)  # total alkalinity
    TC = np.full(ntps, np.nan)  # dissolved inorganic carbon
    PH = np.full(ntps, np.nan)  # pH
    PC = np.full(ntps, np.nan)  # CO2 partial pressure
    FC = np.full(ntps, np.nan)  # CO2 fugacity
    CARB = np.full(ntps, np.nan)  # carbonate ions
    HCO3 = np.full(ntps, np.nan)  # bicarbonate ions
    CO2 = np.full(ntps, np.nan)  # aqueous CO2
    XC = np.full(ntps, np.nan)  # dry mole fraction of CO2
    OC = np.full(ntps, np.nan)  # saturation state w.r.t. calcite
    OA = np.full(ntps, np.nan)  # saturation state w.r.t. aragonite
    # Assign values to empty vectors & convert micro[mol|atm] to [mol|atm] if requested
    assert isinstance(convert_units, bool), "`convert_units` must be `True` or `False`."
    if convert_units:
        cfac = 1e-6
    else:
        cfac = 1.0
    # par1
    TA = np.where(par1type == 1, par1 * cfac, TA)
    TC = np.where(par1type == 2, par1 * cfac, TC)
    PH = np.where(par1type == 3, par1, PH)
    PC = np.where(par1type == 4, par1 * cfac, PC)
    FC = np.where(par1type == 5, par1 * cfac, FC)
    CARB = np.where(par1type == 6, par1 * cfac, CARB)
    HCO3 = np.where(par1type == 7, par1 * cfac, HCO3)
    CO2 = np.where(par1type == 8, par1 * cfac, CO2)
    XC = np.where(par1type == 9, par1 * cfac, XC)
    OC = np.where(par1type == 10, par1, OC)
    OA = np.where(par1type == 11, par1, OA)
    # par2
    TA = np.where(par2type == 1, par2 * cfac, TA)
    TC = np.where(par2type == 2, par2 * cfac, TC)
    PH = np.where(par2type == 3, par2, PH)
    PC = np.where(par2type == 4, par2 * cfac, PC)
    FC = np.where(par2type == 5, par2 * cfac, FC)
    CARB = np.where(par2type == 6, par2 * cfac, CARB)
    HCO3 = np.where(par2type == 7, par2 * cfac, HCO3)
    CO2 = np.where(par2type == 8, par2 * cfac, CO2)
    XC = np.where(par2type == 9, par2 * cfac, XC)
    OC = np.where(par2type == 10, par2, OC)
    OA = np.where(par2type == 11, par2, OA)
    if checks:
        _core_sanity(TC, PC, FC, CARB, HCO3, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC, OC, OA


def getIcase(par1type, par2type, checks=True):
    """Generate vector describing the combination of input parameters.

    Options for `par1type` and `par2type`:

      * `1` = total alkalinity
      * `2` = dissolved inorganic carbon
      * `3` = pH
      * `4` = partial pressure of CO2
      * `5` = fugacity of CO2
      * `6` = carbonate ion
      * `7` = bicarbonate ion
      * `8` = aqueous CO2
      * `9` = dry mole fraction of CO2
      * `10` = saturation state w.r.t. calcite
      * `11` = saturation state w.r.t. aragonite

    `Icase` is `100*parXtype + parYtype` where `parXtype` is whichever of `par1type` or
    `par2type` is smaller.

    Noting that a pair of any two from pCO2, fCO2, xCO2 CO2(aq) is not allowed, the
    valid `Icase` options are:

        102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
             203, 204, 205, 206, 207, 208, 209, 210, 211,
                  304, 305, 306, 307, 308, 309, 310, 311,
                            406, 407,           410, 411,
                            506, 507,           510, 511,
                                 607, 608, 609,
                                      708, 709, 710, 711,
                                                810, 811,
                                                910, 911.

    The optional argument `checks` allows you to decide whether the function should test
    the validity of the entered combinations or not.
    """
    # Check validity of separate `par1type` and `par2type` inputs
    if checks:
        all_pars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        assert np.all(
            np.isin(par1type, all_pars) & np.isin(par2type, all_pars)
        ), "All `par1type` and `par2type` values must be integers from 1 to 11."
        assert ~np.any(
            par1type == par2type
        ), "`par1type` and `par2type` must be different from each other."
    # Combine inputs into `Icase` and check its validity
    Icase = np.where(
        par1type < par2type, 100 * par1type + par2type, par1type + 100 * par2type
    )
    if checks:
        assert ~np.any(
            np.isin(Icase, [405, 408, 409, 508, 509, 809])
        ), "Combinations of pCO2, fCO2, xCO2 and [CO2(aq)] are not valid argument pairs."
        assert ~np.any(
            np.isin(Icase, [610, 611, 1011])
        ), "Combinations of [CO3] and/or saturation states are not valid."
    return Icase


def fill(Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC, OC, OA, totals, Ks):
    """Fill part-empty core marine carbonate system variable columns with solutions."""
    # For convenience
    PengCx = totals["PengCorrection"]
    # Convert any pCO2, CO2(aq) and xCO2 values into fCO2
    PCgiven = np.isin(Icase, [104, 204, 304, 406, 407, 410, 411])
    CO2given = np.isin(Icase, [108, 208, 308, 608, 708, 710, 711])
    XCgiven = np.isin(Icase, [109, 209, 309, 609, 709, 910, 911])
    FC = np.where(PCgiven, convert.pCO2_to_fCO2(PC, Ks), FC)
    FC = np.where(CO2given, convert.CO2aq_to_fCO2(CO2, Ks), FC)
    FC = np.where(XCgiven, convert.xCO2_to_fCO2(XC, Ks), FC)
    # Deal with zero-CARB or zero-HCO3 inputs: set DIC to zero,
    # and do something weird with Icase (can't remember why...)
    zCARB_1 = (Icase == 106) & (CARB == 0)
    TC = np.where(zCARB_1, 0, TC)
    Icase = np.where(zCARB_1, 102, Icase)
    zCARB_3 = (Icase == 306) & (CARB == 0)
    TC = np.where(zCARB_3, 0, TC)
    Icase = np.where(zCARB_3, 203, Icase)
    zHCO3_1 = (Icase == 107) & (HCO3 == 0)
    TC = np.where(zHCO3_1, 0, TC)
    Icase = np.where(zHCO3_1, 102, Icase)
    zHCO3_3 = (Icase == 307) & (HCO3 == 0)
    TC = np.where(zHCO3_3, 0, TC)
    Icase = np.where(zHCO3_3, 203, Icase)
    # Convert any saturation states to [CO3] - added in v1.8.1
    # Note that, unlike for pCO2 variants, we don't return OC and OA from this function
    OCgiven = np.isin(Icase, [110, 210, 310, 410, 510, 710, 810, 910])
    OAgiven = np.isin(Icase, [111, 211, 311, 411, 511, 711, 811, 911])
    CARB = np.where(OCgiven, solubility.CARB_from_OC(OC, totals, Ks), CARB)
    CARB = np.where(OAgiven, solubility.CARB_from_OA(OA, totals, Ks), CARB)
    # === SOLVE THE MARINE CARBONATE SYSTEM ============================================
    # ----------------------------------------------------------------------------------
    # Arguments: TA and TC -------------------------------------------------------------
    F = Icase == 102
    if np.any(F):
        PH = np.where(F, get.pHfromTATC(TA - PengCx, TC, totals, Ks), PH)
        # ^pH is returned on the same scale as `Ks`
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    # Arguments: TA and pH -------------------------------------------------------------
    F = Icase == 103
    if np.any(F):
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    # Arguments: TA and pCO2, fCO2, CO2aq or xCO2 --------------------------------------
    F = np.isin(Icase, [104, 105, 108, 109])
    if np.any(F):
        PH = np.where(F, get.pHfromTAfCO2(TA - PengCx, FC, totals, Ks), PH)
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 108, TC - CARB - CO2, HCO3)
    # Arguments: TA and CARB, OC or OA -------------------------------------------------
    F = np.isin(Icase, [106, 110, 111])
    if np.any(F):
        PH = np.where(F, get.pHfromTACarb(TA - PengCx, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    # Arguments: TA and HCO3 -----------------------------------------------------------
    F = Icase == 107
    if np.any(F):
        PH = np.where(F, get.pHfromTAHCO3(TA - PengCx, HCO3, totals, Ks), PH)
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
    # ----------------------------------------------------------------------------------
    # Arguments: TC and pH -------------------------------------------------------------
    F = Icase == 203
    if np.any(F):
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    # Arguments: TC and pCO2, fCO2, CO2aq or xCO2 --------------------------------------
    F = np.isin(Icase, [204, 205, 208, 209])
    if np.any(F):
        PH = np.where(F, get.pHfromTCfCO2(TC, FC, totals, Ks), PH)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 208, TC - CARB - CO2, HCO3)
    # Arguments: TC and CARB, OC or OA -------------------------------------------------
    F = np.isin(Icase, [206, 210, 211])
    if np.any(F):
        PH = np.where(F, get.pHfromTCCarb(TC, CARB, totals, Ks), PH)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    # Arguments: TC and HCO3 -----------------------------------------------------------
    F = Icase == 207
    if np.any(F):
        PH = np.where(F, get.pHfromTCHCO3(TC, HCO3, totals, Ks), PH)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
    # ----------------------------------------------------------------------------------
    # Arguments: pH and pCO2, fCO2, CO2aq or xCO2 --------------------------------------
    F = np.isin(Icase, [304, 305, 308, 309])
    if np.any(F):
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 308, TC - CARB - CO2, HCO3)
    # Arguments: pH and CARB, OC or OA -------------------------------------------------
    F = np.isin(Icase, [306, 310, 311])
    if np.any(F):
        FC = np.where(F, get.fCO2frompHCarb(PH, CARB, totals, Ks), FC)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    # Arguments: pH and HCO3 -----------------------------------------------------------
    F = Icase == 307
    if np.any(F):
        TC = np.where(F, get.TCfrompHHCO3(PH, HCO3, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
    # ----------------------------------------------------------------------------------
    # Arguments: pCO2, fCO2, CO2aq or xCO2 and CARB, OC or OA --------------------------
    F = np.isin(Icase, [406, 410, 411, 506, 510, 511, 608, 810, 811, 609, 910, 911])
    if np.any(F):
        PH = np.where(F, get.pHfromfCO2Carb(FC, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 608, TC - CARB - CO2, HCO3)
    # Arguments: pCO2, fCO2, CO2aq or xCO2 and HCO3 ------------------------------------
    F = np.isin(Icase, [407, 507, 708, 709])
    if np.any(F):
        CARB = np.where(F, get.CarbfromfCO2HCO3(FC, HCO3, totals, Ks), CARB)
        PH = np.where(F, get.pHfromfCO2Carb(FC, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TC = np.where(Icase == 708, CO2 + HCO3 + CARB, TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
    # ----------------------------------------------------------------------------------
    # Arguments: CARB, OC or OA and HCO3 ----------------------------------------------
    F = np.isin(Icase, [607, 710, 711])
    if np.any(F):
        FC = np.where(F, get.fCO2fromCarbHCO3(CARB, HCO3, totals, Ks), FC)
        PH = np.where(F, get.pHfromfCO2Carb(FC, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
    # ----------------------------------------------------------------------------------
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 and CO2(aq) values:
    PC = np.where(~PCgiven, convert.fCO2_to_pCO2(FC, Ks), PC)
    # CO2 = np.where(~CO2given, FC * K0, CO2)  # up to v1.6.0
    CO2 = np.where(~CO2given, TC - CARB - HCO3, CO2)  # v1.7.0 onwards
    XC = np.where(~XCgiven, convert.fCO2_to_xCO2(FC, Ks), XC)  # added in v1.7.0
    # ^this assumes pTot = 1 atm
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC


def core(par1, par2, par1type, par2type, totals, Ks, convert_units=True):
    """Solve the core marine carbonate system (MCS) from any 2 of its variables.

    The core MCS outputs (in a dict) and associated `par1type`/`par2type` inputs are:

      * Type `1`, `TA`: total alkalinity in (μ)mol/kg-sw.
      * Type `2`, `TC`: dissolved inorganic carbon in (μ)mol/kg-sw.
      * Type `3`, `PH`: pH on whichever scale(s) the constants in `Ks` are provided.
      * Type `4`, `PC`: partial pressure of CO2 in (μ)atm.
      * Type `5`, `FC`: fugacity of CO2 in (μ)atm.
      * Type `6`, `CARB`: carbonate ion in (μ)mol/kg-sw.
      * Type `7`, `HCO3`: bicarbonate ion in (μ)mol/kg-sw.
      * Type `8`, `CO2`: aqueous CO2 in (μ)mol/kg-sw.
      * Type `9`, `XC`: dry mole fraction of CO2 in ppm.
      * Type `10`: saturation state w.r.t. calcite.
      * Type `11`: saturation state w.r.t. aragonite.

    The input `convert_units` specifies whether the inputs `par1` and `par2` are in
    μmol/kg and μatm units (`True`) or mol/kg and atm units (`False`).
    """
    # Expand inputs `par1` and `par2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC, OC, OA = pair2core(
        par1, par2, par1type, par2type, convert_units=convert_units, checks=True
    )
    # Generate vector describing the combination(s) of input parameters
    Icase = getIcase(par1type, par2type)
    # Solve the core marine carbonate system
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC = fill(
        Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC, OC, OA, totals, Ks
    )
    return {
        "TA": TA,
        "TC": TC,
        "PH": PH,
        "PC": PC,
        "FC": FC,
        "CARB": CARB,
        "HCO3": HCO3,
        "CO2": CO2,
        "XC": XC,
    }


def others(
    core_solved,
    TempC,
    Pdbar,
    totals,
    Ks,
    pHScale,
    WhichKs,
    opt_buffers_mode,
):
    """Calculate all peripheral marine carbonate system variables returned by CO2SYS."""
    # Unpack for convenience
    Sal = totals["Sal"]
    TA = core_solved["TA"]
    TC = core_solved["TC"]
    PH = core_solved["PH"]
    PC = core_solved["PC"]
    FC = core_solved["FC"]
    CARB = core_solved["CARB"]
    HCO3 = core_solved["HCO3"]
    CO2 = core_solved["CO2"]
    # Apply Peng correction
    TAPeng = TA - totals["PengCorrection"]
    # Calculate pKs
    pK1 = -np.log10(Ks["K1"])
    pK2 = -np.log10(Ks["K2"])
    # Components of alkalinity and DIC
    # alks = get.AlkParts(TC, PH, totals, Ks)  # <=1.5.1
    sw = get.speciation_func(TC, PH, totals, Ks)  # >=1.6.0
    sw["PAlk"] = sw["PAlk"] + totals["PengCorrection"]
    # CaCO3 solubility
    OmegaCa, OmegaAr = solubility.CaCO3(CARB, totals, Ks)
    # Just for reference, convert pH at input conditions to the other scales
    pHT, pHS, pHF, pHN = convert.pH_to_all_scales(PH, pHScale, totals, Ks)
    # Get buffers as and if requested
    assert np.all(
        np.isin(opt_buffers_mode, [0, 1, 2])
    ), "Valid options for opt_buffers_mode are 0, 1, or 2."
    isoQx = np.full(np.shape(Sal), np.nan)
    isoQ = np.full(np.shape(Sal), np.nan)
    Revelle = np.full(np.shape(Sal), np.nan)
    psi = np.full(np.shape(Sal), np.nan)
    esm10buffers = [
        "gammaTC",
        "betaTC",
        "omegaTC",
        "gammaTA",
        "betaTA",
        "omegaTA",
    ]
    allbuffers_ESM10 = {
        buffer: np.full(np.shape(Sal), np.nan) for buffer in esm10buffers
    }
    F = opt_buffers_mode == 1
    if np.any(F):
        # Evaluate buffers with automatic differentiation [added v1.3.0]
        auto_ESM10 = buffers.all_ESM10(
            TAPeng,
            TC,
            PH,
            CARB,
            Sal,
            convert.celsius_to_kelvin(TempC),
            convert.decibar_to_bar(Pdbar),
            totals,
            Ks,
            WhichKs,
        )
        for buffer in esm10buffers:
            allbuffers_ESM10[buffer] = np.where(
                F, auto_ESM10[buffer], allbuffers_ESM10[buffer]
            )
        isoQ = np.where(F, buffers.isocap(TAPeng, TC, PH, FC, totals, Ks), isoQ)
        Revelle = np.where(
            F, buffers.RevelleFactor_ESM10(TC, allbuffers_ESM10["gammaTC"]), Revelle
        )
    F = opt_buffers_mode == 2
    if np.any(F):
        # Evaluate buffers with explicit equations, but these don't include nutrients
        # (i.e. only carbonate, borate and water alkalinities are accounted for)
        expl_ESM10 = buffers.explicit.all_ESM10(
            TC,
            TAPeng,
            CO2,
            HCO3,
            CARB,
            PH,
            sw["OH"],
            sw["BAlk"],
            Ks["KB"],
        )
        for buffer in esm10buffers:
            allbuffers_ESM10[buffer] = np.where(
                F, expl_ESM10[buffer], allbuffers_ESM10[buffer]
            )
        isoQ = np.where(
            F,
            buffers.explicit.isocap(
                CO2, PH, Ks["K1"], Ks["K2"], Ks["KB"], Ks["KW"], totals["TB"]
            ),
            isoQ,
        )
        Revelle = np.where(
            F, buffers.explicit.RevelleFactor(TAPeng, TC, totals, Ks), Revelle
        )
    F = opt_buffers_mode != 0
    if np.any(F):
        # Approximate isocapnic quotient of HDW18
        isoQx = np.where(
            F,
            buffers.explicit.isocap_approx(TC, PC, Ks["K0"], Ks["K1"], Ks["K2"]),
            isoQx,
        )
        # psi of FCG94 following HDW18
        psi = np.where(F, buffers.psi(isoQ), psi)
    # Substrate:inhibitor ratio of B15
    SIR = bio.SIratio(HCO3, pHF)
    others_out = {
        "pK1": pK1,
        "pK2": pK2,
        "OmegaCa": OmegaCa,
        "OmegaAr": OmegaAr,
        "pHT": pHT,
        "pHS": pHS,
        "pHF": pHF,
        "pHN": pHN,
        "Revelle": Revelle,
        "gammaTC": allbuffers_ESM10["gammaTC"],
        "betaTC": allbuffers_ESM10["betaTC"],
        "omegaTC": allbuffers_ESM10["omegaTC"],
        "gammaTA": allbuffers_ESM10["gammaTA"],
        "betaTA": allbuffers_ESM10["betaTA"],
        "omegaTA": allbuffers_ESM10["omegaTA"],
        "isoQ": isoQ,
        "isoQx": isoQx,
        "psi": psi,
        # Added in v1.4.0:
        "SIR": SIR,
    }
    # Added in v1.6.0:
    others_out.update(sw)
    return others_out


def get_lnfCO2(
    alkalinity,
    dic,
    temperature,
    pressure,
    totals,
    opt_pH_scale,
    opt_k_carbonic,
    opt_k_bisulfate,
    opt_k_fluoride,
    opt_gas_constant,
    k_constants_user,
    pressure_atmosphere,
    opt_pressured_kCO2,
):
    """Calculate the natural log of fCO2 from alkalinity and DIC.

    Parameters
    ----------
    alkalinity : float
        Total alkalinity in µmol/kg.
    dic : float
        Dissolve inorganic carbon in µmol/kg.
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    totals : dict
        Total salt contents, generated by pyco2.salts.assemble.
    opt_pH_scale : int
        Which pH scale to use (1-4).
    opt_k_carbonic : int
        Which carbonic acid dissociation constants to use (1-17).
    opt_k_bisulfate : int
        Which bisulfate dissociation constant to use (1-2).
    opt_k_fluoride : int
        Which HF dissociation constant to use (1-2).
    opt_gas_constant : int
        Which gas constant to use (1-3).
    k_constants_user : dict
        Any user-provided equilibrium constant values (not reevaluated).
    pressure_atmosphere : float
        Atmospheric pressure in atm.
    opt_pressured_kCO2 : int
        Whether to account for hydrostatic pressure in kCO2 (0-1).

    Returns
    -------
    float
        The natural log of fCO2.
    """
    k_constants = equilibria.assemble(
        temperature,
        pressure,
        totals,
        opt_pH_scale,
        opt_k_carbonic,
        opt_k_bisulfate,
        opt_k_fluoride,
        opt_gas_constant,
        Ks=k_constants_user,
        pressure_atmosphere=pressure_atmosphere,
        opt_pressured_kCO2=opt_pressured_kCO2,
    )
    fCO2 = get.fCO2fromTATC(
        alkalinity - totals["PengCorrection"], dic, totals, k_constants
    )
    return np.log(fCO2)


def get_dlnfCO2_dT(
    alkalinity,
    dic,
    temperature,
    pressure,
    totals,
    opt_pH_scale,
    opt_k_carbonic,
    opt_k_bisulfate,
    opt_k_fluoride,
    opt_gas_constant,
    k_constants_user,
    pressure_atmosphere,
    opt_pressured_kCO2,
):
    """Calculate the temperature derivative of the natural log of fCO2 from alkalinity
    and DIC.

    Parameters
    ----------
    alkalinity : float
        Total alkalinity in µmol/kg.
    dic : float
        Dissolve inorganic carbon in µmol/kg.
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    totals : dict
        Total salt contents, generated by pyco2.salts.assemble.
    opt_pH_scale : int
        Which pH scale to use (1-4).
    opt_k_carbonic : int
        Which carbonic acid dissociation constants to use (1-17).
    opt_k_bisulfate : int
        Which bisulfate dissociation constant to use (1-2).
    opt_k_fluoride : int
        Which HF dissociation constant to use (1-2).
    opt_gas_constant : int
        Which gas constant to use (1-3).
    k_constants_user : dict
        Any user-provided equilibrium constant values (not reevaluated).
    pressure_atmosphere : float
        Atmospheric pressure in atm.
    opt_pressured_kCO2 : int
        Whether to account for hydrostatic pressure in kCO2 (0-1).

    Returns
    -------
    float
        The temperature derivative of the natural log of fCO2.
    """
    return egrad(get_lnfCO2, argnum=2)(
        alkalinity,
        dic,
        temperature,
        pressure,
        totals,
        opt_pH_scale,
        opt_k_carbonic,
        opt_k_bisulfate,
        opt_k_fluoride,
        opt_gas_constant,
        k_constants_user,
        pressure_atmosphere,
        opt_pressured_kCO2,
    )


def get_lnpCO2(
    alkalinity,
    dic,
    temperature,
    pressure,
    totals,
    opt_pH_scale,
    opt_k_carbonic,
    opt_k_bisulfate,
    opt_k_fluoride,
    opt_gas_constant,
    k_constants_user,
    pressure_atmosphere,
    opt_pressured_kCO2,
):
    """Calculate the natural log of pCO2 from alkalinity and DIC.

    Parameters
    ----------
    alkalinity : float
        Total alkalinity in µmol/kg.
    dic : float
        Dissolve inorganic carbon in µmol/kg.
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    totals : dict
        Total salt contents, generated by pyco2.salts.assemble.
    opt_pH_scale : int
        Which pH scale to use (1-4).
    opt_k_carbonic : int
        Which carbonic acid dissociation constants to use (1-17).
    opt_k_bisulfate : int
        Which bisulfate dissociation constant to use (1-2).
    opt_k_fluoride : int
        Which HF dissociation constant to use (1-2).
    opt_gas_constant : int
        Which gas constant to use (1-3).
    k_constants_user : dict
        Any user-provided equilibrium constant values (not reevaluated).
    pressure_atmosphere : float
        Atmospheric pressure in atm.
    opt_pressured_kCO2 : int
        Whether to account for hydrostatic pressure in kCO2 (0-1).

    Returns
    -------
    float
        The natural log of pCO2.
    """
    k_constants = equilibria.assemble(
        temperature,
        pressure,
        totals,
        opt_pH_scale,
        opt_k_carbonic,
        opt_k_bisulfate,
        opt_k_fluoride,
        opt_gas_constant,
        Ks=k_constants_user,
        pressure_atmosphere=pressure_atmosphere,
        opt_pressured_kCO2=opt_pressured_kCO2,
    )
    fCO2 = get.fCO2fromTATC(
        alkalinity - totals["PengCorrection"], dic, totals, k_constants
    )
    pCO2 = convert.fCO2_to_pCO2(fCO2, k_constants)
    return np.log(pCO2)


def get_dlnpCO2_dT(
    alkalinity,
    dic,
    temperature,
    pressure,
    totals,
    opt_pH_scale,
    opt_k_carbonic,
    opt_k_bisulfate,
    opt_k_fluoride,
    opt_gas_constant,
    k_constants_user,
    pressure_atmosphere,
    opt_pressured_kCO2,
):
    """Calculate the temperature derivative of the natural log of pCO2 from alkalinity
    and DIC.

    Parameters
    ----------
    alkalinity : float
        Total alkalinity in µmol/kg.
    dic : float
        Dissolve inorganic carbon in µmol/kg.
    temperature : float
        Temperature in °C.
    pressure : float
        Hydrostatic pressure in dbar.
    totals : dict
        Total salt contents, generated by pyco2.salts.assemble.
    opt_pH_scale : int
        Which pH scale to use (1-4).
    opt_k_carbonic : int
        Which carbonic acid dissociation constants to use (1-17).
    opt_k_bisulfate : int
        Which bisulfate dissociation constant to use (1-2).
    opt_k_fluoride : int
        Which HF dissociation constant to use (1-2).
    opt_gas_constant : int
        Which gas constant to use (1-3).
    k_constants_user : dict
        Any user-provided equilibrium constant values (not reevaluated).
    pressure_atmosphere : float
        Atmospheric pressure in atm.
    opt_pressured_kCO2 : int
        Whether to account for hydrostatic pressure in kCO2 (0-1).

    Returns
    -------
    float
        The temperature derivative of the natural log of pCO2.
    """
    return egrad(get_lnpCO2, argnum=2)(
        alkalinity,
        dic,
        temperature,
        pressure,
        totals,
        opt_pH_scale,
        opt_k_carbonic,
        opt_k_bisulfate,
        opt_k_fluoride,
        opt_gas_constant,
        k_constants_user,
        pressure_atmosphere,
        opt_pressured_kCO2,
    )
