# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Solve the marine carbonate system from any two of its variables."""

from autograd import numpy as np
from . import delta, initialise, get
from .. import bio, buffers, convert, gas, solubility

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
    # Assign values to empty vectors & convert micro[mol|atm] to [mol|atm] if requested
    assert isinstance(convert_units, bool), "`convert_units` must be `True` or `False`."
    if convert_units:
        cfac = 1e-6
    else:
        cfac = 1.0
    TA = np.where(par1type == 1, par1 * cfac, TA)
    TC = np.where(par1type == 2, par1 * cfac, TC)
    PH = np.where(par1type == 3, par1, PH)
    PC = np.where(par1type == 4, par1 * cfac, PC)
    FC = np.where(par1type == 5, par1 * cfac, FC)
    CARB = np.where(par1type == 6, par1 * cfac, CARB)
    HCO3 = np.where(par1type == 7, par1 * cfac, HCO3)
    CO2 = np.where(par1type == 8, par1 * cfac, CO2)
    XC = np.where(par1type == 9, par1 * cfac, XC)
    TA = np.where(par2type == 1, par2 * cfac, TA)
    TC = np.where(par2type == 2, par2 * cfac, TC)
    PH = np.where(par2type == 3, par2, PH)
    PC = np.where(par2type == 4, par2 * cfac, PC)
    FC = np.where(par2type == 5, par2 * cfac, FC)
    CARB = np.where(par2type == 6, par2 * cfac, CARB)
    HCO3 = np.where(par2type == 7, par2 * cfac, HCO3)
    CO2 = np.where(par2type == 8, par2 * cfac, CO2)
    XC = np.where(par2type == 9, par2 * cfac, XC)
    if checks:
        _core_sanity(TC, PC, FC, CARB, HCO3, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC


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

    `Icase` is `10*parXtype + parYtype` where `parXtype` is whichever of `par1type` or
    `par2type` is greater.

    Noting that a pair of any two from pCO2, fCO2, xCO2 CO2(aq) is not allowed, the
    valid `Icase` options are:

        12, 13, 14, 15, 16, 17, 18, 19,
            23, 24, 25, 26, 27, 28, 29,
                34, 35, 36, 37, 38, 39,
                        46, 47,
                        56, 57,
                            67, 68, 69,
                                78, 79.

    The optional argument `checks` allows you to decide whether the function should test
    the validity of the entered combinations or not.
    """
    # Check validity of separate `par1type` and `par2type` inputs
    if checks:
        assert np.all(
            np.isin(par1type, [1, 2, 3, 4, 5, 6, 7, 8, 9])
            & np.isin(par2type, [1, 2, 3, 4, 5, 6, 7, 8, 9])
        ), "All `par1type` and `par2type` values must be integers from 1 to 9."
        assert ~np.any(
            par1type == par2type
        ), "`par1type` and `par2type` must be different from each other."
    # Combine inputs into `Icase` and check its validity
    Icase = np.where(
        par1type < par2type, 10 * par1type + par2type, par1type + 10 * par2type
    )
    if checks:
        assert ~np.any(
            np.isin(Icase, [45, 48, 49, 58, 59, 89])
        ), "Combinations of pCO2, fCO2, xCO2 and CO2(aq) are not valid argument pairs."
    return Icase


def fill(Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC, totals, Ks):
    """Fill part-empty core marine carbonate system variable columns with solutions."""
    # For convenience
    PengCx = totals["PengCorrection"]
    # Convert any pCO2 and CO2(aq) values into fCO2
    PCgiven = np.isin(Icase, [14, 24, 34, 46, 47])
    FC = np.where(PCgiven, convert.pCO2_to_fCO2(PC, Ks), FC)
    CO2given = np.isin(Icase, [18, 28, 38, 68, 78])
    FC = np.where(CO2given, convert.CO2aq_to_fCO2(CO2, Ks), FC)
    XCgiven = np.isin(Icase, [19, 29, 39, 69, 79])
    FC = np.where(XCgiven, convert.xCO2_to_fCO2(XC, Ks), FC)
    # Deal with zero-CARB or zero-HCO3 inputs
    zCARB_1 = (Icase == 16) & (CARB == 0)
    TC = np.where(zCARB_1, 0, TC)
    Icase = np.where(zCARB_1, 12, Icase)
    zCARB_3 = (Icase == 36) & (CARB == 0)
    TC = np.where(zCARB_3, 0, TC)
    Icase = np.where(zCARB_3, 23, Icase)
    zHCO3_1 = (Icase == 17) & (HCO3 == 0)
    TC = np.where(zHCO3_1, 0, TC)
    Icase = np.where(zHCO3_1, 12, Icase)
    zHCO3_3 = (Icase == 37) & (HCO3 == 0)
    TC = np.where(zHCO3_3, 0, TC)
    Icase = np.where(zHCO3_3, 23, Icase)
    # Solve the marine carbonate system
    F = Icase == 12  # input TA, TC
    if np.any(F):
        PH = np.where(F, get.pHfromTATC(TA - PengCx, TC, totals, Ks), PH)
        # ^pH is returned on the same scale as `Ks`
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    F = Icase == 13  # input TA, pH
    if np.any(F):
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    F = (
        (Icase == 14) | (Icase == 15) | (Icase == 18) | (Icase == 19)
    )  # input TA, [pCO2|fCO2|CO2aq|xCO2]
    if np.any(F):
        PH = np.where(F, get.pHfromTAfCO2(TA - PengCx, FC, totals, Ks), PH)
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 18, TC - CARB - CO2, HCO3)
    F = Icase == 16  # input TA, CARB
    if np.any(F):
        PH = np.where(F, get.pHfromTACarb(TA - PengCx, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    F = Icase == 17  # input TA, HCO3
    if np.any(F):
        PH = np.where(F, get.pHfromTAHCO3(TA - PengCx, HCO3, totals, Ks), PH)
        TC = np.where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
    F = Icase == 23  # input TC, pH
    if np.any(F):
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    F = (
        (Icase == 24) | (Icase == 25) | (Icase == 28) | (Icase == 29)
    )  # input TC, [pCO2|fCO2|CO2aq|xCO2]
    if np.any(F):
        PH = np.where(F, get.pHfromTCfCO2(TC, FC, totals, Ks), PH)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 28, TC - CARB - CO2, HCO3)
    F = Icase == 26  # input TC, CARB
    if np.any(F):
        PH = np.where(F, get.pHfromTCCarb(TC, CARB, totals, Ks), PH)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    F = Icase == 27  # input TC, HCO3
    if np.any(F):
        PH = np.where(F, get.pHfromTCHCO3(TC, HCO3, totals, Ks), PH)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
    F = (
        (Icase == 34) | (Icase == 35) | (Icase == 38) | (Icase == 39)
    )  # input pH, [pCO2|fCO2|CO2aq|xCO2]
    if np.any(F):
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 38, TC - CARB - CO2, HCO3)
    F = Icase == 36  # input pH, CARB
    if np.any(F):
        FC = np.where(F, get.fCO2frompHCarb(PH, CARB, totals, Ks), FC)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
    F = Icase == 37  # input pH, HCO3
    if np.any(F):
        TC = np.where(F, get.TCfrompHHCO3(PH, HCO3, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        FC = np.where(F, get.fCO2fromTCpH(TC, PH, totals, Ks), FC)
        CARB = np.where(F, get.CarbfromTCpH(TC, PH, totals, Ks), CARB)
    F = (
        (Icase == 46) | (Icase == 56) | (Icase == 68) | (Icase == 69)
    )  # input [pCO2|fCO2|CO2aq|xCO2], CARB
    if np.any(F):
        PH = np.where(F, get.pHfromfCO2Carb(FC, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = np.where(F, get.HCO3fromTCpH(TC, PH, totals, Ks), HCO3)
        HCO3 = np.where(Icase == 68, TC - CARB - CO2, HCO3)
    F = (
        (Icase == 47) | (Icase == 57) | (Icase == 78) | (Icase == 79)
    )  # input [pCO2|fCO2|CO2aq|xCO2], HCO3
    if np.any(F):
        CARB = np.where(F, get.CarbfromfCO2HCO3(FC, HCO3, totals, Ks), CARB)
        PH = np.where(F, get.pHfromfCO2Carb(FC, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TC = np.where(Icase == 78, CO2 + HCO3 + CARB, TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
    F = Icase == 67  # input CO3, HCO3
    if np.any(F):
        FC = np.where(F, get.fCO2fromCarbHCO3(CARB, HCO3, totals, Ks), FC)
        PH = np.where(F, get.pHfromfCO2Carb(FC, CARB, totals, Ks), PH)
        TC = np.where(F, get.TCfrompHfCO2(PH, FC, totals, Ks), TC)
        TA = np.where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
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

    The input `convert_units` specifies whether the inputs `par1` and `par2` are in
    μmol/kg and μatm units (`True`) or mol/kg and atm units (`False`).
    """
    # Expand inputs `par1` and `par2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC = pair2core(
        par1, par2, par1type, par2type, convert_units=convert_units, checks=True
    )
    # Generate vector describing the combination(s) of input parameters
    Icase = getIcase(par1type, par2type)
    # Solve the core marine carbonate system
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC = fill(
        Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, XC, totals, Ks
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
    buffers_mode,
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
        np.isin(buffers_mode, ["auto", "explicit", "none"])
    ), "Valid options for buffers_mode are 'auto', 'explicit' or 'none'."
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
    F = buffers_mode == "auto"
    if np.any(F):
        # Evaluate buffers with automatic differentiation [added v1.3.0]
        auto_ESM10 = buffers.all_ESM10(
            TAPeng,
            TC,
            PH,
            CARB,
            Sal,
            convert.TempC2K(TempC),
            convert.Pdbar2bar(Pdbar),
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
    F = buffers_mode == "explicit"
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
    F = buffers_mode != "none"
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
