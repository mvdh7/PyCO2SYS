# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Solve the marine carbonate system from any two of its variables."""

from autograd.numpy import errstate, full, isin, isnan, log10, nan, size, where
from autograd.numpy import all as np_all
from autograd.numpy import any as np_any
from . import delta, initialise, get
from .. import bio, buffers, convert, gas, solubility

__all__ = ["delta", "initialise", "get"]


@errstate(invalid="ignore")
def _core_sanity(TC, PC, FC, CARB, HCO3, CO2):
    """Run basic sanity checks on core marine carbonate system parameter values."""
    assert np_all(
        (TC >= 0) | isnan(TC)
    ), "Dissolved inorganic carbon cannot be negative."
    assert np_all((PC >= 0) | isnan(PC)), "CO2 partial pressure cannot be negative."
    assert np_all((FC >= 0) | isnan(FC)), "CO2 fugacity cannot be negative."
    assert np_all(
        (CARB >= 0) | isnan(CARB)
    ), "Carbonate ion molinity cannot be negative."
    assert np_all(
        (HCO3 >= 0) | isnan(HCO3)
    ), "Bicarbonate ion molinity cannot be negative."
    assert np_all((CO2 >= 0) | isnan(CO2)), "Aqueous CO2 molinity cannot be negative."
    assert np_all(
        (CARB < TC) | isnan(CARB) | isnan(TC)
    ), "Carbonate ion molinity must be less than DIC."
    assert np_all(
        (HCO3 < TC) | isnan(HCO3) | isnan(TC)
    ), "Bicarbonate ion molinity must be less than DIC."
    assert np_all(
        (CO2 < TC) | isnan(CO2) | isnan(TC)
    ), "Aqueous CO2 molinity must be less than DIC."


def pair2core(par1, par2, par1type, par2type, convert_units=False, checks=True):
    """Expand `par1` and `par2` inputs into one array per core variable of the marine
    carbonate system.  Convert units from microX to X if requested with the input
    logical `convertunits`.
    """
    assert (
        size(par1) == size(par2) == size(par1type) == size(par2type)
    ), "`par1`, `par2`, `par1type` and `par2type` must all be the same size."
    ntps = size(par1)
    # Generate empty vectors for...
    TA = full(ntps, nan)  # total alkalinity
    TC = full(ntps, nan)  # dissolved inorganic carbon
    PH = full(ntps, nan)  # pH
    PC = full(ntps, nan)  # CO2 partial pressure
    FC = full(ntps, nan)  # CO2 fugacity
    CARB = full(ntps, nan)  # carbonate ions
    HCO3 = full(ntps, nan)  # bicarbonate ions
    CO2 = full(ntps, nan)  # aqueous CO2
    # Assign values to empty vectors & convert micro[mol|atm] to [mol|atm] if requested
    assert isinstance(convert_units, bool), "`convert_units` must be `True` or `False`."
    if convert_units:
        cfac = 1e-6
    else:
        cfac = 1.0
    TA = where(par1type == 1, par1 * cfac, TA)
    TC = where(par1type == 2, par1 * cfac, TC)
    PH = where(par1type == 3, par1, PH)
    PC = where(par1type == 4, par1 * cfac, PC)
    FC = where(par1type == 5, par1 * cfac, FC)
    CARB = where(par1type == 6, par1 * cfac, CARB)
    HCO3 = where(par1type == 7, par1 * cfac, HCO3)
    CO2 = where(par1type == 8, par1 * cfac, CO2)
    TA = where(par2type == 1, par2 * cfac, TA)
    TC = where(par2type == 2, par2 * cfac, TC)
    PH = where(par2type == 3, par2, PH)
    PC = where(par2type == 4, par2 * cfac, PC)
    FC = where(par2type == 5, par2 * cfac, FC)
    CARB = where(par2type == 6, par2 * cfac, CARB)
    HCO3 = where(par2type == 7, par2 * cfac, HCO3)
    CO2 = where(par2type == 8, par2 * cfac, CO2)
    if checks:
        _core_sanity(TC, PC, FC, CARB, HCO3, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2


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

    `Icase` is `10*parXtype + parYtype` where `parXtype` is whichever of `par1type` or
    `par2type` is greater.

    Noting that a pair of any two from pCO2, fCO2 and CO2(aq) is not allowed, the valid
    `Icase` options are therefore:

        12, 13, 14, 15, 16, 17, 18,
            23, 24, 25, 26, 27, 28,
                34, 35, 36, 37, 38,
                        46, 47,
                        56, 57,
                            67, 68,
                                78.

    The optional input `checks` allows you to decide whether the function should test
    the validity of the entered combinations or not.
    """
    # Check validity of separate `par1type` and `par2type` inputs
    if checks:
        assert np_all(
            isin(par1type, [1, 2, 3, 4, 5, 6, 7, 8])
            & isin(par2type, [1, 2, 3, 4, 5, 6, 7, 8])
        ), "All `par1type` and `par2type` values must be integers from 1 to 8."
        assert ~np_any(
            par1type == par2type
        ), "`par1type` and `par2type` must be different from each other."
    # Combine inputs into `Icase` and check its validity
    Icase = where(
        par1type < par2type, 10 * par1type + par2type, par1type + 10 * par2type
    )
    if checks:
        assert ~np_any(
            isin(Icase, [45, 48, 58])
        ), "Combinations of pCO2, fCO2 and CO2(aq) are not valid input pairs."
    return Icase


def fill(Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks):
    """Fill part-empty core marine carbonate system variable columns with solutions."""
    # For convenience
    K0 = Ks["K0"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    PengCx = totals["PengCorrection"]
    # Convert any pCO2 and CO2(aq) values into fCO2
    PCgiven = isin(Icase, [14, 24, 34, 46, 47])
    FC = where(PCgiven, PC * Ks["FugFac"], FC)
    CO2given = isin(Icase, [18, 28, 38, 68, 78])
    FC = where(CO2given, CO2 / K0, FC)
    # Solve the marine carbonate system
    F = Icase == 12  # input TA, TC
    if any(F):
        PH = where(F, get.pHfromTATC(TA - PengCx, TC, totals, Ks), PH)
        # ^pH is returned on the same scale as `Ks`
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 13  # input TA, pH
    if any(F):
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = (Icase == 14) | (Icase == 15) | (Icase == 18)  # input TA, [pCO2|fCO2|CO2aq]
    if any(F):
        PH = where(F, get.pHfromTAfCO2(TA - PengCx, FC, totals, Ks), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 16  # input TA, CARB
    if any(F):
        PH = where(F, get.pHfromTACarb(TA - PengCx, CARB, totals, Ks), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 17  # input TA, HCO3
    if any(F):
        PH = where(F, get.pHfromTAHCO3(TA - PengCx, HCO3, totals, Ks), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, totals, Ks), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = Icase == 23  # input TC, pH
    if any(F):
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = (Icase == 24) | (Icase == 25) | (Icase == 28)  # input TC, [pCO2|fCO2|CO2aq]
    if any(F):
        PH = where(F, get.pHfromTCfCO2(TC, FC, K0, K1, K2), PH)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 26  # input TC, CARB
    if any(F):
        PH = where(F, get.pHfromTCCarb(TC, CARB, K1, K2), PH)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 27  # input TC, HCO3
    if any(F):
        PH = where(F, get.pHfromTCHCO3(TC, HCO3, K1, K2), PH)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = (Icase == 34) | (Icase == 35) | (Icase == 38)  # input pH, [pCO2|fCO2|CO2aq]
    if any(F):
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 36  # input pH, CARB
    if any(F):
        FC = where(F, get.fCO2frompHCarb(PH, CARB, K0, K1, K2), FC)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 37  # input pH, HCO3
    if any(F):
        TC = where(F, get.TCfrompHHCO3(PH, HCO3, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = (Icase == 46) | (Icase == 56) | (Icase == 68)  # input [pCO2|fCO2|CO2aq], CARB
    if any(F):
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 67  # input CO3, HCO3
    if any(F):
        FC = where(F, get.fCO2fromCarbHCO3(CARB, HCO3, K0, K1, K2), FC)
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
    F = (Icase == 47) | (Icase == 57) | (Icase == 78)  # input [pCO2|fCO2|CO2aq], HCO3
    if any(F):
        CARB = where(F, get.CarbfromfCO2HCO3(FC, HCO3, K0, K1, K2), CARB)
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, totals, Ks) + PengCx, TA)
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 and CO2(aq) values:
    PC = where(~PCgiven, FC / Ks["FugFac"], PC)
    CO2 = where(~CO2given, FC * K0, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2


def core(par1, par2, par1type, par2type, totals, Ks, convert_units):
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

    The input `convert_units` specifies whether the inputs `par1` and `par2` are in
    μmol/kg and μatm units (`True`) or mol/kg and atm units (`False`).
    """
    # Expand inputs `par1` and `par2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pair2core(
        par1, par2, par1type, par2type, convert_units=convert_units, checks=True
    )
    # Generate vector describing the combination(s) of input parameters
    Icase = getIcase(par1type, par2type)
    # Solve the core marine carbonate system
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = fill(
        Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, totals, Ks
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
    }


def others(
    core_solved, TempC, Pdbar, totals, Ks, pHScale, WhichKs, buffers_mode,
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
    pK1 = -log10(Ks["K1"])
    pK2 = -log10(Ks["K2"])
    # Components of alkalinity and DIC
    FREEtoTOT = convert.free2tot(totals["TSO4"], Ks["KSO4"])
    alks = get.AlkParts(TC, PH, FREEtoTOT, totals, Ks)
    alks["PAlk"] = alks["PAlk"] + totals["PengCorrection"]
    # CaCO3 solubility
    OmegaCa, OmegaAr = solubility.CaCO3(Sal, TempC, Pdbar, CARB, totals["TCa"], WhichKs)
    # Dry mole fraction of CO2
    VPFac = gas.vpfactor(TempC, Sal)
    xCO2dry = PC / VPFac  # this assumes pTot = 1 atm
    # Just for reference, convert pH at input conditions to the other scales
    pHT, pHS, pHF, pHN = convert.pH2allscales(
        PH, pHScale, Ks["KSO4"], Ks["KF"], totals["TSO4"], totals["TF"], Ks["fH"]
    )
    # Get buffers as and if requested
    assert all(
        isin(buffers_mode, ["auto", "explicit", "none"])
    ), "Valid options for buffers_mode are 'auto', 'explicit' or 'none'."
    isoQx = full(size(Sal), nan)
    isoQ = full(size(Sal), nan)
    Revelle = full(size(Sal), nan)
    psi = full(size(Sal), nan)
    esm10buffers = [
        "gammaTC",
        "betaTC",
        "omegaTC",
        "gammaTA",
        "betaTA",
        "omegaTA",
    ]
    allbuffers_ESM10 = {buffer: full(size(Sal), nan) for buffer in esm10buffers}
    F = buffers_mode == "auto"
    if any(F):
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
            allbuffers_ESM10[buffer] = where(
                F, auto_ESM10[buffer], allbuffers_ESM10[buffer]
            )
        isoQ = where(F, buffers.isocap(TAPeng, TC, PH, FC, totals, Ks), isoQ)
        Revelle = where(
            F, buffers.RevelleFactor_ESM10(TC, allbuffers_ESM10["gammaTC"]), Revelle
        )
    F = buffers_mode == "explicit"
    if any(F):
        # Evaluate buffers with explicit equations, but these don't include nutrients
        # (i.e. only carbonate, borate and water alkalinities are accounted for)
        expl_ESM10 = buffers.explicit.all_ESM10(
            TC, TAPeng, CO2, HCO3, CARB, PH, alks["OH"], alks["BAlk"], Ks["KB"],
        )
        for buffer in esm10buffers:
            allbuffers_ESM10[buffer] = where(
                F, expl_ESM10[buffer], allbuffers_ESM10[buffer]
            )
        isoQ = where(
            F,
            buffers.explicit.isocap(
                CO2, PH, Ks["K1"], Ks["K2"], Ks["KB"], Ks["KW"], totals["TB"]
            ),
            isoQ,
        )
        Revelle = where(
            F, buffers.explicit.RevelleFactor(TAPeng, TC, totals, Ks), Revelle
        )
    F = buffers_mode != "none"
    if any(F):
        # Approximate isocapnic quotient of HDW18
        isoQx = where(
            F,
            buffers.explicit.isocap_approx(TC, PC, Ks["K0"], Ks["K1"], Ks["K2"]),
            isoQx,
        )
        # psi of FCG94 following HDW18
        psi = where(F, buffers.psi(isoQ), psi)
    # Substrate:inhibitor ratio of B15
    SIR = bio.SIratio(HCO3, pHF)
    return {
        "pK1": pK1,
        "pK2": pK2,
        "BAlk": alks["BAlk"],
        "OH": alks["OH"],
        "PAlk": alks["PAlk"],
        "SiAlk": alks["SiAlk"],
        "NH3Alk": alks["NH3Alk"],
        "H2SAlk": alks["H2SAlk"],
        "Hfree": alks["Hfree"],
        "HSO4": alks["HSO4"],
        "HF": alks["HF"],
        "OmegaCa": OmegaCa,
        "OmegaAr": OmegaAr,
        "VPFac": VPFac,
        "xCO2dry": xCO2dry,
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
        # Added in v1.4.0
        "SIR": SIR,
    }
