# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Helpers for the main CO2SYS program."""

from autograd.numpy import array, full, isin, nan, shape, size, unique, where
from autograd.numpy import all as np_all
from autograd.numpy import any as np_any
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max
from . import equilibria, gas, salts, solve


def inputs(input_locals):
    """Condition inputs for use with CO2SYS (sub)functions."""
    # Determine and check lengths of input vectors
    veclengths = array([size(v) for v in input_locals.values()])
    assert (
        size(unique(veclengths[veclengths != 1])) <= 1
    ), "CO2SYS function inputs must all be of same length, or of length 1."
    # Make vectors of all inputs
    ntps = max(veclengths)
    args = {
        k: full(ntps, v) if size(v) == 1 else v.ravel() for k, v in input_locals.items()
    }
    # Convert to float where appropriate
    float_vars = [
        "SAL",
        "TEMPIN",
        "TEMPOUT",
        "PRESIN",
        "PRESOUT",
        "SI",
        "PO4",
        "NH3",
        "H2S",
        "PAR1",
        "PAR2",
    ]
    for k in args.keys():
        if k in float_vars:
            args[k] = args[k].astype("float64")
    return args, ntps


def pair2core(par1, par2, par1type, par2type, convertunits):
    """Expand `par1` and `par2` inputs into one array per core variable of the marine
    carbonate system and convert units from microX to X if requested with the input
    logical `convertunits`.
    """
    assert (
        size(par1) == size(par2) == size(par1type) == size(par2type)
    ), "`par1`, `par2`, `par1type` and `par2type` must all be the same size."
    ntps = size(par1)
    # Generate empty vectors for...
    TA = full(ntps, nan)  # total alkalinity
    TC = full(ntps, nan)  # DIC
    PH = full(ntps, nan)  # pH
    PC = full(ntps, nan)  # CO2 partial pressure
    FC = full(ntps, nan)  # CO2 fugacity
    CARB = full(ntps, nan)  # carbonate ions
    HCO3 = full(ntps, nan)  # bicarbonate ions
    CO2 = full(ntps, nan)  # aqueous CO2
    # Assign values to empty vectors & convert micro[mol|atm] to [mol|atm] if requested
    assert type(convertunits) == bool, "`convertunits` must be `True` or `False`."
    if convertunits:
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
    Iarr = array([par1type, par2type])
    if checks:
        assert np_all(
            isin(Iarr, [1, 2, 3, 4, 5, 6, 7, 8])
        ), "All `par1type` and `par2type` values must be integers from 1 to 8."
        assert ~np_any(
            par1type == par2type
        ), "`par1type` and `par2type` must be different from each other."
    # Combine inputs into `Icase` and check its validity
    Icase = 10 * np_min(Iarr, axis=0) + np_max(Iarr, axis=0)
    if checks:
        assert ~np_any(
            isin(Icase, [45, 48, 58])
        ), "Combinations of pCO2, fCO2 and CO2(aq) are not valid input pairs."
    return Icase


def solvecore(par1, par2, par1type, par2type, totals, FugFac, Ks, convertunits):
    """Solve the core marine carbonate system (MCS) from any 2 of its variables.

    The core MCS outputs (in a dict) and associated `par1type`/`par2type` inputs are:

      * Type `1`, `TA`: total alkalinity in mol/kg-sw.
      * Type `2`, `TC`: dissolved inorganic carbon in mol/kg-sw.
      * Type `3`, `PH`: pH on whichever scale(s) the constants in `Ks` are provided.
      * Type `4`, `PC`: partial pressure of CO2 in atm.
      * Type `5`, `FC`: fugacity of CO2 in atm.
      * Type `6`, `CARB`: carbonate ion in mol/kg-sw.
      * Type `7`, `HCO3`: bicarbonate ion in mol/kg-sw.
      * Type `8`, `CO2`: aqueous CO2 in mol/kg-sw.

    The input `convertunits` specifies whether the inputs `par1` and `par2` are in
    micro-mol/kg and micro-atm units (`True`) or mol/kg and atm units (`False`).
    """
    # Expand inputs `par1` and `par2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pair2core(
        par1, par2, par1type, par2type, convertunits
    )
    # Generate vector describing the combination(s) of input parameters
    Icase = getIcase(par1type, par2type)
    # Solve the core marine carbonate system
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = solve.core(
        Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFac, Ks, totals
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


def _CO2SYS(
    PAR1,
    PAR2,
    PAR1TYPE,
    PAR2TYPE,
    SAL,
    TEMPIN,
    TEMPOUT,
    PRESIN,
    PRESOUT,
    SI,
    PO4,
    NH3,
    H2S,
    pHSCALEIN,
    K1K2CONSTANTS,
    KSO4CONSTANT,
    KFCONSTANT,
    BORON,
    buffers_mode,
    KSO4CONSTANTS=0,
):
    # Condition inputs and assign input values to the 'historical' variable names
    args, ntps = inputs(locals())
    PAR1 = args["PAR1"]
    PAR2 = args["PAR2"]
    p1 = args["PAR1TYPE"]
    p2 = args["PAR2TYPE"]
    Sal = args["SAL"]
    TempCi = args["TEMPIN"]
    TempCo = args["TEMPOUT"]
    Pdbari = args["PRESIN"]
    Pdbaro = args["PRESOUT"]
    TSi = args["SI"]
    TP = args["PO4"]
    TNH3 = args["NH3"]
    TH2S = args["H2S"]
    pHScale = args["pHSCALEIN"]
    WhichKs = args["K1K2CONSTANTS"]
    WhoseKSO4 = args["KSO4CONSTANT"]
    WhoseKF = args["KFCONSTANT"]
    WhoseTB = args["BORON"]
    buffers_mode = args["buffers_mode"]
    # Prepare to solve the core marine carbonate system at input conditions
    totals = salts.assemble(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB)
    Sal = totals["Sal"]
    Kis = equilibria.assemble(
        TempCi, Pdbari, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    # Calculate fugacity factor
    FugFaci = gas.fugacityfactor(TempCi, WhichKs)
    # Solve the core marine carbonate system at input conditions
    core_in = solvecore(PAR1, PAR2, p1, p2, totals, FugFaci, Kis, True)
    # Calculate all other results at input conditions
    others_in = solve.others(
        core_in, Sal, TempCi, Pdbari, Kis, totals, pHScale, WhichKs, buffers_mode,
    )
    # Prepare to solve the core MCS at output conditions
    Kos = equilibria.assemble(
        TempCo, Pdbaro, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    # Calculate fugacity factor
    FugFaco = gas.fugacityfactor(TempCo, WhichKs)
    TAtype = full(ntps, 1)
    TCtype = full(ntps, 2)
    # Solve the core MCS at output conditions
    core_out = solvecore(
        core_in["TA"], core_in["TC"], TAtype, TCtype, totals, FugFaco, Kos, False,
    )
    # Calculate all other results at output conditions
    others_out = solve.others(
        core_out, Sal, TempCo, Pdbaro, Kos, totals, pHScale, WhichKs, buffers_mode,
    )
    # Save data directly as a dict to avoid ordering issues
    CO2dict = {
        "TAlk": core_in["TA"] * 1e6,
        "TCO2": core_in["TC"] * 1e6,
        "pHin": core_in["PH"],
        "pCO2in": core_in["PC"] * 1e6,
        "fCO2in": core_in["FC"] * 1e6,
        "HCO3in": core_in["HCO3"] * 1e6,
        "CO3in": core_in["CARB"] * 1e6,
        "CO2in": core_in["CO2"] * 1e6,
        "BAlkin": others_in["BAlk"] * 1e6,
        "OHin": others_in["OH"] * 1e6,
        "PAlkin": others_in["PAlk"] * 1e6,
        "SiAlkin": others_in["SiAlk"] * 1e6,
        "NH3Alkin": others_in["NH3Alk"] * 1e6,
        "H2SAlkin": others_in["H2SAlk"] * 1e6,
        "Hfreein": others_in["Hfree"] * 1e6,
        "RFin": others_in["Revelle"],
        "OmegaCAin": others_in["OmegaCa"],
        "OmegaARin": others_in["OmegaAr"],
        "xCO2in": others_in["xCO2dry"] * 1e6,
        "pHout": core_out["PH"],
        "pCO2out": core_out["PC"] * 1e6,
        "fCO2out": core_out["FC"] * 1e6,
        "HCO3out": core_out["HCO3"] * 1e6,
        "CO3out": core_out["CARB"] * 1e6,
        "CO2out": core_out["CO2"] * 1e6,
        "BAlkout": others_out["BAlk"] * 1e6,
        "OHout": others_out["OH"] * 1e6,
        "PAlkout": others_out["PAlk"] * 1e6,
        "SiAlkout": others_out["SiAlk"] * 1e6,
        "NH3Alkout": others_out["NH3Alk"] * 1e6,
        "H2SAlkout": others_out["H2SAlk"] * 1e6,
        "Hfreeout": others_out["Hfree"] * 1e6,
        "RFout": others_out["Revelle"],
        "OmegaCAout": others_out["OmegaCa"],
        "OmegaARout": others_out["OmegaAr"],
        "xCO2out": others_out["xCO2dry"] * 1e6,
        "pHinTOTAL": others_in["pHT"],
        "pHinSWS": others_in["pHS"],
        "pHinFREE": others_in["pHF"],
        "pHinNBS": others_in["pHN"],
        "pHoutTOTAL": others_out["pHT"],
        "pHoutSWS": others_out["pHS"],
        "pHoutFREE": others_out["pHF"],
        "pHoutNBS": others_out["pHN"],
        "TEMPIN": args["TEMPIN"],
        "TEMPOUT": args["TEMPOUT"],
        "PRESIN": args["PRESIN"],
        "PRESOUT": args["PRESOUT"],
        "PAR1TYPE": args["PAR1TYPE"],
        "PAR2TYPE": args["PAR2TYPE"],
        "K1K2CONSTANTS": args["K1K2CONSTANTS"],
        "KSO4CONSTANTS": args["KSO4CONSTANTS"],
        "KSO4CONSTANT": args["KSO4CONSTANT"],
        "KFCONSTANT": args["KFCONSTANT"],
        "BORON": args["BORON"],
        "pHSCALEIN": args["pHSCALEIN"],
        "SAL": args["SAL"],
        "PO4": args["PO4"],
        "SI": args["SI"],
        "NH3": args["NH3"],
        "H2S": args["H2S"],
        "K0input": Kis["K0"],
        "K1input": Kis["K1"],
        "K2input": Kis["K2"],
        "pK1input": others_in["pK1"],
        "pK2input": others_in["pK2"],
        "KWinput": Kis["KW"],
        "KBinput": Kis["KB"],
        "KFinput": Kis["KF"],
        "KSinput": Kis["KSO4"],
        "KP1input": Kis["KP1"],
        "KP2input": Kis["KP2"],
        "KP3input": Kis["KP3"],
        "KSiinput": Kis["KSi"],
        "KNH3input": Kis["KNH3"],
        "KH2Sinput": Kis["KH2S"],
        "K0output": Kos["K0"],
        "K1output": Kos["K1"],
        "K2output": Kos["K2"],
        "pK1output": others_out["pK1"],
        "pK2output": others_out["pK2"],
        "KWoutput": Kos["KW"],
        "KBoutput": Kos["KB"],
        "KFoutput": Kos["KF"],
        "KSoutput": Kos["KSO4"],
        "KP1output": Kos["KP1"],
        "KP2output": Kos["KP2"],
        "KP3output": Kos["KP3"],
        "KSioutput": Kos["KSi"],
        "KNH3output": Kos["KNH3"],
        "KH2Soutput": Kos["KH2S"],
        "TB": totals["TB"] * 1e6,
        "TF": totals["TF"] * 1e6,
        "TS": totals["TSO4"] * 1e6,
        # Added in v1.2.0:
        "gammaTCin": others_in["gammaTC"],
        "betaTCin": others_in["betaTC"],
        "omegaTCin": others_in["omegaTC"],
        "gammaTAin": others_in["gammaTA"],
        "betaTAin": others_in["betaTA"],
        "omegaTAin": others_in["omegaTA"],
        "gammaTCout": others_out["gammaTC"],
        "betaTCout": others_out["betaTC"],
        "omegaTCout": others_out["omegaTC"],
        "gammaTAout": others_out["gammaTA"],
        "betaTAout": others_out["betaTA"],
        "omegaTAout": others_out["omegaTA"],
        "isoQin": others_in["isoQ"],
        "isoQout": others_out["isoQ"],
        "isoQapprox_in": others_in["isoQx"],
        "isoQapprox_out": others_out["isoQx"],
        "psi_in": others_in["psi"],
        "psi_out": others_out["psi"],
        # Added in v1.3.0:
        "TCa": totals["TCa"] * 1e6,
        "buffers_mode": buffers_mode,
    }
    return CO2dict


def CO2SYS(
    PAR1,
    PAR2,
    PAR1TYPE,
    PAR2TYPE,
    SAL,
    TEMPIN,
    TEMPOUT,
    PRESIN,
    PRESOUT,
    SI,
    PO4,
    pHSCALEIN,
    K1K2CONSTANTS,
    KSO4CONSTANTS,
    NH3=0.0,
    H2S=0.0,
    KFCONSTANT=1,
    buffers_mode="auto",
):
    """Solve the carbonate system using the input parameters.

    Originally based on CO2SYS v1.21 and v2.0.5, both for MATLAB, which have been built
    over many years based on an original program by Ernie Lewis and Doug Wallace, with
    later contributions from S.M.A.C. van Heuven, J.W.B. Rae, J.C. Orr, J.-M. Epitalon,
    A.G. Dickson, J.-P. Gattuso, and D. Pierrot.  Translated into Python and
    subsequently extended by M.P. Humphreys.
    """
    # Convert traditional inputs to new format before running CO2SYS
    if shape(KSO4CONSTANTS) == ():
        KSO4CONSTANTS = array([KSO4CONSTANTS])
    only2KSO4 = {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
    }
    only2BORON = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
    }
    KSO4CONSTANT = array([only2KSO4[K] for K in KSO4CONSTANTS.ravel()])
    BORON = array([only2BORON[K] for K in KSO4CONSTANTS.ravel()])
    return _CO2SYS(
        PAR1,
        PAR2,
        PAR1TYPE,
        PAR2TYPE,
        SAL,
        TEMPIN,
        TEMPOUT,
        PRESIN,
        PRESOUT,
        SI,
        PO4,
        NH3,
        H2S,
        pHSCALEIN,
        K1K2CONSTANTS,
        KSO4CONSTANT,
        KFCONSTANT,
        BORON,
        buffers_mode,
        KSO4CONSTANTS=KSO4CONSTANTS,
    )
