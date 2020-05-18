# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Helpers for the main CO2SYS program."""

from autograd.numpy import (
    array,
    full,
    size,
    unique,
)
from autograd.numpy import max as np_max
from autograd import elementwise_grad as egrad
from . import convert, equilibria, salts, solve, uncertainty


def inputs(input_locals):
    """Condition inputs for use with CO2SYS (sub)functions."""
    # Determine and check lengths of input vectors
    veclengths = array([size(v) for v in input_locals.values()])
    assert (
        size(unique(veclengths[veclengths != 1])) <= 1
    ), "CO2SYS function inputs must all be of same length, or of length 1."
    # Make vectors of all inputs
    ntps = np_max(veclengths)
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
        "TA",
        "TC",
        "PH",
        "PC",
        "FC",
        "CARB",
        "HCO3",
        "CO2",
        "TempC",
        "Pdbar",
        "TSi",
        "TPO4",
        "TNH3",
        "TH2S",
    ]
    for k in args.keys():
        if k in float_vars:
            args[k] = args[k].astype("float64")
    return args, ntps


# Define all gradable outputs
gradables = array(
    [
        "TAlk",
        "TCO2",
        "pHin",
        "pCO2in",
        "fCO2in",
        "HCO3in",
        "CO3in",
        "CO2in",
        "BAlkin",
        "OHin",
        "PAlkin",
        "SiAlkin",
        "NH3Alkin",
        "H2SAlkin",
        "Hfreein",
        "RFin",
        "OmegaCAin",
        "OmegaARin",
        "xCO2in",
        "pHout",
        "pCO2out",
        "fCO2out",
        "HCO3out",
        "CO3out",
        "CO2out",
        "BAlkout",
        "OHout",
        "PAlkout",
        "SiAlkout",
        "NH3Alkout",
        "H2SAlkout",
        "Hfreeout",
        "RFout",
        "OmegaCAout",
        "OmegaARout",
        "xCO2out",
        "pHinTOTAL",
        "pHinSWS",
        "pHinFREE",
        "pHinNBS",
        "pHoutTOTAL",
        "pHoutSWS",
        "pHoutFREE",
        "pHoutNBS",
        "TEMPIN",
        "TEMPOUT",
        "PRESIN",
        "PRESOUT",
        "SAL",
        "PO4",
        "SI",
        "NH3",
        "H2S",
        "K0input",
        "K1input",
        "K2input",
        "pK1input",
        "pK2input",
        "KWinput",
        "KBinput",
        "KFinput",
        "KSinput",
        "KP1input",
        "KP2input",
        "KP3input",
        "KSiinput",
        "KNH3input",
        "KH2Sinput",
        "K0output",
        "K1output",
        "K2output",
        "pK1output",
        "pK2output",
        "KWoutput",
        "KBoutput",
        "KFoutput",
        "KSoutput",
        "KP1output",
        "KP2output",
        "KP3output",
        "KSioutput",
        "KNH3output",
        "KH2Soutput",
        "TB",
        "TF",
        "TS",
        "gammaTCin",
        "betaTCin",
        "omegaTCin",
        "gammaTAin",
        "betaTAin",
        "omegaTAin",
        "gammaTCout",
        "betaTCout",
        "omegaTCout",
        "gammaTAout",
        "betaTAout",
        "omegaTAout",
        "isoQin",
        "isoQout",
        "isoQapprox_in",
        "isoQapprox_out",
        "psi_in",
        "psi_out",
        "TCa",
        "SIRin",
        "SIRout",
        "PAR1",
        "PAR2",
        "PengCorrection",
        "FugFacinput",
        "FugFacoutput",
        "fHinput",
        "fHoutput",
    ]
)


def _outputs_grad(args, core_in, core_out, others_in, others_out, totals, Kis, Kos):
    """Assemble Autograd-able portion of CO2SYS's output dict."""
    return {
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
        # Added in v1.4.0:
        "SIRin": others_in["SIR"],
        "SIRout": others_out["SIR"],
        "PAR1": args["PAR1"],
        "PAR2": args["PAR2"],
        "PengCorrection": totals["PengCorrection"] * 1e6,
        "FugFacinput": Kis["FugFac"],
        "FugFacoutput": Kos["FugFac"],
        "fHinput": Kis["fH"],
        "fHoutput": Kos["fH"],
    }


def _outputs_nograd(args, buffers_mode):
    """Assemble non-Autograd-able portion of CO2SYS's output dict."""
    return {
        "PAR1TYPE": args["PAR1TYPE"],
        "PAR2TYPE": args["PAR2TYPE"],
        "K1K2CONSTANTS": args["K1K2CONSTANTS"],
        "KSO4CONSTANTS": args["KSO4CONSTANTS"],
        "KSO4CONSTANT": args["KSO4CONSTANT"],
        "KFCONSTANT": args["KFCONSTANT"],
        "BORON": args["BORON"],
        "pHSCALEIN": args["pHSCALEIN"],
        # Added in v1.3.0:
        "buffers_mode": buffers_mode,
    }


def _outputdict(
    args, core_in, core_out, others_in, others_out, totals, Kis, Kos, buffers_mode
):
    """Assemble CO2SYS's complete output dict."""
    outputs_grad = _outputs_grad(
        args, core_in, core_out, others_in, others_out, totals, Kis, Kos
    )
    outputs_nograd = _outputs_nograd(args, buffers_mode)
    return {**outputs_grad, **outputs_nograd}


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
    # Solve the core marine carbonate system at input conditions
    core_in = solve.core(PAR1, PAR2, p1, p2, totals, Kis, True)
    # Calculate all other results at input conditions
    others_in = solve.others(
        core_in, Sal, TempCi, Pdbari, totals, Kis, pHScale, WhichKs, buffers_mode,
    )
    # Solve the core MCS at output conditions
    Kos = equilibria.assemble(
        TempCo, Pdbaro, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    TAtype = full(ntps, 1)
    TCtype = full(ntps, 2)
    core_out = solve.core(
        core_in["TA"], core_in["TC"], TAtype, TCtype, totals, Kos, False,
    )
    # Calculate all other results at output conditions
    others_out = solve.others(
        core_out, Sal, TempCo, Pdbaro, totals, Kos, pHScale, WhichKs, buffers_mode,
    )
    # Save data directly as a dict to avoid ordering issues
    return _outputdict(
        args, core_in, core_out, others_in, others_out, totals, Kis, Kos, buffers_mode
    )


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
    KSO4CONSTANT, BORON = convert.options_old2new(KSO4CONSTANTS)
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


def dict2totals(co2dict):
    """Extract `totals` dict from the `CO2SYS` output dict."""
    return dict(
        # from salinity
        TB=co2dict["TB"] * 1e-6,
        TF=co2dict["TF"] * 1e-6,
        TSO4=co2dict["TS"] * 1e-6,
        TCa=co2dict["TCa"] * 1e-6,
        # from inputs
        TPO4=co2dict["PO4"] * 1e-6,
        TSi=co2dict["SI"] * 1e-6,
        TNH3=co2dict["NH3"] * 1e-6,
        TH2S=co2dict["H2S"] * 1e-6,
        # misc.
        Sal=co2dict["SAL"],
        PengCorrection=co2dict["PengCorrection"] * 1e-6,
    )


def dict2Ks(co2dict):
    """Extract `Kis` and `Kos` dicts from the `CO2SYS` output dict."""
    Kvars = [
        "K0",
        "K1",
        "K2",
        "KW",
        "KB",
        "KF",
        "KP1",
        "KP2",
        "KP3",
        "KSi",
        "KNH3",
        "KH2S",
        "FugFac",
        "fH",
    ]
    Kis = {Kvar: co2dict[Kvar + "input"] for Kvar in Kvars}
    Kis["KSO4"] = co2dict["KSinput"]
    Kos = {Kvar: co2dict[Kvar + "output"] for Kvar in Kvars}
    Kos["KSO4"] = co2dict["KSoutput"]
    return Kis, Kos


def uCO2SYS(co2dict, uncertainties={}):
    """Do uncertainty propagation."""
    # Extract results from the `co2dict` for convenience
    totals = dict2totals(co2dict)
    Kis, Kos = dict2Ks(co2dict)
    # par1 = co2dict["PAR1"]
    # par2 = co2dict["PAR2"]
    par1type = co2dict["PAR1TYPE"]
    par2type = co2dict["PAR2TYPE"]
    # psal = co2dict["SAL"]
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
    dcoreo_dTA__TC = uncertainty.dcore_dparX__parY(
        par1type_TA, par2type_TC, TA, TC, PHo, FCo, CARBo, HCO3o, totals, Kos
    )
    dcoreo_dTC__TA = uncertainty.dcore_dparX__parY(
        par2type_TC, par1type_TA, TA, TC, PHo, FCo, CARBo, HCO3o, totals, Kos
    )
    if "PAR1" in uncertainties:
        dcorei_dp1 = uncertainty.dcore_dparX__parY(
            par1type, par2type, TA, TC, PHi, FCi, CARBi, HCO3i, totals, Kis
        )
        dcoreo_dp1 = {
            k: dcorei_dp1["TA"] * dcoreo_dTA__TC[k]
            + dcorei_dp1["TC"] * dcoreo_dTC__TA[k]
            for k in dcorei_dp1
        }
    if "PAR2" in uncertainties:
        dcorei_dp2 = uncertainty.dcore_dparX__parY(
            par2type, par1type, TA, TC, PHi, FCi, CARBi, HCO3i, totals, Kis
        )
        dcoreo_dp2 = {
            k: dcorei_dp2["TA"] * dcoreo_dTA__TC[k]
            + dcorei_dp2["TC"] * dcoreo_dTC__TA[k]
            for k in dcorei_dp2
        }
    # Merge everything into output dicts
    iosame = ["TA", "TC"]
    dvars_dp1 = {k: dcorei_dp1[k] for k in iosame}
    dvars_dp1.update(
        {"{}i".format(k): v for k, v in dcorei_dp1.items() if k not in iosame}
    )
    dvars_dp1.update(
        {"{}o".format(k): v for k, v in dcoreo_dp1.items() if k not in iosame}
    )
    dvars_dp2 = {k: dcorei_dp2[k] for k in iosame}
    dvars_dp2.update(
        {"{}i".format(k): v for k, v in dcorei_dp2.items() if k not in iosame}
    )
    dvars_dp2.update(
        {"{}o".format(k): v for k, v in dcoreo_dp2.items() if k not in iosame}
    )
    return {"_dPAR1": dvars_dp1, "_dPAR2": dvars_dp2}
