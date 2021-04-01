# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Helpers for the main CO2SYS program."""

from autograd import numpy as np
from .. import convert, equilibria, salts, solve
from . import nd


def condition(input_locals, npts=None):
    """Condition inputs for use with CO2SYS (sub)functions."""
    # Determine and check lengths of input vectors
    veclengths = np.array([np.size(v) for v in input_locals.values()])
    assert (
        np.size(np.unique(veclengths[veclengths != 1])) <= 1
    ), "CO2SYS function inputs must all be of same length, or of length 1."
    # Make vectors of all inputs
    npts_in = np.max(veclengths)
    if npts is None:
        npts = npts_in
    else:
        assert npts_in in (
            1,
            npts,
        ), "Input `npts` does not agree with input array sizes."
    args = {
        k: np.full(npts, v) if np.size(v) == 1 else v.ravel()
        for k, v in input_locals.items()
    }
    # Convert to float where appropriate
    float_vars = {
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
        "XC",
        "TempC",
        "Pdbar",
        "TSi",
        "TPO4",
        "TNH3",
        "TH2S",
        "TB",
        "TF",
        "TS",
        "TCa",
        "K0",
        "K1",
        "K2",
        "KW",
        "KB",
        "KF",
        "KS",
        "KP1",
        "KP2",
        "KP3",
        "KSi",
        "KNH3",
        "KH2S",
        "RGas",
        "KCa",
        "KAr",
    }
    for k in args.keys():
        if k in float_vars:
            args[k] = args[k].astype("float64")
    return args, npts


# Define all gradable outputs
gradables = np.array(
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
        "RGas",
        "KCainput",
        "KCaoutput",
        "KArinput",
        "KAroutput",
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
        "xCO2in": core_in["XC"] * 1e6,
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
        "xCO2out": core_out["XC"] * 1e6,
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
        "SAL": totals["Sal"],  # v1.4.0: take from totals, not args
        "PO4": totals["TPO4"] * 1e6,  # v1.4.0: take from totals, not args
        "SI": totals["TSi"] * 1e6,  # v1.4.0: take from totals, not args
        "NH3": totals["TNH3"] * 1e6,  # v1.4.0: take from totals, not args
        "H2S": totals["TH2S"] * 1e6,  # v1.4.0: take from totals, not args
        "K0input": Kis["K0"],
        "K1input": Kis["K1"],
        "K2input": Kis["K2"],
        "pK1input": others_in["pK1"],
        "pK2input": others_in["pK2"],
        "KWinput": Kis["KW"],
        "KBinput": Kis["KB"],
        "KFinput": Kis["KF"],
        "KSinput": Kis["KSO4"],  # to be replaced by "KSO4input"
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
        "KSoutput": Kos["KSO4"],  # to be replaced by "KSO4output"
        "KP1output": Kos["KP1"],
        "KP2output": Kos["KP2"],
        "KP3output": Kos["KP3"],
        "KSioutput": Kos["KSi"],
        "KNH3output": Kos["KNH3"],
        "KH2Soutput": Kos["KH2S"],
        "TB": totals["TB"] * 1e6,
        "TF": totals["TF"] * 1e6,
        "TS": totals["TSO4"] * 1e6,  # to be replaced by "TSO4"
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
        "TSO4": totals["TSO4"] * 1e6,  # will eventually replace "TS"
        "KSO4input": Kis["KSO4"],  # will eventually replace "KSinput"
        "KSO4output": Kos["KSO4"],  # will eventually replace "KSoutput"
        # Added in v1.4.1:
        "RGas": Kis["RGas"],
        # Added in v1.5.0:
        "KCainput": Kis["KCa"],
        "KCaoutput": Kos["KCa"],
        "KArinput": Kis["KAr"],
        "KAroutput": Kos["KAr"],
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
        # Added in v1.4.1:
        "WhichR": args["WhichR"],
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
    WhichR,
    KSO4CONSTANTS=0,
    totals=None,
    equilibria_in=None,
    equilibria_out=None,
):
    # Aliases
    Kis = equilibria_in
    Kos = equilibria_out
    # Condition inputs and assign input values to the 'historical' variable names
    args, npts = condition(
        {
            "PAR1": PAR1,
            "PAR2": PAR2,
            "PAR1TYPE": PAR1TYPE,
            "PAR2TYPE": PAR2TYPE,
            "SAL": SAL,
            "TEMPIN": TEMPIN,
            "TEMPOUT": TEMPOUT,
            "PRESIN": PRESIN,
            "PRESOUT": PRESOUT,
            "SI": SI,
            "PO4": PO4,
            "NH3": NH3,
            "H2S": H2S,
            "pHSCALEIN": pHSCALEIN,
            "K1K2CONSTANTS": K1K2CONSTANTS,
            "KSO4CONSTANT": KSO4CONSTANT,
            "KFCONSTANT": KFCONSTANT,
            "BORON": BORON,
            "buffers_mode": buffers_mode,
            "KSO4CONSTANTS": KSO4CONSTANTS,
            "WhichR": WhichR,
        }
    )
    PAR1 = args["PAR1"]
    PAR2 = args["PAR2"]
    p1 = args["PAR1TYPE"]
    p2 = args["PAR2TYPE"]
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
    WhichR = args["WhichR"]
    # Prepare to solve the core marine carbonate system at input conditions
    if totals is not None:
        totals = condition(totals, npts=npts)[0]
        totals = {k: v * 1e-6 for k, v in totals.items() if k != "SAL"}
    totals = salts.assemble(
        args["SAL"], TSi, TP, TNH3, TH2S, WhichKs, WhoseTB, totals=totals
    )
    if Kis is not None:
        Kis = condition(Kis, npts=npts)[0]
    Kis = equilibria.assemble(
        TempCi, Pdbari, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF, WhichR, Ks=Kis
    )
    if Kos is not None:
        Kos = condition(Kos, npts=npts)[0]
    Kos = equilibria.assemble(
        TempCo, Pdbaro, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF, WhichR, Ks=Kos
    )
    # Solve the core marine carbonate system at input conditions
    core_in = solve.core(PAR1, PAR2, p1, p2, totals, Kis, True)
    # Calculate all other results at input conditions
    others_in = solve.others(
        core_in, TempCi, Pdbari, totals, Kis, pHScale, WhichKs, buffers_mode,
    )
    # Solve the core MCS at output conditions
    TAtype = np.full(npts, 1)
    TCtype = np.full(npts, 2)
    core_out = solve.core(
        core_in["TA"], core_in["TC"], TAtype, TCtype, totals, Kos, False,
    )
    # Calculate all other results at output conditions
    others_out = solve.others(
        core_out, TempCo, Pdbaro, totals, Kos, pHScale, WhichKs, buffers_mode,
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
    totals=None,
    equilibria_in=None,
    equilibria_out=None,
    WhichR=1,
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
        WhichR,
        KSO4CONSTANTS=KSO4CONSTANTS,
        totals=totals,
        equilibria_in=equilibria_in,
        equilibria_out=equilibria_out,
    )


def dict2totals(co2dict):
    """Extract `totals` dict in mol/kg from the `CO2SYS` output dict."""
    return dict(
        # From salinity
        TB=co2dict["TB"] * 1e-6,
        TF=co2dict["TF"] * 1e-6,
        TSO4=co2dict["TSO4"] * 1e-6,
        TCa=co2dict["TCa"] * 1e-6,
        # From inputs
        TPO4=co2dict["PO4"] * 1e-6,
        TSi=co2dict["SI"] * 1e-6,
        TNH3=co2dict["NH3"] * 1e-6,
        TH2S=co2dict["H2S"] * 1e-6,
        # Miscellaneous
        Sal=co2dict["SAL"],
        PengCorrection=co2dict["PengCorrection"] * 1e-6,
    )


def dict2totals_umol(co2dict):
    """Extract `totals` dict in μmol/kg from the `CO2SYS` output dict."""
    return dict(
        # From salinity
        TB=co2dict["TB"],
        TF=co2dict["TF"],
        TSO4=co2dict["TSO4"],
        TCa=co2dict["TCa"],
        # From inputs
        TPO4=co2dict["PO4"],
        TSi=co2dict["SI"],
        TNH3=co2dict["NH3"],
        TH2S=co2dict["H2S"],
        # Miscellaneous
        Sal=co2dict["SAL"],
        PengCorrection=co2dict["PengCorrection"],
    )


def dict2equilibria(co2dict):
    """Extract `Kis`/`equilibria_in` and `Kos`/`equilibria_out` dicts from the
    `CO2SYS` output dict.
    """
    Kvars = [
        "K0",
        "K1",
        "K2",
        "KW",
        "KB",
        "KF",
        "KSO4",
        "KP1",
        "KP2",
        "KP3",
        "KSi",
        "KNH3",
        "KH2S",
        "FugFac",
        "fH",
        "KCa",
        "KAr",
    ]
    Kis = {Kvar: co2dict[Kvar + "input"] for Kvar in Kvars}
    Kos = {Kvar: co2dict[Kvar + "output"] for Kvar in Kvars}
    RGas = {"RGas": co2dict["RGas"]}
    Kis.update(RGas)
    Kos.update(RGas)
    return Kis, Kos
