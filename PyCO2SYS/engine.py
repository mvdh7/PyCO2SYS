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
        # need to add to docs below here!
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
    gradable = outputs_grad.keys()
    return {**outputs_grad, **outputs_nograd}, gradable


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
    outputdict, gradable = _outputdict(
        args, core_in, core_out, others_in, others_out, totals, Kis, Kos, buffers_mode
    )
    return outputdict, gradable


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
    )[0]


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
    # Extract results from the `co2dict`
    totals = dict2totals(co2dict)
    Kis, Kos = dict2Ks(co2dict)
    par1type = co2dict["PAR1TYPE"]
    par2type = co2dict["PAR2TYPE"]
    TA = co2dict["TAlk"] * 1e-6
    TC = co2dict["TCO2"] * 1e-6
    PHi = co2dict["pHin"]
    FCi = co2dict["fCO2in"] * 1e-6
    CARBi = co2dict["CO3in"] * 1e-6
    HCO3i = co2dict["HCO3in"] * 1e-6
    # Get par1/part derivatives
    if "PAR1" in uncertainties:
        dcore_dp1__i = uncertainty.dcore_dparX__parY(
            par1type, par2type, TA, TC, PHi, FCi, CARBi, HCO3i, totals, Kis
        )
    if "PAR2" in uncertainties:
        dcore_dp2__i = uncertainty.dcore_dparX__parY(
            par2type, par1type, TA, TC, PHi, FCi, CARBi, HCO3i, totals, Kis
        )
    return dcore_dp1__i, dcore_dp2__i


# def _CO2SYS_u(
#     PAR1,
#     PAR2,
#     PAR1TYPE,
#     PAR2TYPE,
#     SAL,
#     TEMPIN,
#     TEMPOUT,
#     PRESIN,
#     PRESOUT,
#     SI,
#     PO4,
#     NH3,
#     H2S,
#     pHSCALEIN,
#     K1K2CONSTANTS,
#     KSO4CONSTANT,
#     KFCONSTANT,
#     BORON,
#     buffers_mode,
#     KSO4CONSTANTS=0,
#     uncertainty=True,
#     PAR1_U=0,
#     PAR2_U=0,
# ):
#     # Condition inputs and assign input values to the 'historical' variable names
#     args, ntps = inputs(locals())
#     PAR1 = args["PAR1"]
#     PAR2 = args["PAR2"]
#     p1 = args["PAR1TYPE"]
#     p2 = args["PAR2TYPE"]
#     Sal = args["SAL"]
#     TempCi = args["TEMPIN"]
#     TempCo = args["TEMPOUT"]
#     Pdbari = args["PRESIN"]
#     Pdbaro = args["PRESOUT"]
#     TSi = args["SI"]
#     TP = args["PO4"]
#     TNH3 = args["NH3"]
#     TH2S = args["H2S"]
#     pHScale = args["pHSCALEIN"]
#     WhichKs = args["K1K2CONSTANTS"]
#     WhoseKSO4 = args["KSO4CONSTANT"]
#     WhoseKF = args["KFCONSTANT"]
#     WhoseTB = args["BORON"]
#     buffers_mode = args["buffers_mode"]
#     PAR1_U = args["PAR1_U"]
#     PAR2_U = args["PAR2_U"]
#     # Expand inputs `PAR1` and `PAR2` and their uncertainties into one array per core
#     # MCS variable
#     TA, TC, PHi, PCi, FCi, CARBi, HCO3i, CO2i = pair2core(
#         PAR1, PAR2, PAR1TYPE, PAR2TYPE, convert_units=True, checks=True
#     )
#     if uncertainty:
#         TA_U, TC_U, PHi_U, PCi_U, FCi_U, CARBi_U, HCO3i_U, CO2i_U = pair2core(
#             PAR1_U, PAR2_U, PAR1TYPE, PAR2TYPE, convert_units=True, checks=False
#         )
#     # Generate vector describing the combination(s) of input parameters
#     Icase = getIcase(PAR1TYPE, PAR2TYPE)
#     # Do the rest
#     uparsargs = (
#         Icase,
#         Sal,
#         TempCi,
#         TempCo,
#         Pdbari,
#         Pdbaro,
#         TSi,
#         TP,
#         TNH3,
#         TH2S,
#         pHScale,
#         WhichKs,
#         WhoseKSO4,
#         WhoseKF,
#         WhoseTB,
#         buffers_mode,
#         ntps,
#         args,
#     )
#     outputdict, gradables = _co2sys_u_pars(
#         TA, TC, PHi, PCi, FCi, CARBi, HCO3i, CO2i, *uparsargs
#     )
#     if uncertainty:
#
#         # Define input variables to get derivatives with respect to
#         pars = [TA, TC, PHi, PCi, FCi, CARBi, HCO3i, CO2i]
#         pars_names = [
#             "TAlk",
#             "TCO2",
#             "pHin",
#             "pCO2in",
#             "fCO2in",
#             "CO3in",
#             "HCO3in",
#             "CO2in",
#         ]
#         # Work out which outputs we want to calculate derivatives for based on user input
#         getgrads = ["pHout", "CO2in"]
#         allgrads = {}
#         if type(getgrads) == str:
#             if getgrads == "all":
#                 getgrads = gradables
#             elif getgrads == "none":
#                 getgrads = []
#             else:
#                 getgrads = [getgrads]
#         # Loop through all outputs and calculate derivatives w.r.t. all inputs
#         for gradable in gradables:
#             if gradable in getgrads:
#                 print("Getting derivatives of {}...".format(gradable))
#                 par_grads = egrad(
#                     lambda pars: _co2sys_u_pars(*pars, *uparsargs)[0][gradable]
#                 )(pars)
#                 allgrads[gradable] = {
#                     pars_names[i]: par_grad for i, par_grad in enumerate(par_grads)
#                 }
#                 # Reassemble uncertainties in terms of original `PAR1` and `PAR2`
#                 allgrads[gradable]["PAR1"] = full(ntps, nan)
#                 for i, par in enumerate( pars_names):
#                     allgrads[gradable]["PAR1"] = where(PAR1TYPE == 1,
#                                                        allgrads[gradable][par],
#                                                        allgrads[gradable]["PAR1"])
#                 allgrads[gradable]["PAR2"] = full(ntps, nan)
#
#
#     return outputdict, allgrads
#
#
# def _co2sys_u_pars(
#     TA,
#     TC,
#     PHi,
#     PCi,
#     FCi,
#     CARBi,
#     HCO3i,
#     CO2i,
#     Icase,
#     Sal,
#     TempCi,
#     TempCo,
#     Pdbari,
#     Pdbaro,
#     TSi,
#     TP,
#     TNH3,
#     TH2S,
#     pHScale,
#     WhichKs,
#     WhoseKSO4,
#     WhoseKF,
#     WhoseTB,
#     buffers_mode,
#     ntps,
#     args,
# ):
#     # Prepare to solve the core marine carbonate system at input conditions
#     totals = salts.assemble(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB)
#     Sal = totals["Sal"]
#     Kis = equilibria.assemble(
#         TempCi, Pdbari, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
#     )
#     # Calculate fugacity factor
#     FugFaci = gas.fugacityfactor(TempCi, WhichKs)
#     # Solve the core marine carbonate system at input conditions
#     TA, TC, PHi, PCi, FCi, CARBi, HCO3i, CO2i = solve.core(
#         Icase, TA, TC, PHi, PCi, FCi, CARBi, HCO3i, CO2i, FugFaci, Kis, totals
#     )
#     core_in = {
#         "TA": TA,
#         "TC": TC,
#         "PH": PHi,
#         "PC": PCi,
#         "FC": FCi,
#         "CARB": CARBi,
#         "HCO3": HCO3i,
#         "CO2": CO2i,
#     }
#     # Calculate all other results at input conditions
#     others_in = solve.others(
#         core_in, Sal, TempCi, Pdbari, Kis, totals, pHScale, WhichKs, buffers_mode,
#     )
#     # Prepare to solve the core MCS at output conditions - get equilibrium constants
#     Kos = equilibria.assemble(
#         TempCo, Pdbaro, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
#     )
#     # Calculate fugacity factor
#     FugFaco = gas.fugacityfactor(TempCo, WhichKs)
#     # Expand inputs `par1` and `par2` into one array per core MCS variable
#     TAtype = full(ntps, 1)
#     TCtype = full(ntps, 2)
#     _, _, PHo, PCo, FCo, CARBo, HCO3o, CO2o = pair2core(
#         TA, TC, TAtype, TCtype, convert_units=False, checks=True
#     )
#     # Solve the core MCS at output conditions
#     Icase_out = getIcase(TAtype, TCtype, checks=False)
#     TA, TC, PHo, PCo, FCo, CARBo, HCO3o, CO2o = solve.core(
#         Icase_out, TA, TC, PHo, PCo, FCo, CARBo, HCO3o, CO2o, FugFaco, Kos, totals
#     )
#     core_out = {
#         "TA": TA,
#         "TC": TC,
#         "PH": PHo,
#         "PC": PCo,
#         "FC": FCo,
#         "CARB": CARBo,
#         "HCO3": HCO3o,
#         "CO2": CO2o,
#     }
#     # Calculate all other results at output conditions
#     others_out = solve.others(
#         core_out, Sal, TempCo, Pdbaro, Kos, totals, pHScale, WhichKs, buffers_mode,
#     )
#     # Save data directly as a dict to avoid ordering issues
#     outputdict, gradable = _outputdict(
#         args, core_in, core_out, others_in, others_out, Kis, Kos, totals, buffers_mode
#     )
#     return outputdict, gradable
