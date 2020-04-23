# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Solve the marine carbonate system and calculate related seawater properties."""
from . import (
    buffers,
    constants,
    convert,
    engine,
    equilibria,
    gas,
    meta,
    original,
    salts,
    solubility,
    solve,
    test,
)

__all__ = [
    "assemble",
    "buffers",
    "constants",
    "convert",
    "engine",
    "equilibria",
    "gas",
    "meta",
    "original",
    "salts",
    "solubility",
    "solve",
    "test",
]

__author__ = meta.authors
__version__ = meta.version

from autograd.numpy import array, full, shape


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
    KSO4CONSTANTS=0,
):
    # Condition inputs and assign input values to the 'historical' variable names
    args, ntps = engine.inputs(locals())
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
    # Prepare to solve the core marine carbonate system at input conditions
    Sal, TCa, PengCorrection, totals = salts.assemble(
        Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB
    )
    K0i, FugFaci, fHi, Kis = equilibria.assemble(
        TempCi, Pdbari, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    # Solve the core marine carbonate system at input conditions
    core_in = engine.solvecore(
        PAR1, PAR2, p1, p2, PengCorrection, totals, K0i, FugFaci, Kis
    )
    # Calculate all other results at input conditions
    others_in = solve.others(
        core_in,
        Sal,
        TempCi,
        Pdbari,
        K0i,
        Kis,
        fHi,
        totals,
        PengCorrection,
        TCa,
        pHScale,
        WhichKs,
    )
    # Prepare to solve the core MCS at output conditions
    K0o, FugFaco, fHo, Kos = equilibria.assemble(
        TempCo, Pdbaro, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    TAtype = full(ntps, 1)
    TCtype = full(ntps, 2)
    # Solve the core MCS at output conditions
    core_out = engine.solvecore(
        core_in["TA"],
        core_in["TC"],
        TAtype,
        TCtype,
        PengCorrection,
        totals,
        K0o,
        FugFaco,
        Kos,
    )
    # Calculate all other results at output conditions
    others_out = solve.others(
        core_out,
        Sal,
        TempCo,
        Pdbaro,
        K0o,
        Kos,
        fHo,
        totals,
        PengCorrection,
        TCa,
        pHScale,
        WhichKs,
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
        "K0input": K0i,
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
        "K0output": K0o,
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
        "TCa": TCa * 1e6,
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
        KSO4CONSTANTS=KSO4CONSTANTS,
    )
