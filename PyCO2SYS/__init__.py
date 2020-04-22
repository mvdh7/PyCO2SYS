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
    # Condition inputs and assign input to the 'historical' variable names.
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
    # Solve the core marine carbonate system at input conditions
    Sal, TCa, PengCorrection, totals = salts.assemble(
        Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB
    )
    K0i, FugFaci, fHi, Kis = equilibria.assemble(
        TempCi, Pdbari, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    TAc, TCc, PHic, PCic, FCic, CARBic, HCO3ic, CO2ic = engine.solvecore(
        PAR1, PAR2, p1, p2, PengCorrection, totals, K0i, FugFaci, Kis
    )
    # Calculate all other results at input conditions
    (
        pK1i,
        pK2i,
        BAlkinp,
        OHinp,
        PAlkinp,
        SiAlkinp,
        NH3Alkinp,
        H2SAlkinp,
        Hfreeinp,
        HSO4inp,
        HFinp,
        OmegaCainp,
        OmegaArinp,
        VPFaci,
        xCO2dryinp,
        pHicT,
        pHicS,
        pHicF,
        pHicN,
        Revelleinp,
        gammaTCi,
        betaTCi,
        omegaTCi,
        gammaTAi,
        betaTAi,
        omegaTAi,
        isoQi,
        isoQxi,
        psii,
    ) = solve.others(
        TAc,
        TCc,
        PHic,
        PCic,
        CARBic,
        HCO3ic,
        CO2ic,
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
    # Solve the core MCS at output conditions
    K0o, FugFaco, fHo, Kos = equilibria.assemble(
        TempCo, Pdbaro, Sal, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF
    )
    TAtype = full(ntps, 1)
    TCtype = full(ntps, 2)
    _, _, PHoc, PCoc, FCoc, CARBoc, HCO3oc, CO2oc = engine.solvecore(
        TAc, TCc, TAtype, TCtype, PengCorrection, totals, K0o, FugFaco, Kos
    )
    # Calculate all other results at output conditions
    (
        pK1o,
        pK2o,
        BAlkout,
        OHout,
        PAlkout,
        SiAlkout,
        NH3Alkout,
        H2SAlkout,
        Hfreeout,
        HSO4out,
        HFout,
        OmegaCaout,
        OmegaArout,
        VPFaco,
        xCO2dryout,
        pHocT,
        pHocS,
        pHocF,
        pHocN,
        Revelleout,
        gammaTCo,
        betaTCo,
        omegaTCo,
        gammaTAo,
        betaTAo,
        omegaTAo,
        isoQo,
        isoQxo,
        psio,
    ) = solve.others(
        TAc,
        TCc,
        PHoc,
        PCoc,
        CARBoc,
        HCO3oc,
        CO2oc,
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
        "TAlk": TAc * 1e6,
        "TCO2": TCc * 1e6,
        "pHin": PHic,
        "pCO2in": PCic * 1e6,
        "fCO2in": FCic * 1e6,
        "HCO3in": HCO3ic * 1e6,
        "CO3in": CARBic * 1e6,
        "CO2in": CO2ic * 1e6,
        "BAlkin": BAlkinp * 1e6,
        "OHin": OHinp * 1e6,
        "PAlkin": PAlkinp * 1e6,
        "SiAlkin": SiAlkinp * 1e6,
        "NH3Alkin": NH3Alkinp * 1e6,
        "H2SAlkin": H2SAlkinp * 1e6,
        "Hfreein": Hfreeinp * 1e6,
        "RFin": Revelleinp,
        "OmegaCAin": OmegaCainp,
        "OmegaARin": OmegaArinp,
        "xCO2in": xCO2dryinp * 1e6,
        "pHout": PHoc,
        "pCO2out": PCoc * 1e6,
        "fCO2out": FCoc * 1e6,
        "HCO3out": HCO3oc * 1e6,
        "CO3out": CARBoc * 1e6,
        "CO2out": CO2oc * 1e6,
        "BAlkout": BAlkout * 1e6,
        "OHout": OHout * 1e6,
        "PAlkout": PAlkout * 1e6,
        "SiAlkout": SiAlkout * 1e6,
        "NH3Alkout": NH3Alkout * 1e6,
        "H2SAlkout": H2SAlkout * 1e6,
        "Hfreeout": Hfreeout * 1e6,
        "RFout": Revelleout,
        "OmegaCAout": OmegaCaout,
        "OmegaARout": OmegaArout,
        "xCO2out": xCO2dryout * 1e6,
        "pHinTOTAL": pHicT,
        "pHinSWS": pHicS,
        "pHinFREE": pHicF,
        "pHinNBS": pHicN,
        "pHoutTOTAL": pHocT,
        "pHoutSWS": pHocS,
        "pHoutFREE": pHocF,
        "pHoutNBS": pHocN,
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
        "pK1input": pK1i,
        "pK2input": pK2i,
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
        "pK1output": pK1o,
        "pK2output": pK2o,
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
        "gammaTCin": gammaTCi,
        "betaTCin": betaTCi,
        "omegaTCin": omegaTCi,
        "gammaTAin": gammaTAi,
        "betaTAin": betaTAi,
        "omegaTAin": omegaTAi,
        "gammaTCout": gammaTCo,
        "betaTCout": betaTCo,
        "omegaTCout": omegaTCo,
        "gammaTAout": gammaTAo,
        "betaTAout": betaTAo,
        "omegaTAout": omegaTAo,
        "isoQin": isoQi,
        "isoQout": isoQo,
        "isoQapprox_in": isoQxi,
        "isoQapprox_out": isoQxo,
        "psi_in": psii,
        "psi_out": psio,
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
