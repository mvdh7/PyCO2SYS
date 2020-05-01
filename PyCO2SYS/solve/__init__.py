# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Solve the marine carbonate system from any two of its variables."""

from . import delta, initialise, get
from autograd.numpy import full, isin, log10, nan, size, where
from .. import buffers, convert, gas, solubility

__all__ = ["delta", "initialise", "get"]


def core(Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFac, Ks, totals):
    """Fill part-empty core marine carbonate system variable columns with solutions."""
    # For convenience
    K0 = Ks["K0"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    PengCx = totals["PengCorrection"]
    # Convert any pCO2 and CO2(aq) values into fCO2
    PCgiven = isin(Icase, [14, 24, 34, 46, 47])
    FC = where(PCgiven, PC * FugFac, FC)
    CO2given = isin(Icase, [18, 28, 38, 68, 78])
    FC = where(CO2given, CO2 / K0, FC)
    # Solve the marine carbonate system
    F = Icase == 12  # input TA, TC
    if any(F):
        PH = where(F, get.pHfromTATC(TA - PengCx, TC, Ks, totals), PH)
        # ^pH is returned on the same scale as `Ks`
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 13  # input TA, pH
    if any(F):
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, Ks, totals), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = (Icase == 14) | (Icase == 15) | (Icase == 18)  # input TA, [pCO2|fCO2|CO2aq]
    if any(F):
        PH = where(F, get.pHfromTAfCO2(TA - PengCx, FC, Ks, totals), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, Ks, totals), TC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 16  # input TA, CARB
    if any(F):
        PH = where(F, get.pHfromTACarb(TA - PengCx, CARB, Ks, totals), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, Ks, totals), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 17  # input TA, HCO3
    if any(F):
        PH = where(F, get.pHfromTAHCO3(TA - PengCx, HCO3, Ks, totals), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, Ks, totals), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = Icase == 23  # input TC, pH
    if any(F):
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = (Icase == 24) | (Icase == 25) | (Icase == 28)  # input TC, [pCO2|fCO2|CO2aq]
    if any(F):
        PH = where(F, get.pHfromTCfCO2(TC, FC, K0, K1, K2), PH)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 26  # input TC, CARB
    if any(F):
        PH = where(F, get.pHfromTCCarb(TC, CARB, K1, K2), PH)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 27  # input TC, HCO3
    if any(F):
        PH = where(F, get.pHfromTCHCO3(TC, HCO3, K1, K2), PH)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = (Icase == 34) | (Icase == 35) | (Icase == 38)  # input pH, [pCO2|fCO2|CO2aq]
    if any(F):
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 36  # input pH, CARB
    if any(F):
        FC = where(F, get.fCO2frompHCarb(PH, CARB, K0, K1, K2), FC)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 37  # input pH, HCO3
    if any(F):
        TC = where(F, get.TCfrompHHCO3(PH, HCO3, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = (Icase == 46) | (Icase == 56) | (Icase == 68)  # input [pCO2|fCO2|CO2aq], CARB
    if any(F):
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 67  # input CO3, HCO3
    if any(F):
        FC = where(F, get.fCO2fromCarbHCO3(CARB, HCO3, K0, K1, K2), FC)
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
    F = (Icase == 47) | (Icase == 57) | (Icase == 78)  # input [pCO2|fCO2|CO2aq], HCO3
    if any(F):
        CARB = where(F, get.CarbfromfCO2HCO3(FC, HCO3, K0, K1, K2), CARB)
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, Ks, totals) + PengCx, TA)
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 and CO2(aq) values:
    PC = where(~PCgiven, FC / FugFac, PC)
    CO2 = where(~CO2given, FC * K0, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2


def others(
    core_solved, Sal, TempC, Pdbar, Ks, totals, pHScale, WhichKs, buffers_mode,
):
    """Calculate all peripheral marine carbonate system variables returned by CO2SYS."""
    # Unpack for convenience
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
    alks = get.AlkParts(TC, PH, FREEtoTOT, Ks, totals)
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
            WhichKs,
            Ks,
            totals,
        )
        for buffer in esm10buffers:
            allbuffers_ESM10[buffer] = where(
                F, auto_ESM10[buffer], allbuffers_ESM10[buffer]
            )
        isoQ = where(F, buffers.isocap(TAPeng, TC, PH, FC, Ks, totals), isoQ)
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
            F, buffers.explicit.RevelleFactor(TAPeng, TC, Ks, totals), Revelle
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
    }
