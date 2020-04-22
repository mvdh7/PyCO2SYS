# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Solve the marine carbonate system from any two of its variables."""
from . import initialise, get
from autograd.numpy import array, full, isin, log10, nan, size, where
from autograd.numpy import all as np_all
from autograd.numpy import any as np_any
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max
from .. import buffers, convert, gas, solubility

__all__ = ["initialise", "get"]


def pair2core(par1, par2, par1type, par2type):
    """Expand `par1` and `par2` inputs into one array per core variable of the marine 
    carbonate system.
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
    # Assign values to empty vectors and convert micro[mol|atm] to [mol|atm]
    TA = where(par1type == 1, par1 * 1e-6, TA)
    TC = where(par1type == 2, par1 * 1e-6, TC)
    PH = where(par1type == 3, par1, PH)
    PC = where(par1type == 4, par1 * 1e-6, PC)
    FC = where(par1type == 5, par1 * 1e-6, FC)
    CARB = where(par1type == 6, par1 * 1e-6, CARB)
    HCO3 = where(par1type == 7, par1 * 1e-6, HCO3)
    CO2 = where(par1type == 8, par1 * 1e-6, CO2)
    TA = where(par2type == 1, par2 * 1e-6, TA)
    TC = where(par2type == 2, par2 * 1e-6, TC)
    PH = where(par2type == 3, par2, PH)
    PC = where(par2type == 4, par2 * 1e-6, PC)
    FC = where(par2type == 5, par2 * 1e-6, FC)
    CARB = where(par2type == 6, par2 * 1e-6, CARB)
    HCO3 = where(par2type == 7, par2 * 1e-6, HCO3)
    CO2 = where(par2type == 8, par2 * 1e-6, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2


def getIcase(par1type, par2type):
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
    """
    # Check validity of separate `par1type` and `par2type` inputs
    Iarr = array([par1type, par2type])
    assert np_all(
        isin(Iarr, [1, 2, 3, 4, 5, 6, 7, 8])
    ), "All `par1type` and `par2type` values must be integers from 1 to 8."
    assert ~np_any(
        par1type == par2type
    ), "`par1type` and `par2type` must be different from each other."
    # Combine inputs into `Icase` and check its validity
    Icase = 10 * np_min(Iarr, axis=0) + np_max(Iarr, axis=0)
    assert ~np_any(
        isin(Icase, [45, 48, 58])
    ), "Combinations of pCO2, fCO2 and CO2(aq) are not valid input pairs."
    return Icase


def _fill(Icase, K0, TA, TC, PH, PC, FC, CARB, HCO3, CO2, PengCx, FugFac, Ks, totals):
    """Fill part-empty core marine carbonate system variable columns."""
    # Convert any pCO2 and CO2(aq) values into fCO2
    PCgiven = isin(Icase, [14, 24, 34, 46, 47])
    FC = where(PCgiven, PC * FugFac, FC)
    CO2given = isin(Icase, [18, 28, 38, 68, 78])
    FC = where(CO2given, CO2 * FugFac / K0, FC)
    # For convenience
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    # Solve the marine carbonate system
    F = Icase == 12  # input TA, TC
    if any(F):
        PH = where(F, get.pHfromTATC(TA - PengCx, TC, **Ks, **totals), PH)
        # ^pH is returned on the scale requested in `pHscale`
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 13  # input TA, pH
    if any(F):
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, **Ks, **totals), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = (Icase == 14) | (Icase == 15) | (Icase == 18)  # input TA, [pCO2|fCO2|CO2aq]
    if any(F):
        PH = where(F, get.pHfromTAfCO2(TA - PengCx, FC, K0, **Ks, **totals), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, **Ks, **totals), TC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 16  # input TA, CARB
    if any(F):
        PH = where(F, get.pHfromTACarb(TA - PengCx, CARB, **Ks, **totals), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, **Ks, **totals), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 17  # input TA, HCO3
    if any(F):
        PH = where(F, get.pHfromTAHCO3(TA - PengCx, HCO3, **Ks, **totals), PH)
        TC = where(F, get.TCfromTApH(TA - PengCx, PH, **Ks, **totals), TC)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = Icase == 23  # input TC, pH
    if any(F):
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = (Icase == 24) | (Icase == 25) | (Icase == 28)  # input TC, [pCO2|fCO2|CO2aq]
    if any(F):
        PH = where(F, get.pHfromTCfCO2(TC, FC, K0, K1, K2), PH)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 26  # input TC, CARB
    if any(F):
        PH = where(F, get.pHfromTCCarb(TC, CARB, K1, K2), PH)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 27  # input TC, HCO3
    if any(F):
        PH = where(F, get.pHfromTCHCO3(TC, HCO3, K1, K2), PH)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = (Icase == 34) | (Icase == 35) | (Icase == 38)  # input pH, [pCO2|fCO2|CO2aq]
    if any(F):
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 36  # input pH, CARB
    if any(F):
        FC = where(F, get.fCO2frompHCarb(PH, CARB, K0, K1, K2), FC)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 37  # input pH, HCO3
    if any(F):
        TC = where(F, get.TCfrompHHCO3(PH, HCO3, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        FC = where(F, get.fCO2fromTCpH(TC, PH, K0, K1, K2), FC)
        CARB = where(F, get.CarbfromTCpH(TC, PH, K1, K2), CARB)
    F = (Icase == 46) | (Icase == 56) | (Icase == 68)  # input [pCO2|fCO2|CO2aq], CARB
    if any(F):
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        HCO3 = where(F, get.HCO3fromTCpH(TC, PH, K1, K2), HCO3)
    F = Icase == 67  # input CO3, HCO3
    if any(F):
        FC = where(F, get.fCO2fromCarbHCO3(CARB, HCO3, K0, K1, K2), FC)
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
    F = (Icase == 47) | (Icase == 57) | (Icase == 78)  # input [pCO2|fCO2|CO2aq], HCO3
    if any(F):
        CARB = where(F, get.CarbfromfCO2HCO3(FC, HCO3, K0, K1, K2), CARB)
        PH = where(F, get.pHfromfCO2Carb(FC, CARB, K0, K1, K2), PH)
        TC = where(F, get.TCfrompHfCO2(PH, FC, K0, K1, K2), TC)
        TA = where(F, get.TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 and CO2(aq) values:
    PC = where(~PCgiven, FC / FugFac, PC)
    CO2 = where(~CO2given, FC * K0 / FugFac, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2


def core(par1, par2, par1type, par2type, PengCx, totals, K0, FugFac, Ks):
    """Solve the core marine carbonate system (MCS) from any 2 of its variables.
    
    The core MCS outputs and associated `par1type`/`par2type` inputs are:
        
      * Type `1`, `TA`: total alkalinity in mol/kg-sw.
      * Type `2`, `TC`: dissolved inorganic carbon in mol/kg-sw.
      * Type `3`, `PH`: pH on whichever scale(s) the constants in `Ks` are provided.
      * Type `4`, `PC`: partial pressure of CO2 in atm.
      * Type `5`, `FC`: fugacity of CO2 in atm.
      * Type `6`, `CARB`: carbonate ion in mol/kg-sw.
      * Type `7`, `HCO3`: bicarbonate ion in mol/kg-sw.
      * Type `8`, `CO2`: aqueous CO2 in mol/kg-sw.
    """
    # Expand inputs `par1` and `par2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pair2core(par1, par2, par1type, par2type)
    # Generate vector describing the combination(s) of input parameters
    Icase = getIcase(par1type, par2type)
    # Solve the core marine carbonate system
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = _fill(
        Icase, K0, TA, TC, PH, PC, FC, CARB, HCO3, CO2, PengCx, FugFac, Ks, totals
    )
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2


def allothers(
    TA,
    TC,
    PH,
    PC,
    CARB,
    HCO3,
    CO2,
    Sal,
    TempC,
    Pdbar,
    K0,
    Ks,
    fH,
    totals,
    PengCx,
    TCa,
    pHScale,
    WhichKs,
):
    # pKs
    pK1 = -log10(Ks["K1"])
    pK2 = -log10(Ks["K2"])
    # Components of alkalinity and DIC
    FREEtoTOT = convert.free2tot(totals["TSO4"], Ks["KSO4"])
    _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = get.AlkParts(
        PH, TC, FREEtoTOT, **Ks, **totals
    )
    PAlk = PAlk + PengCx
    # CaCO3 solubility
    OmegaCa, OmegaAr = solubility.CaCO3(
        Sal, TempC, Pdbar, CARB, TCa, WhichKs, Ks["K1"], Ks["K2"]
    )
    # Dry mole fraction of CO2
    VPFac = gas.vpfactor(TempC, Sal)
    xCO2dry = PC / VPFac  # this assumes pTot = 1 atm
    # Just for reference, convert pH at input conditions to the other scales
    pHT, pHS, pHF, pHN = convert.pH2allscales(
        PH, pHScale, Ks["KSO4"], Ks["KF"], totals["TSO4"], totals["TF"], fH
    )
    # Buffers by explicit calculation
    Revelle = buffers.RevelleFactor(TA - PengCx, TC, K0, Ks, totals)
    # Evaluate ESM10 buffer factors (corrected following RAH18) [added v1.2.0]
    gammaTC, betaTC, omegaTC, gammaTA, betaTA, omegaTA = buffers.buffers_ESM10(
        TC, TA, CO2, HCO3, CARB, PH, OH, BAlk, Ks["KB"]
    )
    # Evaluate (approximate) isocapnic quotient [HDW18] and psi [FCG94] [added v1.2.0]
    isoQ = buffers.bgc_isocap(
        CO2, PH, Ks["K1"], Ks["K2"], Ks["KB"], Ks["KW"], totals["TB"]
    )
    isoQx = buffers.bgc_isocap_approx(TC, PC, K0, Ks["K1"], Ks["K2"])
    psi = buffers.psi(CO2, PH, Ks["K1"], Ks["K2"], Ks["KB"], Ks["KW"], totals["TB"])
    return (
        pK1,
        pK2,
        BAlk,
        OH,
        PAlk,
        SiAlk,
        NH3Alk,
        H2SAlk,
        Hfree,
        HSO4,
        HF,
        OmegaCa,
        OmegaAr,
        VPFac,
        xCO2dry,
        pHT,
        pHS,
        pHF,
        pHN,
        Revelle,
        gammaTC,
        betaTC,
        omegaTC,
        gammaTA,
        betaTA,
        omegaTA,
        isoQ,
        isoQx,
        psi,
    )
