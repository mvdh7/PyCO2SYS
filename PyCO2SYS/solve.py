# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Solve the marine carbonate system from any two of its variables."""

from autograd.numpy import array, full, isin, log, log10, nan, size, sqrt, where
from autograd.numpy import abs as np_abs
from autograd.numpy import all as np_all
from autograd.numpy import any as np_any
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max
from . import assemble, buffers, convert, gas, solubility

def _goodH0_CO2(CBAlk, CO2, TB, K1, K2, KB):
    """Find initial value for pH solvers with TC as the second variable
    inspired by M13 section 3.2.2 for fCO2 as the second variable,
    assuming that CBAlk is within a suitable range.
    """
    c2 = KB - (TB*KB + K1*CO2)/CBAlk
    c1 = -K1*(2*K2*CO2 + KB*CO2)/CBAlk
    c0 = -2*K1*K2*KB*CO2/CBAlk
    c21min = c2**2 - 3*c1
    c21min_positive = c21min > 0
    sq21 = where(c21min_positive, sqrt(c21min), 0.0)
    Hmin = where(c2 < 0, -c2 + sq21/3, -c1/(c2 + sq21))
    H0 = where(c21min_positive, # i.e. sqrt(c21min) is real
                Hmin + sqrt(-(c2*Hmin**2 + c1*Hmin + c0)/sq21),
                1e-7) # default pH=7 if 2nd order approx has no solution
    return H0

def _guesspH_CO2(CBAlk, CO2, TB, K1, K2, KB):
    """Find initial value for pH solvers with fCO2 as the second variable
    inspired by M13 section 3.2.2 for fCO2 as the second variable.
    """
    H0 = where(CBAlk > 0,
               _goodH0_CO2(CBAlk, CO2, TB, K1, K2, KB),
               1e-3) # default pH=3 for negative alkalinity
    return -log10(H0)

def _goodH0_TC(CBAlk, TC, TB, K1, K2, KB):
    """Find initial value for pH solvers with TC as the second variable
    following M13 section 3.2.2 for TC as the second variable,
    assuming that CBAlk is within a suitable range.
    """
    c2 = KB*(1 - TB/CBAlk) + K1*(1 - TC/CBAlk)
    c1 = K1*(KB*(1 - TB/CBAlk - TC/CBAlk) + K2*(1 - 2*TC/CBAlk))
    c0 = K1*K2*KB*(1 - (2*TC + TB)/CBAlk)
    c21min = c2**2 - 3*c1
    c21min_positive = c21min > 0
    sq21 = where(c21min_positive, sqrt(c21min), 0.0)
    Hmin = where(c2 < 0, -c2 + sq21/3, -c1/(c2 + sq21))
    H0 = where(c21min_positive, # i.e. sqrt(c21min) is real
               Hmin + sqrt(-(c2*Hmin**2 + c1*Hmin + c0)/sq21),
               1e-7) # default pH=7 if 2nd order approx has no solution
    return H0

def _guesspH_TC(CBAlk, TC, TB, K1, K2, KB):
    """Find initial value for pH solvers with TC as the second variable
    following M13's 3.2.2 and its implementation in mocsy/phsolvers.f90 (OE13).
    """
    # Logical conditions and defaults from mocsy phsolvers.f90
    H0 = where(CBAlk <= 0,
               1e-3, # default pH=3 for negative alkalinity
               1e-10) # default pH=10 for very high alkalinity relative to DIC
    F = (CBAlk > 0) & (CBAlk < 2*TC + TB)
    if any(F): # use better estimate if alkalinity in suitable range
        H0 = where(F, _goodH0_TC(CBAlk, TC, TB, K1, K2, KB), H0)
    return -log10(H0)

pHTol = 1e-6 # tolerance for ending iterations in all pH solvers

def CarbfromTCpH(TC, pH, K1, K2):
    """Calculate carbonate ion from dissolved inorganic carbon and pH.

    Based on CalculateCarbfromTCpH, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    H = 10.0**-pH
    CARB = TC*K1*K2/(H**2 + K1*H + K1*K2)
    return CARB

def AlkParts(pH, TC, FREEtoTOT,
        K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate the different components of total alkalinity from pH and
    dissolved inorganic carbon.

    Although coded for H on the Total pH scale, for the pH values occuring in
    seawater (pH > 6) this will be equally valid on any pH scale (i.e. H terms
    are negligible) as long as the K Constants are on that scale.

    Based on CalculateAlkParts, version 01.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0**-pH
    HCO3 = TC*K1*H/(K1*H + H**2 + K1*K2)
    CO3 = CarbfromTCpH(TC, pH, K1, K2)
    BAlk = TB*KB/(KB + H)
    OH = KW/H
    PAlk = (TPO4*(KP1*KP2*H + 2*KP1*KP2*KP3 - H**3)/
            (H**3 + KP1*H**2 + KP1*KP2*H + KP1*KP2*KP3))
    SiAlk = TSi*KSi/(KSi + H)
    NH3Alk = TNH3*KNH3/(KNH3 + H)
    H2SAlk = TH2S*KH2S/(KH2S + H)
    Hfree = H/FREEtoTOT # for H on the Total scale
    HSO4 = TSO4/(1 + KSO4/Hfree) # since KSO4 is on the Free scale
    HF = TF/(1 + KF/Hfree) # since KF is on the Free scale
    return HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF

def pHfromTATC(TA, TC,
        K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4,  TSi, TNH3, TH2S):
    """Calculate pH from total alkalinity and dissolved inorganic carbon.

    This calculates pH from TA and TC using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess uses the carbonate-borate alkalinity estimate of M13 and
    OE13 as implemented in mocsy.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculatepHfromTATC, version 04.01, 10-13-96, by Ernie Lewis.
    SVH2007: Made this to accept vectors. It will continue iterating until all
    values in the vector are "abs(deltapH) < pHTol".
    """
    pH = _guesspH_TC(TA, TC, TB, K1, K2, KB) # following M13/OE13, added v1.3.0
    deltapH = 1.0 + pHTol
    ln10 = log(10)
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    while np_any(np_abs(deltapH) > pHTol):
        HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            AlkParts(pH, TC, FREEtoTOT,
                K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TSO4, TPO4, TSi, TNH3, TH2S)
        CAlk = HCO3 + 2*CO3
        H = 10.0**-pH
        Denom = H**2 + K1*H + K1*K2
        Residual = (TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                    Hfree + HSO4 + HF)
        # Find slope dTA/dpH
        # Calculation of phosphate component of slope makes virtually no
        # difference to end result and makes code run much slower, so not used.
        # PDenom = (KP1*KP2*KP3 + KP1*KP2*H + KP1*H**2 + H**3)
        # PSlope = -TPO4*H*((KP1*KP2 - 2*H)/PDenom -
        #     (KP1*KP2 + 2*KP1*H + 3*H**2)*(2*KP1*KP2*KP3 + KP1*KP2*H - H**2)/
        #     PDenom**2)
        # Adding other nutrients doesn't really impact speed but makes so little
        # difference that they've been excluded for consistency with previous
        # versions of CO2SYS.
        Slope = ln10*(TC*K1*H*(H**2 + K1*K2 + 4*H*K2)/Denom**2 +
            BAlk*H/(KB + H) + OH + H) # terms after here would add nutrients
            # SiAlk*H/(KSi + H) + NH3Alk*H/(KNH3 + H) + H2SAlk*H/(KH2S + H))
            # + PSlope) # to add phosphate component
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        deltapH = where(np_abs(deltapH) > 1, deltapH/2, deltapH)
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        pH = where(np_abs(deltapH) > pHTol, pH+deltapH, pH)
        # ^pH is on the same scale as K1 and K2 were calculated.
    return pH

def pHfromTAfCO2(TA, fCO2, K0,
        K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate pH from total alkalinity and CO2 fugacity.

    This calculates pH from TA and fCO2 using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculatepHfromTAfCO2, version 04.01, 10-13-97, by Ernie Lewis.
    """
    pHGuess = 8.0 # this is the first guess
    pH = full(size(TA), pHGuess) # first guess for all samples
    deltapH = 1 + pHTol
    ln10 = log(10)
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    while np_any(np_abs(deltapH) > pHTol):
        H = 10.0**-pH
        HCO3 = K0*K1*fCO2/H
        CO3 = K0*K1*K2*fCO2/H**2
        CAlk = HCO3 + 2*CO3
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            AlkParts(pH, 0.0, FREEtoTOT,
                K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TSO4, TPO4, TSi, TNH3, TH2S)
        Residual = (TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                    Hfree + HSO4 + HF)
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10*(HCO3 + 4*CO3 + BAlk*H/(KB + H) + OH + H)
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        deltapH = where(np_abs(deltapH) > 1, deltapH/2, deltapH)
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        pH = where(np_abs(deltapH) > pHTol, pH+deltapH, pH)
    return pH

def pHfromTACarb(TA, CARB,
        K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate pH from total alkalinity and carbonate ion.

    This calculates pH from TA and Carb using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K constants are on that scale.

    Based on CalculatepHfromTACarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    pHGuess = 8.0 # this is the first guess
    pH = full(size(TA), pHGuess) # first guess for all samples
    deltapH = 1 + pHTol
    ln10 = log(10)
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    while np_any(np_abs(deltapH) > pHTol):
        H = 10.0**-pH
        CAlk = CARB*(H + 2*K2)/K2
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            AlkParts(pH, 0.0, FREEtoTOT,
                K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TSO4, TPO4, TSi, TNH3, TH2S)
        Residual = (TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk -
                    H2SAlk + Hfree + HSO4 + HF)
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10*(-CARB*H/K2 + BAlk*H/(KB + H) + OH + H)
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        deltapH = where(np_abs(deltapH) > 1, deltapH/2, deltapH)
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        pH = where(np_abs(deltapH) > pHTol, pH+deltapH, pH)
    return pH

def fCO2fromTCpH(TC, pH, K0, K1, K2):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.

    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0**-pH
    fCO2 = TC*H**2/(H**2 + K1*H + K1*K2)/K0
    return fCO2

def TCfromTApH(TA, pH,
        K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.

    This calculates TC from TA and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0**-pH
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
        AlkParts(pH, 0.0, FREEtoTOT,
            K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
            TB, TF, TSO4, TPO4, TSi, TNH3, TH2S)
    CAlk = (TA - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF)
    TC = CAlk*(H**2 + K1*H + K1*K2)/(K1*(H + 2*K2))
    return TC

def TAfromTCpH(TC, pH,
        K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate total alkalinity from dissolved inorganic carbon and pH.

    This calculates TA from TC and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTAfromTCpH, version 02.02, 10-10-97, by Ernie Lewis.
    """
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
        AlkParts(pH, TC, FREEtoTOT,
            K1, K2, KW, KB, KF, KSO4, KP1, KP2, KP3, KSi, KNH3, KH2S,
            TB, TF, TSO4, TPO4, TSi, TNH3, TH2S)
    CAlk = HCO3 + 2*CO3
    TAc = (CAlk + BAlk + OH + PAlk + SiAlk + NH3Alk + H2SAlk - Hfree -
           HSO4 - HF)
    return TAc

def pHfromTCfCO2(TC, fCO2, K0, K1, K2):
    """Calculate pH from dissolved inorganic carbon and CO2 fugacity.

    This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the
    quadratic in H: fCO2*K0 = TC*H*H/(K1*H + H*H + K1*K2).
    If there is not a real root, then pH is returned as NaN.

    Based on CalculatepHfromTCfCO2, version 02.02, 11-12-96, by Ernie Lewis.
    """
    RR = K0*fCO2/TC
    Discr = (K1*RR)**2 + 4*(1 - RR)*K1*K2*RR
    H = 0.5*(K1*RR + sqrt(Discr))/(1 - RR)
    H[H < 0] = nan
    pH = -log10(H)
    return pH

def TCfrompHfCO2(pH, fCO2, K0, K1, K2):
    """Calculate dissolved inorganic carbon from pH and CO2 fugacity.

    Based on CalculateTCfrompHfCO2, version 01.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0**-pH
    TC = K0*fCO2*(H**2 + K1*H + K1*K2)/H**2
    return TC

def pHfromTCCarb(TC, CARB, K1, K2):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    This calculates pH from Carbonate and TC using K1, and K2 by solving the
    quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).

    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis
    Pierrot.
    """
    RR = 1 - TC/CARB
    Discr = K1**2 - 4*K1*K2*RR
    H = (-K1 + sqrt(Discr))/2
    pH = -log10(H)
    return pH

def fCO2frompHCarb(pH, CARB, K0, K1, K2):
    """Calculate CO2 fugacity from pH and carbonate ion.

    Based on CalculatefCO2frompHCarb, version 01.0, 06-12-2019, by Denis
    Pierrot.
    """
    H = 10.0**-pH
    fCO2 = CARB*H**2/(K0*K1*K2)
    return fCO2

def pHfromfCO2Carb(fCO2, CARB, K0, K1, K2):
    """Calculate pH from CO2 fugacity and carbonate ion.

    This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving
    the equation in H: fCO2 * K0 * K1* K2 = Carb * H * H

    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis
    Pierrot.
    """
    H = sqrt(K0*K1*K2*fCO2/CARB)
    pH = -log10(H)
    return pH

def pars2mcs(par1, par2, par1type, par2type):
    """Expand `par1` and `par2` inputs into one array per core variable of the
    marine carbonate system.
    """
    ntps = size(par1)
    # Generate empty vectors for...
    TA = full(ntps, nan) # Talk
    TC = full(ntps, nan) # DIC
    PH = full(ntps, nan) # pH
    PC = full(ntps, nan) # pCO2
    FC = full(ntps, nan) # fCO2
    CARB = full(ntps, nan) # CO3 ions
    # Assign values to empty vectors and convert micro[mol|atm] to [mol|atm]
    TA = where(par1type==1, par1*1e-6, TA)
    TC = where(par1type==2, par1*1e-6, TC)
    PH = where(par1type==3, par1, PH)
    PC = where(par1type==4, par1*1e-6, PC)
    FC = where(par1type==5, par1*1e-6, FC)
    CARB = where(par1type==6, par1*1e-6, CARB)
    TA = where(par2type==1, par2*1e-6, TA)
    TC = where(par2type==2, par2*1e-6, TC)
    PH = where(par2type==3, par2, PH)
    PC = where(par2type==4, par2*1e-6, PC)
    FC = where(par2type==5, par2*1e-6, FC)
    CARB = where(par2type==6, par2*1e-6, CARB)
    return TA, TC, PH, PC, FC, CARB

def getIcase(par1type, par2type):
    """Generate vector describing the combination of input parameters.

    Noting that the pCO2 and fCO2 pair is not allowed, the valid ones are:
        12, 13, 14, 15, 16,
            23, 24, 25, 26,
                34, 35, 36,
                        46,
                        56.
    """
    Iarr = array([par1type, par2type])
    assert np_all(isin(Iarr, [1, 2, 3, 4, 5, 6])), \
        'All `PAR1TYPE` and `PAR2TYPE` values must be integers from 1 to 6.'
    Icase = 10*np_min(Iarr, axis=0) + np_max(Iarr, axis=0)
    assert ~np_any(Icase == 45), 'pCO2 and fCO2 is not a valid input pair.'
    return Icase

def from2to6constants(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB):
    """Calculate constants (w.r.t. temperature and pressure) in preparation for
    `fill6` or `_from2to6` functions."""
    # Pure Water case:
    Sal = where(WhichKs==8, 0.0, Sal)
    # GEOSECS and Pure Water:
    F = (WhichKs==6) | (WhichKs==8)
    TP = where(F, 0.0, TP)
    TSi = where(F, 0.0, TSi)
    TNH3 = where(F, 0.0, TNH3)
    TH2S = where(F, 0.0, TH2S)
    # Convert micromol to mol
    TP = TP*1e-6
    TSi = TSi*1e-6
    TNH3 = TNH3*1e-6
    TH2S = TH2S*1e-6
    TCa, totals = assemble.concentrations(Sal, WhichKs, WhoseTB)
    # Add equilibrating user inputs except DIC to `totals` dict
    totals['TPO4'] = TP
    totals['TSi'] = TSi
    totals['TNH3'] = TNH3
    totals['TH2S'] = TH2S
    # The vector `PengCorrection` is used to modify the value of TA, for those
    # cases where WhichKs==7, since PAlk(Peng) = PAlk(Dickson) + TP.
    # Thus, PengCorrection is 0 for all cases where WhichKs is not 7.
    PengCorrection = where(WhichKs==7, totals['TPO4'], 0.0)
    return Sal, TCa, totals, PengCorrection

def fill6(Icase, K0, TA, TC, PH, PC, FC, CARB, PengCx, FugFac, Ks, totals):
    """Fill the 6 partly empty MCS variable columns with solutions."""
    # pCO2 will be calculated at the end, the functions here work with fCO2
    PCgiven = isin(Icase, [14, 24, 34, 46])
    FC = where(PCgiven, PC*FugFac, FC)
    F = Icase==12 # input TA, TC
    if any(F):
        PH = where(F, pHfromTATC(TA-PengCx, TC, **Ks, **totals), PH)
        # ^pH is returned on the scale requested in `pHscale`
        FC = where(F, fCO2fromTCpH(TC, PH, K0, Ks['K1'], Ks['K2']), FC)
        CARB = where(F, CarbfromTCpH(TC, PH, Ks['K1'], Ks['K2']), CARB)
    F = Icase==13 # input TA, pH
    if any(F):
        TC = where(F, TCfromTApH(TA-PengCx, PH, **Ks, **totals), TC)
        FC = where(F, fCO2fromTCpH(TC, PH, K0, Ks['K1'], Ks['K2']), FC)
        CARB = where(F, CarbfromTCpH(TC, PH, Ks['K1'], Ks['K2']), CARB)
    F = (Icase==14) | (Icase==15) # input TA, (pCO2 or fCO2)
    if any(F):
        PH = where(F, pHfromTAfCO2(TA-PengCx, FC, K0, **Ks, **totals), PH)
        TC = where(F, TCfromTApH(TA-PengCx, PH, **Ks, **totals), TC)
        CARB = where(F, CarbfromTCpH(TC, PH, Ks['K1'], Ks['K2']), CARB)
    F = Icase==16 # input TA, CARB
    if any(F):
        PH = where(F, pHfromTACarb(TA-PengCx, CARB, **Ks, **totals), PH)
        TC = where(F, TCfromTApH(TA-PengCx, PH, **Ks, **totals), TC)
        FC = where(F, fCO2fromTCpH(TC, PH, K0, Ks['K1'], Ks['K2']), FC)
    F = Icase==23 # input TC, pH
    if any(F):
        TA = where(F, TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        FC = where(F, fCO2fromTCpH(TC, PH, K0, Ks['K1'], Ks['K2']), FC)
        CARB = where(F, CarbfromTCpH(TC, PH, Ks['K1'], Ks['K2']), CARB)
    F = (Icase==24) | (Icase==25) # input TC, (pCO2 or fCO2)
    if any(F):
        PH = where(F, pHfromTCfCO2(TC, FC, K0, Ks['K1'], Ks['K2']), PH)
        TA = where(F, TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        CARB = where(F, CarbfromTCpH(TC, PH, Ks['K1'], Ks['K2']), CARB)
    F = Icase==26 # input TC, CARB
    if any(F):
        PH = where(F, pHfromTCCarb(TC, CARB, Ks['K1'], Ks['K2']), PH)
        FC = where(F, fCO2fromTCpH(TC, PH, K0, Ks['K1'], Ks['K2']), FC)
        TA = where(F, TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
    F = (Icase==34) | (Icase==35) # input pH, (pCO2 or fCO2)
    if any(F):
        TC = where(F, TCfrompHfCO2(PH, FC, K0, Ks['K1'], Ks['K2']), TC)
        TA = where(F, TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
        CARB = where(F, CarbfromTCpH(TC, PH, Ks['K1'], Ks['K2']), CARB)
    F = Icase==36 # input pH, CARB
    if any(F):
        FC = where(F, fCO2frompHCarb(PH, CARB, K0, Ks['K1'], Ks['K2']), FC)
        TC = where(F, TCfrompHfCO2(PH, FC, K0, Ks['K1'], Ks['K2']), TC)
        TA = where(F, TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
    F = (Icase==46) | (Icase==56) # input (pCO2 or fCO2), CARB
    if any(F):
        PH = where(F, pHfromfCO2Carb(FC, CARB, K0, Ks['K1'], Ks['K2']), PH)
        TC = where(F, TCfrompHfCO2(PH, FC, K0, Ks['K1'], Ks['K2']), TC)
        TA = where(F, TAfromTCpH(TC, PH, **Ks, **totals) + PengCx, TA)
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 values:
    PC = where(~PCgiven, FC/FugFac, PC)
    return TA, TC, PH, PC, FC, CARB

def from2to6variables(TempC, Pdbar, Sal, totals, pHScale, WhichKs, WhoseKSO4,
        WhoseKF):
    """Calculate variables (w.r.t. temperature and pressure) in preparation for
    `fill6` or `_from2to6` functions."""
    # Calculate the constants for all samples at input conditions.
    # The constants calculated for each sample will be on the input pH scale.
    K0, fH, Ks = assemble.equilibria(TempC, Pdbar, Sal, totals, pHScale,
        WhichKs, WhoseKSO4, WhoseKF)
    # Make sure fCO2 is available for each sample that has pCO2
    FugFac = gas.fugacityfactor(TempC, WhichKs)
    return K0, fH, Ks, FugFac

def from2to6(par1, par2, par1type, par2type, PengCx, totals, K0, FugFac, Ks):
    """Solve the core marine carbonate system from any 2 of its variables."""
    # Expand inputs `PAR1` and `PAR2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB = pars2mcs(par1, par2, par1type, par2type)
    # Generate vector describing the combination(s) of input parameters
    Icase = getIcase(par1type, par2type)
    # Solve the core marine carbonate system
    TA, TC, PH, PC, FC, CARB = fill6(Icase, K0, TA, TC, PH, PC, FC, CARB,
        PengCx, FugFac, Ks, totals)
    return TA, TC, PH, PC, FC, CARB

def allothers(TA, TC, PH, PC, CARB, Sal, TempC, Pdbar, K0, Ks, fH, totals,
        PengCx, TCa, pHScale, WhichKs):
    # pKs
    pK1 = -log10(Ks['K1'])
    pK2 = -log10(Ks['K2'])
    # Components of alkalinity and DIC
    FREEtoTOT = convert.free2tot(totals['TSO4'], Ks['KSO4'])
    HCO3, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
        AlkParts(PH, TC, FREEtoTOT, **Ks, **totals)
    PAlk = PAlk + PengCx
    CO2 = TC - CARB - HCO3
    # CaCO3 solubility
    OmegaCa, OmegaAr = solubility.CaCO3(Sal, TempC, Pdbar, CARB, TCa, WhichKs,
        Ks['K1'], Ks['K2'])
    # Dry mole fraction of CO2
    VPFac = gas.vpfactor(TempC, Sal)
    xCO2dry = PC/VPFac # this assumes pTot = 1 atm
    # Just for reference, convert pH at input conditions to the other scales
    pHT, pHS, pHF, pHN = convert.pH2allscales(PH, pHScale, Ks['KSO4'], Ks['KF'],
        totals['TSO4'], totals['TF'], fH)
    # Buffers by explicit calculation
    Revelle = buffers.RevelleFactor(TA-PengCx, TC, K0, Ks, totals)
    # Evaluate ESM10 buffer factors (corrected following RAH18) [added v1.2.0]
    gammaTC, betaTC, omegaTC, gammaTA, betaTA, omegaTA = \
        buffers.buffers_ESM10(TC, TA, CO2, HCO3, CARB, PH, OH, BAlk, Ks['KB'])
    # Evaluate (approximate) isocapnic quotient [HDW18] and psi [FCG94]
    # [added v1.2.0]
    isoQ = buffers.bgc_isocap(CO2, PH, Ks['K1'], Ks['K2'], Ks['KB'], Ks['KW'],
        totals['TB'])
    isoQx = buffers.bgc_isocap_approx(TC, PC, K0, Ks['K1'], Ks['K2'])
    psi = buffers.psi(CO2, PH, Ks['K1'], Ks['K2'], Ks['KB'], Ks['KW'],
        totals['TB'])
    return (pK1, pK2, HCO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4,
        HF, CO2, OmegaCa, OmegaAr, VPFac, xCO2dry, pHT, pHS, pHF, pHN, Revelle,
        gammaTC, betaTC, omegaTC, gammaTA, betaTA, omegaTA, isoQ, isoQx, psi)
