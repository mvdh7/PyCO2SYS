# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Solve the marine carbonate system from any two of its variables."""

from autograd.numpy import (array, full, full_like, log, log10, nan, size, sqrt,
                            where)
from autograd.numpy import abs as np_abs
from autograd.numpy import any as np_any
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max
from . import convert

pHTol = 1e-6 # tolerance for ending iterations in all pH solvers

def AlkParts(pH, TC,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
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
    CO3 = TC*K1*K2/(K1*H + H**2 + K1*K2)
    BAlk = TB*KB/(KB + H)
    OH = KW/H
    PAlk = (TPO4*(KP1*KP2*H + 2*KP1*KP2*KP3 - H**3)/
            (H**3 + KP1*H**2 + KP1*KP2*H + KP1*KP2*KP3))
    SiAlk = TSi*KSi/(KSi + H)
    NH3Alk = TNH3*KNH3/(KNH3 + H)
    H2SAlk = TH2S*KH2S/(KH2S + H)
    FREEtoTOT = convert.free2tot(TSO4, KS)
    Hfree = H/FREEtoTOT # for H on the Total scale
    HSO4 = TSO4/(1 + KS/Hfree) # since KS is on the Free scale
    HF = TF/(1 + KF/Hfree) # since KF is on the Free scale
    return HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF

def pHfromTATC(TA, TC,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate pH from total alkalinity and dissolved inorganic carbon.

    This calculates pH from TA and TC using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculatepHfromTATC, version 04.01, 10-13-96, by Ernie Lewis.
    SVH2007: Made this to accept vectors. It will continue iterating until all
    values in the vector are "abs(deltapH) < pHTol".
    """
    pHGuess = 8.0 # this is the first guess
    pH = full(size(TA), pHGuess) # first guess for all samples
    deltapH = 1.0 + pHTol
    ln10 = log(10)
    while np_any(np_abs(deltapH) > pHTol):
        HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            AlkParts(pH, TC,
                K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TSO4, TPO4, TSi, TNH3, TH2S)
        CAlk = HCO3 + 2*CO3
        H = 10.0**-pH
        Denom = H**2 + K1*H + K1*K2
        Residual = (TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                    Hfree + HSO4 + HF)
        # Find slope dTA/dpH (this is not exact, but keeps important terms):
        Slope = ln10*(TC*K1*H*(H**2 + K1*K2 + 4*H*K2)/Denom**2 +
                      BAlk*H/(KB + H) + OH + H)
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

def fCO2fromTCpH(TC, pH, K0, K1, K2):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.

    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0**-pH
    fCO2 = TC*H**2/(H**2 + K1*H + K1*K2)/K0
    return fCO2

def TCfromTApH(TA, pH,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.

    This calculates TC from TA and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0**-pH
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
        AlkParts(pH, 0.0,
            K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
            TB, TF, TSO4, TPO4, TSi, TNH3, TH2S)
    CAlk = (TA - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 +
            HF)
    TC = CAlk*(H**2 + K1*H + K1*K2)/(K1*(H + 2*K2))
    return TC

def pHfromTAfCO2(TA, fCO2, K0,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
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
    pH = full_like(TA, pHGuess) # first guess for all samples
    deltapH = 1 + pHTol
    ln10 = log(10)
    while np_any(np_abs(deltapH) > pHTol):
        H = 10.0**-pH
        HCO3 = K0*K1*fCO2/H
        CO3 = K0*K1*K2*fCO2/H**2
        CAlk = HCO3 + 2*CO3
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            AlkParts(pH, 0.0,
                K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
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

def TAfromTCpH(TC, pH,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Calculate total alkalinity from dissolved inorganic carbon and pH.

    This calculates TA from TC and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTAfromTCpH, version 02.02, 10-10-97, by Ernie Lewis.
    """
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
        AlkParts(pH, TC,
            K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
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

def pHfromTACarb(TA, CARB,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
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
    pH = full_like(TA, pHGuess) # first guess for all samples
    deltapH = 1 + pHTol
    ln10 = log(10)
    while np_any(np_abs(deltapH) > pHTol):
        H = 10.0**-pH
        CAlk = CARB*(H + 2*K2)/K2
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            AlkParts(pH, 0.0,
                K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
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

def CarbfromTCpH(TC, pH, K1, K2):
    """Calculate carbonate ion from dissolved inorganic carbon and pH.

    Based on CalculateCarbfromTCpH, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    H = 10.0**-pH
    CARB = TC*K1*K2/(H**2 + K1*H + K1*K2)
    return CARB

def from2to6(p1, p2, K0, Ks, TA, TC,
        PH, PC, FC, CARB, PengCorrection, FugFac,
        TB, TF, TSO4, TPO4, TSi, TNH3, TH2S):
    """Solve the marine carbonate system from any valid pair of inputs."""
    K1 = Ks[0]
    K2 = Ks[1]
    Ts = [TB, TF, TSO4, TPO4, TSi, TNH3, TH2S]
    # Generate vector describing the combination of input parameters
    # So, the valid ones are: 12,13,14,15,16,23,24,25,26,34,35,36,46,56
    Icase = (10*np_min(array([p1, p2]), axis=0) +
        np_max(array([p1, p2]), axis=0))
    # Calculate missing values for AT, CT, PH, FC, CARB:
    # pCO2 will be calculated later on, routines work with fCO2.
    F = Icase==12 # input TA, TC
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PH[F] = pHfromTATC(TA[F]-PengCorrection[F], TC[F], *KFs, *TFs)
        # ^pH is returned on the scale requested in "pHscale" (see 'constants')
        FC[F] = fCO2fromTCpH(TC[F], PH[F], K0[F], K1[F], K2[F])
        CARB[F] = CarbfromTCpH(TC[F], PH[F], K1[F], K2[F])
    F = Icase==13 # input TA, pH
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        TC[F] = TCfromTApH(TA[F]-PengCorrection[F], PH[F], *KFs, *TFs)
        FC[F] = fCO2fromTCpH(TC[F], PH[F], K0[F], K1[F], K2[F])
        CARB[F] = CarbfromTCpH(TC[F], PH[F], K1[F], K2[F])
    F = (Icase==14) | (Icase==15) # input TA, (pCO2 or fCO2)
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PH[F] = pHfromTAfCO2(TA[F]-PengCorrection[F], FC[F], K0[F], *KFs, *TFs)
        TC[F] = TCfromTApH(TA[F]-PengCorrection[F], PH[F], *KFs, *TFs)
        CARB[F] = CarbfromTCpH(TC[F], PH[F], K1[F], K2[F])
    F = Icase==16 # input TA, CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PH[F] = pHfromTACarb(TA[F]-PengCorrection[F], CARB[F], *KFs, *TFs)
        TC[F] = TCfromTApH(TA[F]-PengCorrection[F], PH[F], *KFs, *TFs)
        FC[F] = fCO2fromTCpH(TC[F], PH[F], K0[F], K1[F], K2[F])
    F = Icase==23 # input TC, pH
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        TA[F] = TAfromTCpH(TC[F], PH[F], *KFs, *TFs) + PengCorrection[F]
        FC[F] = fCO2fromTCpH(TC[F], PH[F], K0[F], K1[F], K2[F])
        CARB[F] = CarbfromTCpH(TC[F], PH[F], K1[F], K2[F])
    F = (Icase==24) | (Icase==25) # input TC, (pCO2 or fCO2)
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PH[F] = pHfromTCfCO2(TC[F], FC[F], K0[F], K1[F], K2[F])
        TA[F] = TAfromTCpH(TC[F], PH[F], *KFs, *TFs) + PengCorrection[F]
        CARB[F] = CarbfromTCpH(TC[F], PH[F], K1[F], K2[F])
    F = Icase==26 # input TC, CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PH[F] = pHfromTCCarb(TC[F], CARB[F], K1[F], K2[F])
        FC[F] = fCO2fromTCpH(TC[F], PH[F], K0[F], K1[F], K2[F])
        TA[F] = TAfromTCpH(TC[F], PH[F], *KFs, *TFs) + PengCorrection[F]
    F = (Icase==34) | (Icase==35) # input pH, (pCO2 or fCO2)
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        TC[F] = TCfrompHfCO2(PH[F], FC[F], K0[F], K1[F], K2[F])
        TA[F] = TAfromTCpH(TC[F], PH[F], *KFs, *TFs) + PengCorrection[F]
        CARB[F] = CarbfromTCpH(TC[F], PH[F], K1[F], K2[F])
    F = Icase==36 # input pH, CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        FC[F] = fCO2frompHCarb(PH[F], CARB[F], K0[F], K1[F], K2[F])
        TC[F] = TCfrompHfCO2(PH[F], FC[F], K0[F], K1[F], K2[F])
        TA[F] = TAfromTCpH(TC[F], PH[F], *KFs, *TFs) + PengCorrection[F]
    F = (Icase==46) | (Icase==56) # input (pCO2 or fCO2), CARB
    if any(F):
        KFs, TFs = [[X[F] for X in Xs] for Xs in [Ks, Ts]]
        PH[F] = pHfromfCO2Carb(FC[F], CARB[F], K0[F], K1[F], K2[F])
        TC[F] = TCfrompHfCO2(PH[F], FC[F], K0[F], K1[F], K2[F])
        TA[F] = TAfromTCpH(TC[F], PH[F], *KFs, *TFs) + PengCorrection[F]
    # By now, an fCO2 value is available for each sample.
    # Generate the associated pCO2 value:
    PC = FC/FugFac
    return TA, TC, PH, PC, FC, CARB
