from numpy import full_like, log, log10, nan, sqrt
from numpy import abs as np_abs
from numpy import any as np_any
from . import convert

pHTol = 1e-6 # tolerance for ending iterations in all pH solvers

def AlkParts(pH, TC,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
    """Calculate the different components of total alkalinity from pH and
    dissolved inorganic carbon.

    Although coded for H on the total pH scale, for the pH values occuring in
    seawater (pH > 6) this will be equally valid on any pH scale (i.e. H terms
    are negligible) as long as the K Constants are on that scale.

    Based on CalculateAlkParts, version 01.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0**-pH
    HCO3 = TC*K1*H/(K1*H + H**2 + K1*K2)
    CO3 = TC*K1*K2/(K1*H + H**2 + K1*K2)
    BAlk = TB*KB/(KB + H)
    OH = KW/H
    PAlk = (TP*(KP1*KP2*H + 2*KP1*KP2*KP3 - H**3)/
            (H**3 + KP1*H**2 + KP1*KP2*H + KP1*KP2*KP3))
    SiAlk = TSi*KSi/(KSi + H)
    NH3Alk = TNH3*KNH3/(KNH3 + H)
    H2SAlk = TH2S*KH2S/(KH2S + H)
    FREEtoTOT = convert.free2tot(TS, KS)
    Hfree = H/FREEtoTOT # for H on the Total scale
    HSO4 = TS/(1 + KS/Hfree) # since KS is on the Free scale
    HF = TF/(1 + KF/Hfree) # since KF is on the Free scale
    return HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF

def pHfromTATC(TA, TC,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
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
    pH = full_like(TA, pHGuess) # first guess for all samples
    deltapH = 1 + pHTol
    ln10 = log(10)
    while np_any(np_abs(deltapH) > pHTol):
        HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
            AlkParts(pH, TC,
                K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
                TB, TF, TS, TP, TSi, TNH3, TH2S)
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
        while any(np_abs(deltapH) > 1):
            FF = np_abs(deltapH) > 1
            deltapH[FF] /= 2.0
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        F = np_abs(deltapH) > pHTol
        pH[F] += deltapH[F]
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
        TB, TF, TS, TP, TSi, TNH3, TH2S):
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
            TB, TF, TS, TP, TSi, TNH3, TH2S)
    CAlk = (TA - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 +
            HF)
    TC = CAlk*(H**2 + K1*H + K1*K2)/(K1*(H + 2*K2))
    return TC

def pHfromTAfCO2(TA, fCO2, K0,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
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
                TB, TF, TS, TP, TSi, TNH3, TH2S)
        Residual = (TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk +
                    Hfree + HSO4 + HF)
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10*(HCO3 + 4*CO3 + BAlk*H/(KB + H) + OH + H)
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        while np_any(np_abs(deltapH) > 1):
            FF = np_abs(deltapH) > 1
            if any(FF):
                deltapH[FF] /= 2
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        F = np_abs(deltapH) > pHTol
        pH[F] += deltapH[F]
    return pH

def TAfromTCpH(TC, pH,
        K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
        TB, TF, TS, TP, TSi, TNH3, TH2S):
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
            TB, TF, TS, TP, TSi, TNH3, TH2S)
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
        TB, TF, TS, TP, TSi, TNH3, TH2S):
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
                TB, TF, TS, TP, TSi, TNH3, TH2S)
        Residual = (TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk -
                    H2SAlk + Hfree + HSO4 + HF)
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10*(-CARB*H/K2 + BAlk*H/(KB + H) + OH + H)
        deltapH = Residual/Slope # this is Newton's method
        # To keep the jump from being too big:
        while np_any(np_abs(deltapH) > 1):
            FF = np_abs(deltapH) > 1
            if any(FF):
                deltapH[FF] /= 2
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        F = np_abs(deltapH) > pHTol
        pH[F] += deltapH[F]
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
