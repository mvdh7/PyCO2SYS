# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Calculate one new carbonate system variable from various input pairs."""
from autograd.numpy import errstate, log, log10, nan, sqrt, where
from autograd.numpy import abs as np_abs
from autograd.numpy import any as np_any
from .. import convert
from . import initialise

pHTol = 1e-8  # set tolerance for ending iterations in all pH solvers


def CarbfromTCH(TC, H, K1, K2):
    """Calculate carbonate ion from dissolved inorganic carbon and [H+].

    Based on CalculateCarbfromTCpH, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    return TC * K1 * K2 / (H ** 2 + K1 * H + K1 * K2)


def CarbfromTCpH(TC, pH, K1, K2):
    """Calculate carbonate ion from dissolved inorganic carbon and pH."""
    H = 10.0 ** -pH
    return CarbfromTCH(TC, H, K1, K2)


def HCO3fromTCH(TC, H, K1, K2):
    """Calculate bicarbonate ion from dissolved inorganic carbon and [H+]."""
    return TC * K1 * H / (H ** 2 + K1 * H + K1 * K2)


def HCO3fromTCpH(TC, pH, K1, K2):
    """Calculate bicarbonate ion from dissolved inorganic carbon and pH."""
    H = 10.0 ** -pH
    return HCO3fromTCH(TC, H, K1, K2)


def AlkParts(
    pH,
    TC,
    FREEtoTOT,
    K1=1.0,
    K2=1.0,
    KW=1.0,
    KB=1.0,
    KF=1.0,
    KSO4=1.0,
    KP1=1.0,
    KP2=1.0,
    KP3=1.0,
    KSi=1.0,
    KNH3=1.0,
    KH2S=0.0,
    TB=0.0,
    TF=0.0,
    TSO4=0.0,
    TPO4=0.0,
    TSi=0.0,
    TNH3=0.0,
    TH2S=0.0,
):
    """Calculate the different components of total alkalinity from pH and dissolved
    inorganic carbon.

    Although coded for H on the Total pH scale, for the pH values occuring in seawater
    (pH > 6) this will be equally valid on any pH scale (i.e. H terms are negligible) as
    long as the K Constants are on that scale.

    Based on CalculateAlkParts, version 01.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0 ** -pH
    HCO3 = HCO3fromTCH(TC, H, K1, K2)
    CO3 = CarbfromTCH(TC, H, K1, K2)
    BAlk = TB * KB / (KB + H)
    OH = KW / H
    PAlk = (
        TPO4
        * (KP1 * KP2 * H + 2 * KP1 * KP2 * KP3 - H ** 3)
        / (H ** 3 + KP1 * H ** 2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )
    SiAlk = TSi * KSi / (KSi + H)
    NH3Alk = TNH3 * KNH3 / (KNH3 + H)
    H2SAlk = TH2S * KH2S / (KH2S + H)
    Hfree = H / FREEtoTOT  # for H on the Total scale
    HSO4 = TSO4 / (1 + KSO4 / Hfree)  # since KSO4 is on the Free scale
    HF = TF / (1 + KF / Hfree)  # since KF is on the Free scale
    return HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF


@errstate(invalid="ignore")
def pHfromTATC(TA, TC, Ks, totals):
    """Calculate pH from total alkalinity and dissolved inorganic carbon.

    This calculates pH from TA and TC using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess uses the carbonate-borate alkalinity estimate of M13 and OE15 as
    implemented in mocsy.

    Although it is coded for H on the total pH scale, for the pH values occuring in
    seawater (pH > 6) it will be equally valid on any pH scale (H terms negligible) as
    long as the K Constants are on that scale.

    Based on CalculatepHfromTATC, version 04.01, 10-13-96, by Ernie Lewis.
    SVH2007: Made this to accept vectors. It will continue iterating until all values in
    the vector are "abs(deltapH) < pHTol".
    """
    # Relabel for convenience
    TB = totals["TB"]
    TSO4 = totals["TSO4"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    KB = Ks["KB"]
    KSO4 = Ks["KSO4"]
    # Solve
    pH = initialise.fromTC(TA, TC, TB, K1, K2, KB)  # first guess, added v1.3.0
    deltapH = 1.0 + pHTol
    ln10 = log(10)
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    while np_any(np_abs(deltapH) >= pHTol):
        pHdone = np_abs(deltapH) < pHTol
        HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = AlkParts(
            pH, TC, FREEtoTOT, **Ks, **totals
        )
        CAlk = HCO3 + 2 * CO3
        H = 10.0 ** -pH
        Denom = H ** 2 + K1 * H + K1 * K2
        Residual = (
            TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF
        )
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
        Slope = ln10 * (
            TC * K1 * H * (H ** 2 + K1 * K2 + 4 * H * K2) / Denom ** 2
            + BAlk * H / (KB + H)
            + OH
            + H
        )  # terms after here would add nutrients
        # SiAlk*H/(KSi + H) + NH3Alk*H/(KNH3 + H) + H2SAlk*H/(KH2S + H))
        # + PSlope) # to add phosphate component
        deltapH = Residual / Slope  # this is Newton's method
        # To keep the jump from being too big:
        deltapH = where(np_abs(deltapH) > 1, deltapH / 2, deltapH)
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        pH = where(pHdone, pH, pH + deltapH)
        # ^pH is on the same scale as K1 and K2 were calculated.
    return pH


@errstate(invalid="ignore")
def pHfromTAfCO2(TA, fCO2, K0, Ks, totals):
    """Calculate pH from total alkalinity and CO2 fugacity.

    This calculates pH from TA and fCO2 using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculatepHfromTAfCO2, version 04.01, 10-13-97, by Ernie Lewis.
    """
    # Relabel for convenience
    TB = totals["TB"]
    TSO4 = totals["TSO4"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    KB = Ks["KB"]
    KSO4 = Ks["KSO4"]
    # Solve
    pH = initialise.fromCO2(TA, K0 * fCO2, TB, K1, K2, KB)  # first guess, added v1.3.0
    deltapH = 1.0 + pHTol
    ln10 = log(10)
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    while np_any(np_abs(deltapH) > pHTol):
        pHdone = np_abs(deltapH) < pHTol
        H = 10.0 ** -pH
        HCO3 = K0 * K1 * fCO2 / H
        CO3 = K0 * K1 * K2 * fCO2 / H ** 2
        CAlk = HCO3 + 2 * CO3
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = AlkParts(
            pH, 0.0, FREEtoTOT, **Ks, **totals
        )
        Residual = (
            TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF
        )
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10 * (HCO3 + 4 * CO3 + BAlk * H / (KB + H) + OH + H)
        deltapH = Residual / Slope  # this is Newton's method
        # To keep the jump from being too big:
        deltapH = where(np_abs(deltapH) > 1, deltapH / 2, deltapH)
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        pH = where(pHdone, pH, pH + deltapH)
    return pH


@errstate(invalid="ignore")
def pHfromTACarb(TA, CARB, Ks, totals):
    """Calculate pH from total alkalinity and carbonate ion.

    This calculates pH from TA and Carb using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K constants are on that scale.

    Based on CalculatepHfromTACarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    # Relabel for convenience
    TB = totals["TB"]
    TSO4 = totals["TSO4"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    KB = Ks["KB"]
    KSO4 = Ks["KSO4"]
    # Solve
    pH = initialise.fromCO3(TA, CARB, TB, K1, K2, KB)  # first guess
    deltapH = 1.0 + pHTol
    ln10 = log(10)
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    while np_any(np_abs(deltapH) > pHTol):
        pHdone = np_abs(deltapH) < pHTol
        H = 10.0 ** -pH
        CAlk = CARB * (H + 2 * K2) / K2
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = AlkParts(
            pH, 0.0, FREEtoTOT, **Ks, **totals
        )
        Residual = (
            TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF
        )
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10 * (-CARB * H / K2 + BAlk * H / (KB + H) + OH + H)
        deltapH = Residual / Slope  # this is Newton's method
        # To keep the jump from being too big:
        deltapH = where(np_abs(deltapH) > 1, deltapH / 2, deltapH)
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        pH = where(pHdone, pH, pH + deltapH)
    return pH


@errstate(invalid="ignore")
def pHfromTAHCO3(TA, HCO3, Ks, totals):
    """Calculate pH from total alkalinity and bicarbonate ion.

    This calculates pH from TA and HCO3 using K1 and K2 by Newton's method.
    It tries to solve for the pH at which Residual = 0.
    The starting guess is pH = 8.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K constants are on that scale.
    """
    # Relabel for convenience
    TB = totals["TB"]
    TSO4 = totals["TSO4"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    KB = Ks["KB"]
    KSO4 = Ks["KSO4"]
    # Solve
    pH = initialise.fromHCO3(TA, HCO3, TB, K1, K2, KB)  # first guess
    deltapH = 1.0 + pHTol
    ln10 = log(10)
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    while np_any(np_abs(deltapH) >= pHTol):
        pHdone = np_abs(deltapH) < pHTol
        H = 10.0 ** -pH
        CAlk = HCO3 * (1 + 2 * K2 / H)
        _, _, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = AlkParts(
            pH, 0.0, FREEtoTOT, **Ks, **totals
        )
        Residual = (
            TA - CAlk - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF
        )
        # Find Slope dTA/dpH (this is not exact, but keeps all important terms)
        Slope = ln10 * (2 * HCO3 * K2 / H + BAlk * H / (KB + H) + OH + H)
        deltapH = Residual / Slope  # this is Newton's method
        # To keep the jump from being too big:
        deltapH = where(np_abs(deltapH) > 1, deltapH / 2, deltapH)
        # The following logical means that each row stops updating once its
        # deltapH value is beneath the pHTol threshold, instead of continuing
        # to update ALL rows until they all meet the threshold.
        # This approach avoids the problem of reaching a different
        # answer for a given set of input conditions depending on how many
        # iterations the other input rows take to solve. // MPH
        pH = where(pHdone, pH, pH + deltapH)
    return pH


def fCO2fromTCpH(TC, pH, K0, K1, K2):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.

    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0 ** -pH
    return TC * H ** 2 / (H ** 2 + K1 * H + K1 * K2) / K0


def TCfromTApH(TA, pH, Ks, totals):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.

    This calculates TC from TA and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.
    """
    # Relabel for convenience
    TSO4 = totals["TSO4"]
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    KSO4 = Ks["KSO4"]
    # Solve
    H = 10.0 ** -pH
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = AlkParts(
        pH, 0.0, FREEtoTOT, **Ks, **totals
    )
    CAlk = TA - BAlk - OH - PAlk - SiAlk - NH3Alk - H2SAlk + Hfree + HSO4 + HF
    TC = CAlk * (H ** 2 + K1 * H + K1 * K2) / (K1 * (H + 2 * K2))
    return TC


def TAfromTCpH(TC, pH, Ks, totals):
    """Calculate total alkalinity from dissolved inorganic carbon and pH.

    This calculates TA from TC and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTAfromTCpH, version 02.02, 10-10-97, by Ernie Lewis.
    """
    # Relabel for convenience
    TSO4 = totals["TSO4"]
    KSO4 = Ks["KSO4"]
    # Solve
    FREEtoTOT = convert.free2tot(TSO4, KSO4)
    HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = AlkParts(
        pH, TC, FREEtoTOT, **Ks, **totals
    )
    CAlk = HCO3 + 2 * CO3
    TAc = CAlk + BAlk + OH + PAlk + SiAlk + NH3Alk + H2SAlk - Hfree - HSO4 - HF
    return TAc


@errstate(invalid="ignore")
def pHfromTCfCO2(TC, fCO2, K0, K1, K2):
    """Calculate pH from dissolved inorganic carbon and CO2 fugacity.

    This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the quadratic in
    H: fCO2*K0 = TC*H*H/(K1*H + H*H + K1*K2).
    If there is not a real root, then pH is returned as NaN.

    Based on CalculatepHfromTCfCO2, version 02.02, 11-12-96, by Ernie Lewis.
    """
    RR = K0 * fCO2 / TC
    Discr = (K1 * RR) ** 2 + 4 * (1 - RR) * K1 * K2 * RR
    H = 0.5 * (K1 * RR + sqrt(Discr)) / (1 - RR)
    H = where(H < 0, nan, H)
    pH = -log10(H)
    return pH


def TCfrompHfCO2(pH, fCO2, K0, K1, K2):
    """Calculate dissolved inorganic carbon from pH and CO2 fugacity.

    Based on CalculateTCfrompHfCO2, version 01.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0 ** -pH
    return K0 * fCO2 * (H ** 2 + K1 * H + K1 * K2) / H ** 2


def pHfromTCCarb(TC, CARB, K1, K2):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    This calculates pH from Carbonate and TC using K1, and K2 by solving the
    quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).

    Based on CalculatepHfromTCCarb, version 01.00, 06-12-2019, by Denis Pierrot.
    """
    RR = 1 - TC / CARB
    Discr = K1 ** 2 - 4 * K1 * K2 * RR
    H = (-K1 + sqrt(Discr)) / 2
    return -log10(H)


def fCO2frompHCarb(pH, CARB, K0, K1, K2):
    """Calculate CO2 fugacity from pH and carbonate ion.

    Based on CalculatefCO2frompHCarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    H = 10.0 ** -pH
    return CARB * H ** 2 / (K0 * K1 * K2)


def TCfrompHHCO3(pH, HCO3, K1, K2):
    """Calculate dissolved inorganic carbon from pH and bicarbonate ion.

    Follows ZW01 Appendix B (6).
    """
    H = 10.0 ** -pH
    return HCO3 * (1 + H / K1 + K2 / H)


def pHfromfCO2Carb(fCO2, CARB, K0, K1, K2):
    """Calculate pH from CO2 fugacity and carbonate ion.

    This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving
    the equation in H: fCO2 * K0 * K1* K2 = Carb * H * H

    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis
    Pierrot.
    """
    H = sqrt(K0 * K1 * K2 * fCO2 / CARB)
    return -log10(H)


def pHfromTCHCO3(TC, HCO3, K1, K2):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    Follows ZW01 Appendix B (12).
    """
    a = HCO3 / K1
    b = HCO3 - TC
    c = HCO3 * K2
    H = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return -log10(H)


def CarbfromfCO2HCO3(fCO2, HCO3, K0, K1, K2):
    """Calculate carbonate ion from CO2 fugacity and bicarbonate ion."""
    return HCO3 ** 2 * K2 / (K0 * fCO2 * K1)


def fCO2fromCarbHCO3(CARB, HCO3, K0, K1, K2):
    """Calculate CO2 fugacity from carbonate ion and bicarbonate ion."""
    return HCO3 ** 2 * K2 / (CARB * K1 * K0)
