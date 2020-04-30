# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Calculate one new carbonate system variable from various input pairs."""

from autograd.numpy import errstate, log10, nan, sign, sqrt, where
from autograd.numpy import abs as np_abs
from autograd.numpy import any as np_any
from .. import convert
from . import delta, initialise

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


@errstate(invalid="ignore")
def HCO3fromTCH(TC, H, K1, K2):
    """Calculate bicarbonate ion from dissolved inorganic carbon and [H+]."""
    return TC * K1 * H / (H ** 2 + K1 * H + K1 * K2)


def HCO3fromTCpH(TC, pH, K1, K2):
    """Calculate bicarbonate ion from dissolved inorganic carbon and pH."""
    H = 10.0 ** -pH
    return HCO3fromTCH(TC, H, K1, K2)


@errstate(invalid="ignore")
def AlkParts(TC, pH, FREEtoTOT, Ks, totals):
    """Calculate the different components of total alkalinity from dissolved inorganic
    carbon and pH.

    Although coded for H on the Total pH scale, for the pH values occuring in seawater
    (pH > 6) this will be equally valid on any pH scale (i.e. H terms are negligible) as
    long as the K Constants are on that scale.

    Based on CalculateAlkParts, version 01.03, 10-10-97, by Ernie Lewis.
    """
    H = 10.0 ** -pH
    HCO3 = HCO3fromTCH(TC, H, Ks["K1"], Ks["K2"])
    CO3 = CarbfromTCH(TC, H, Ks["K1"], Ks["K2"])
    BAlk = totals["TB"] * Ks["KB"] / (Ks["KB"] + H)
    OH = Ks["KW"] / H
    PAlk = (
        totals["TPO4"]
        * (Ks["KP1"] * Ks["KP2"] * H + 2 * Ks["KP1"] * Ks["KP2"] * Ks["KP3"] - H ** 3)
        / (
            H ** 3
            + Ks["KP1"] * H ** 2
            + Ks["KP1"] * Ks["KP2"] * H
            + Ks["KP1"] * Ks["KP2"] * Ks["KP3"]
        )
    )
    SiAlk = totals["TSi"] * Ks["KSi"] / (Ks["KSi"] + H)
    NH3Alk = totals["TNH3"] * Ks["KNH3"] / (Ks["KNH3"] + H)
    H2SAlk = totals["TH2S"] * Ks["KH2S"] / (Ks["KH2S"] + H)
    Hfree = H / FREEtoTOT  # for H on the Total scale
    HSO4 = totals["TSO4"] / (1 + Ks["KSO4"] / Hfree)  # since KSO4 is on the Free scale
    HF = totals["TF"] / (1 + Ks["KF"] / Hfree)  # since KF is on the Free scale
    return {
        "HCO3": HCO3,
        "CO3": CO3,
        "BAlk": BAlk,
        "OH": OH,
        "PAlk": PAlk,
        "SiAlk": SiAlk,
        "NH3Alk": NH3Alk,
        "H2SAlk": H2SAlk,
        "Hfree": Hfree,
        "HSO4": HSO4,
        "HF": HF,
    }


def TAfromTCpH(TC, pH, Ks, totals):
    """Calculate total alkalinity from dissolved inorganic carbon and pH.

    This calculates TA from TC and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTAfromTCpH, version 02.02, 10-10-97, by Ernie Lewis.
    """
    FREEtoTOT = convert.free2tot(totals["TSO4"], Ks["KSO4"])
    alks = AlkParts(TC, pH, FREEtoTOT, Ks, totals)
    TA = (
        alks["HCO3"]
        + 2 * alks["CO3"]
        + alks["BAlk"]
        + alks["OH"]
        + alks["PAlk"]
        + alks["SiAlk"]
        + alks["NH3Alk"]
        + alks["H2SAlk"]
        - alks["Hfree"]
        - alks["HSO4"]
        - alks["HF"]
    )
    return TA


def TAfrompHfCO2(pH, fCO2, Ks, totals):
    """Calculate total alkalinity from dissolved inorganic carbon and CO2 fugacity."""
    TC = TCfrompHfCO2(pH, fCO2, Ks["K0"], Ks["K1"], Ks["K2"])
    return TAfromTCpH(TC, pH, Ks, totals)


def TCfrompHHCO3(pH, HCO3, K1, K2):
    """Calculate dissolved inorganic carbon from pH and bicarbonate ion.

    Follows ZW01 Appendix B (6).
    """
    H = 10.0 ** -pH
    return HCO3 * (1 + H / K1 + K2 / H)


def TCfrompHCarb(pH, CARB, K1, K2):
    """Calculate dissolved inorganic carbon from pH and carbonate ion.

    Follows ZW01 Appendix B (7).
    """
    H = 10.0 ** -pH
    return CARB * (1 + H / K2 + H ** 2 / (K1 * K2))


def TAfrompHCarb(pH, CARB, Ks, totals):
    """Calculate total alkalinity from dissolved inorganic carbon and carbonate ion."""
    TC = TCfrompHCarb(pH, CARB, Ks["K1"], Ks["K2"])
    return TAfromTCpH(TC, pH, Ks, totals)


def TAfrompHHCO3(pH, HCO3, Ks, totals):
    """Calculate total alkalinity from dissolved inorganic carbon and bicarbonate ion.
    """
    TC = TCfrompHHCO3(pH, HCO3, Ks["K1"], Ks["K2"])
    return TAfromTCpH(TC, pH, Ks, totals)


@errstate(invalid="ignore")
def _pHfromTAVX(TA, VX, Ks, totals, initialfunc, deltafunc):
    """Calculate pH from total alkalinity and DIC or one of its components using a
    Newton-Raphson iterative method.
    
    Although it is coded for H on the total pH scale, for the pH values occuring in
    seawater (pH > 6) it will be equally valid on any pH scale (H terms negligible) as
    long as the K Constants are on that scale.
    
    Based on the CalculatepHfromTA* functions, version 04.01, Oct 96, by Ernie Lewis.
    """
    # First guess inspired by M13/OE15, added v1.3.0:
    pH = initialfunc(TA, VX, totals["TB"], Ks["K1"], Ks["K2"], Ks["KB"])
    deltapH = 1.0 + pHTol
    FREEtoTOT = convert.free2tot(totals["TSO4"], Ks["KSO4"])
    while np_any(np_abs(deltapH) >= pHTol):
        pHdone = np_abs(deltapH) < pHTol  # check which rows don't need updating
        deltapH = deltafunc(pH, TA, VX, FREEtoTOT, Ks, totals)  # the pH jump
        # To keep the jump from being too big:
        abs_deltapH = np_abs(deltapH)
        sign_deltapH = sign(deltapH)
        # Jump by 1 instead if `deltapH` > 5
        deltapH = where(abs_deltapH > 5.0, sign_deltapH, deltapH)
        # Jump by 0.5 instead if 1 < `deltapH` < 5
        deltapH = where(
            (abs_deltapH > 0.5) & (abs_deltapH <= 5.0), 0.5 * sign_deltapH, deltapH,
        )  # assumes that once we're within 1 of the correct pH, we will converge
        pH = where(pHdone, pH, pH + deltapH)  # only update rows that need it
    return pH


def pHfromTATC(TA, TC, Ks, totals):
    """Calculate pH from total alkalinity and dissolved inorganic carbon."""
    return _pHfromTAVX(TA, TC, Ks, totals, initialise.fromTC, delta.pHfromTATC)


def pHfromTAfCO2(TA, fCO2, Ks, totals):
    """Calculate pH from total alkalinity and CO2 fugacity."""
    # Slightly more convoluted than the others because initialise.fromCO2 takes CO2 as
    # an input, while delta.pHfromTAfCO2 takes fCO2.
    return _pHfromTAVX(
        TA,
        fCO2,
        Ks,
        totals,
        lambda TA, fCO2, TB, K1, K2, KB: initialise.fromCO2(
            TA, Ks["K0"] * fCO2, TB, K1, K2, KB
        ),  # this just transforms initalise.fromCO2 to take fCO2 in place of CO2
        delta.pHfromTAfCO2,
    )


def pHfromTACarb(TA, CARB, Ks, totals):
    """Calculate pH from total alkalinity and carbonate ion molinity."""
    return _pHfromTAVX(TA, CARB, Ks, totals, initialise.fromCO3, delta.pHfromTACarb)


def pHfromTAHCO3(TA, HCO3, Ks, totals):
    """Calculate pH from total alkalinity and bicarbonate ion molinity."""
    return _pHfromTAVX(TA, HCO3, Ks, totals, initialise.fromHCO3, delta.pHfromTAHCO3)


def fCO2fromTCpH(TC, pH, K0, K1, K2):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.

    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0 ** -pH
    return TC * H ** 2 / (H ** 2 + K1 * H + K1 * K2) / K0


@errstate(invalid="ignore")
def TCfromTApH(TA, pH, Ks, totals):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.

    This calculates TC from TA and pH.
    Though it is coded for H on the total pH scale, for the pH values occuring
    in seawater (pH > 6) it will be equally valid on any pH scale (H terms
    negligible) as long as the K Constants are on that scale.

    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.
    """
    TA_TC0_pH = TAfromTCpH(0.0, pH, Ks, totals)
    F = TA_TC0_pH > TA
    if any(F):
        print("Some input pH values are impossibly high given the input alkalinity;")
        print("returning NaN rather than negative DIC values.")
    CAlk = where(F, nan, TA - TA_TC0_pH)
    K1 = Ks["K1"]
    K2 = Ks["K2"]
    H = 10.0 ** -pH
    TC = CAlk * (H ** 2 + K1 * H + K1 * K2) / (K1 * (H + 2 * K2))
    return TC


@errstate(divide="ignore", invalid="ignore")
def pHfromTCfCO2(TC, fCO2, K0, K1, K2):
    """Calculate pH from dissolved inorganic carbon and CO2 fugacity.

    This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the quadratic in
    H: fCO2*K0 = TC*H*H/(K1*H + H*H + K1*K2).
    If there is not a real root, then pH is returned as NaN.

    Based on CalculatepHfromTCfCO2, version 02.02, 11-12-96, by Ernie Lewis.
    """
    RR = K0 * fCO2 / TC
    Discr = (K1 * RR) ** 2 + 4 * (1 - RR) * K1 * K2 * RR
    F = (RR >= 1) | (Discr <= 0)
    if any(F):
        print("Some input fCO2 values are impossibly high given the input DIC;")
        print("returning NaN.")
    H = where(F, nan, 0.5 * (K1 * RR + sqrt(Discr)) / (1 - RR))
    pH = -log10(H)
    return pH


def TCfrompHfCO2(pH, fCO2, K0, K1, K2):
    """Calculate dissolved inorganic carbon from pH and CO2 fugacity.

    Based on CalculateTCfrompHfCO2, version 01.02, 12-13-96, by Ernie Lewis.
    """
    H = 10.0 ** -pH
    return K0 * fCO2 * (H ** 2 + K1 * H + K1 * K2) / H ** 2


@errstate(invalid="ignore")
def pHfromTCCarb(TC, CARB, K1, K2):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    This calculates pH from Carbonate and TC using K1, and K2 by solving the
    quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).

    Based on CalculatepHfromTCCarb, version 01.00, 06-12-2019, by Denis Pierrot.
    """
    RR = 1 - TC / CARB
    Discr = K1 ** 2 - 4 * K1 * K2 * RR
    F = (CARB >= TC) | (Discr <= 0)
    if any(F):
        print("Some input CO3 values are impossibly high given the input DIC;")
        print("returning NaN.")
    H = where(F, nan, (-K1 + sqrt(Discr)) / 2)
    return -log10(H)


def fCO2frompHCarb(pH, CARB, K0, K1, K2):
    """Calculate CO2 fugacity from pH and carbonate ion.

    Based on CalculatefCO2frompHCarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    H = 10.0 ** -pH
    return CARB * H ** 2 / (K0 * K1 * K2)


def pHfromfCO2Carb(fCO2, CARB, K0, K1, K2):
    """Calculate pH from CO2 fugacity and carbonate ion.

    This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving
    the equation in H: fCO2 * K0 * K1* K2 = Carb * H * H

    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis
    Pierrot.
    """
    H = sqrt(K0 * K1 * K2 * fCO2 / CARB)
    return -log10(H)


@errstate(invalid="ignore")
def pHfromTCHCO3(TC, HCO3, K1, K2):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    Follows ZW01 Appendix B (12).
    """
    a = HCO3 / K1
    b = HCO3 - TC
    c = HCO3 * K2
    bsq_4ac = b ** 2 - 4 * a * c
    F = (HCO3 >= TC) | (bsq_4ac <= 0)
    if any(F):
        print("Some input HCO3 values are impossibly high given the input DIC;")
        print("returning NaN.")
    H = where(F, nan, (-b - sqrt(bsq_4ac)) / (2 * a))
    return -log10(H)


def CarbfromfCO2HCO3(fCO2, HCO3, K0, K1, K2):
    """Calculate carbonate ion from CO2 fugacity and bicarbonate ion."""
    return HCO3 ** 2 * K2 / (K0 * fCO2 * K1)


def fCO2fromCarbHCO3(CARB, HCO3, K0, K1, K2):
    """Calculate CO2 fugacity from carbonate ion and bicarbonate ion."""
    return HCO3 ** 2 * K2 / (CARB * K1 * K0)


def fCO2fromTATC(TA, TC, Ks, totals):
    """Calculate CO2 fugacity from total alkalinity and dissolved inorganic carbon."""
    pH = pHfromTATC(TA, TC, Ks, totals)
    return fCO2fromTCpH(TC, pH, Ks["K0"], Ks["K1"], Ks["K2"])


def fCO2fromTApH(TA, pH, Ks, totals):
    """Calculate CO2 fugacity from total alkalinity and pH."""
    TC = TCfromTApH(TA, pH, Ks, totals)
    return fCO2fromTCpH(TC, pH, Ks["K0"], Ks["K1"], Ks["K2"])


def CarbfromTATC(TA, TC, Ks, totals):
    """Calculate carbonate ion from total alkalinity and dissolved inorganic carbon."""
    pH = pHfromTATC(TA, TC, Ks, totals)
    return CarbfromTCpH(TC, pH, Ks["K1"], Ks["K2"])


def CarbfromTApH(TA, pH, Ks, totals):
    """Calculate carbonate ion from total alkalinity and pH."""
    TC = TCfromTApH(TA, pH, Ks, totals)
    return CarbfromTCpH(TC, pH, Ks["K1"], Ks["K2"])
