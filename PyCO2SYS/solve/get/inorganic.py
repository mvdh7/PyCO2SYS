# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate one new carbonate system variable from various input pairs considering
only inorganic solutes (i.e., no DOM) and with a fixed ZLP.
"""

from jax import numpy as np, lax
from ... import salts
from .. import delta, initialise, residual, speciate


def alkalinity_from_dic_pH(dic, pH, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and pH."""
    sw = speciate.inorganic(dic, pH, totals, k_constants)
    return sw["alkalinity"] * 1e6


def alkalinity_from_pH_fCO2(pH, fCO2, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and CO2 fugacity."""
    dic = dic_from_pH_fCO2(pH, fCO2, totals, k_constants)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants)


def alkalinity_from_pH_carbonate(pH, carbonate, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and carbonate ion."""
    dic = dic_from_pH_carbonate(pH, carbonate, totals, k_constants)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants)


def alkalinity_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and bicarbonate ion."""
    dic = dic_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants)


def alkalinity_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants):
    """Total alkalinity from CO2 fugacity and carbonate ion."""
    pH = pH_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants)
    return alkalinity_from_pH_fCO2(pH, fCO2, totals, k_constants)


def alkalinity_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants):
    """Total alkalinity from CO2 fugacity and bicarbonate ion."""
    carbonate = carbonate_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants)
    return alkalinity_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants)


def alkalinity_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants):
    """Total alkalinity from carbonate ion and carbonate ion."""
    pH = pH_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants)
    return alkalinity_from_pH_carbonate(pH, carbonate, totals, k_constants)


def dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.
    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.
    """
    alkalinity_with_zero_dic = alkalinity_from_dic_pH(0.0, pH, totals, k_constants)
    F = alkalinity_with_zero_dic > alkalinity
    if np.any(F):
        print("Some input pH values are impossibly high given the input alkalinity;")
        print("returning np.nan rather than negative DIC values.")
    alkalinity_carbonate = np.where(F, np.nan, alkalinity - alkalinity_with_zero_dic)
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    H = 10.0**-pH
    dic = alkalinity_carbonate * (H**2 + K1 * H + K1 * K2) / (K1 * (H + 2 * K2))
    return dic


def dic_from_pH_fCO2(pH, fCO2, totals, k_constants):
    """Calculate dissolved inorganic carbon from pH and CO2 fugacity.
    Based on CalculateTCfrompHfCO2, version 01.02, 12-13-96, by Ernie Lewis.
    """
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    H = 10.0**-pH
    return K0 * fCO2 * (H**2 + K1 * H + K1 * K2) / H**2


def dic_from_pH_carbonate(pH, carbonate, totals, k_constants):
    """Calculate dissolved inorganic carbon from pH and carbonate ion.
    Follows ZW01 Appendix B (7).
    """
    H = 10.0**-pH
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    return carbonate * (1 + H / K2 + H**2 / (K1 * K2))


def dic_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants):
    """Calculate dissolved inorganic carbon from pH and bicarbonate ion.
    Follows ZW01 Appendix B (6).
    """
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    H = 10.0**-pH
    return bicarbonate * (1 + H / K1 + K2 / H)


def dic_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants):
    """Dissolved inorganic carbon from CO2 fugacity and carbonate ion."""
    pH = pH_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants)
    return dic_from_pH_carbonate(pH, carbonate, totals, k_constants)


def dic_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants):
    """Dissolved inorganic carbon from CO2 fugacity and bicarbonate ion."""
    carbonate = carbonate_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants)
    return k_constants["CO2"] * fCO2 + bicarbonate + carbonate


def dic_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants):
    """Dissolved inorganic carbon from carbonate ion and carbonate ion."""
    pH = pH_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants)
    return dic_from_pH_carbonate(pH, carbonate, totals, k_constants)


def pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants):
    """Calculate pH from total alkalinity and DIC."""

    def cond(targets):
        pH = targets
        residuals = np.array(
            [residual.pH_from_alkalinity_dic(pH, alkalinity, dic, totals, k_constants)]
        )
        return np.any(np.abs(residuals) > 1e-9)

    def body(targets):
        pH = targets
        deltas = delta.pH_from_alkalinity_dic(pH, alkalinity, dic, totals, k_constants)
        deltas = np.where(deltas > 1, 1.0, deltas)
        deltas = np.where(deltas < -1, -1.0, deltas)
        return targets + deltas

    # First guess inspired by M13/OE15:
    pH_initial = initialise.from_dic(
        alkalinity,
        dic,
        totals["borate"],
        k_constants["carbonic_1"],
        k_constants["carbonic_2"],
        k_constants["borate"],
    )
    targets = pH_initial
    targets = lax.while_loop(cond, body, targets)
    return targets


# def pH_from_alkalinity_fCO2(alkalinity, fCO2, totals, k_constants):
#     """Calculate pH from total alkalinity and CO2 fugacity."""
#     # Slightly more convoluted than the others because initialise.fromCO2 takes CO2 as
#     # an input, while delta.pHfromTAfCO2 takes fCO2.
#     return _pHfromTAVX(
#         alkalinity,
#         fCO2,
#         totals,
#         k_constants,
#         lambda alkalinity, fCO2, TB, K1, K2, KB: initialise.fromCO2(
#             alkalinity, k_constants["CO2"] * fCO2, TB, K1, K2, KB
#         ),  # this just transforms initalise.fromCO2 to take fCO2 in place of CO2
#         delta.pHfromTAfCO2,
#     )


# def pH_from_alkalinity_carbonate(
#     alkalinity, carbonate, totals, k_constants
# ):
#     """Calculate pH from total alkalinity and carbonate ion molinity."""
#     return _pHfromTAVX(
#         alkalinity,
#         carbonate,
#         totals,
#         k_constants,
#         initialise.fromCO3,
#         delta.pHfromTACarb,
#     )


# def pH_from_alkalinity_bicarbonate(
#     alkalinity, bicarbonate, totals, k_constants
# ):
#     """Calculate pH from total alkalinity and bicarbonate ion molinity."""
#     return _pHfromTAVX(
#         alkalinity,
#         bicarbonate,
#         totals,
#         k_constants,
#         initialise.frombicarbonate,
#         delta.pHfromTAbicarbonate,
#     )


def pH_from_dic_fCO2(dic, fCO2, totals, k_constants):
    """Calculate pH from dissolved inorganic carbon and CO2 fugacity.

    This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the quadratic in
    H: fCO2*K0 = TC*H*H/(K1*H + H*H + K1*K2).
    If there is not a real root, then pH is returned as np.nan.

    Based on CalculatepHfromTCfCO2, version 02.02, 11-12-96, by Ernie Lewis.
    """
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    RR = K0 * fCO2 / dic
    Discr = (K1 * RR) ** 2 + 4 * (1 - RR) * K1 * K2 * RR
    F = (RR >= 1) | (Discr <= 0)
    if np.any(F):
        print("Some input fCO2 values are impossibly high given the input DIC;")
        print("returning np.nan.")
    H = np.where(F, np.nan, 0.5 * (K1 * RR + np.sqrt(Discr)) / (1 - RR))
    pH = -np.log10(H)
    return pH


def pH_from_dic_carbonate(dic, carbonate, totals, k_constants):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    This calculates pH from Carbonate and TC using K1, and K2 by solving the
    quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).

    Based on CalculatepHfromTCCarb, version 01.00, 06-12-2019, by Denis Pierrot.
    """
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    RR = 1 - dic / carbonate
    Discr = K1**2 - 4 * K1 * K2 * RR
    F = (carbonate >= dic) | (Discr <= 0)
    H = np.where(F, np.nan, (-K1 + np.sqrt(Discr)) / 2)
    return -np.log10(H)


def pH_from_dic_bicarbonate(dic, bicarbonate, totals, k_constants):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    Follows ZW01 Appendix B (12).
    """
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    a = bicarbonate / K1
    b = bicarbonate - dic
    c = bicarbonate * K2
    bsq_4ac = b**2 - 4 * a * c
    F = (bicarbonate >= dic) | (bsq_4ac <= 0)
    if np.any(F):
        print("Some input bicarbonate values are impossibly high given the input DIC;")
        print("returning np.nan.")
    H = np.where(F, np.nan, (-b + which_bicarbonate_root * np.sqrt(bsq_4ac)) / (2 * a))
    return -np.log10(H)


def pH_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants):
    """Calculate pH from CO2 fugacity and carbonate ion.

    This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving
    the equation in H: fCO2 * K0 * K1* K2 = Carb * H * H

    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis
    Pierrot.
    """
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    H = np.sqrt(K0 * K1 * K2 * fCO2 / carbonate)
    return -np.log10(H)


def pH_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants):
    """pH from CO2 fugacity and bicarbonate ion."""
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    H = K0 * K1 * fCO2 / bicarbonate
    return -np.log10(H)


def pH_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants):
    """pH from carbonate ion and carbonate ion."""
    H = k_constants["carbonic_2"] * bicarbonate / carbonate
    return -np.log10(H)


def fCO2_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants):
    """Calculate CO2 fugacity from carbonate ion and bicarbonate ion."""
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    return bicarbonate**2 * K2 / (carbonate * K1 * K0)


def fCO2_from_alkalinity_dic(alkalinity, dic, totals, k_constants):
    """Calculate CO2 fugacity from total alkalinity and dissolved inorganic carbon."""
    pH = pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants)
    return fCO2_from_dic_pH(dic, pH, totals, k_constants)


def fCO2_from_alkalinity_pH(alkalinity, pH, totals, k_constants):
    """Calculate CO2 fugacity from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants)
    return fCO2_from_dic_pH(dic, pH, totals, k_constants)


def fCO2_from_dic_pH(dic, pH, totals, k_constants):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.

    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.
    """
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    H = 10.0**-pH
    return dic * H**2 / (H**2 + K1 * H + K1 * K2) / K0


def fCO2_from_pH_carbonate(pH, carbonate, totals, k_constants):
    """Calculate CO2 fugacity from pH and carbonate ion.

    Based on CalculatefCO2frompHCarb, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    H = 10.0**-pH
    return carbonate * H**2 / (K0 * K1 * K2)


def fCO2_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants):
    """Calculate CO2 fugacity from pH and bicarbonate ion."""
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    H = 10.0**-pH
    return bicarbonate * H / (K0 * K1)


def carbonate_from_alkalinity_dic(alkalinity, dic, totals, k_constants):
    """Calculate carbonate ion from total alkalinity and dissolved inorganic carbon."""
    pH = pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants)
    return carbonate_from_dic_pH(dic, pH, totals, k_constants)


def carbonate_from_alkalinity_pH(alkalinity, pH, totals, k_constants):
    """Calculate carbonate ion from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants)
    return carbonate_from_dic_pH(dic, pH, totals, k_constants)


def _carbonate_from_dic_H(dic, H, totals, k_constants):
    """Calculate carbonate ion from dissolved inorganic carbon and [H+].

    Based on CalculateCarbfromdicpH, version 01.0, 06-12-2019, by Denis Pierrot.
    """
    K1, K2 = k_constants["carbonic_1"], k_constants["carbonic_2"]
    return dic * K1 * K2 / (H**2 + K1 * H + K1 * K2)


def carbonate_from_dic_pH(dic, pH, totals, k_constants):
    """Calculate carbonate ion from dissolved inorganic carbon and pH."""
    H = 10.0**-pH
    return _carbonate_from_dic_H(dic, H, totals, k_constants)


def carbonate_from_pH_fCO2(pH, fCO2, totals, k_constants):
    """Calculate carbonate ion from pH and CO2 fugacity."""
    dic = dic_from_pH_fCO2(pH, fCO2, totals, k_constants)
    return carbonate_from_dic_pH(dic, pH, totals, k_constants)


def carbonate_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants):
    """Calculate bicarbonate ion from pH and carbonate ion."""
    H = 10.0**-pH
    return k_constants["carbonic_2"] * bicarbonate / H


def carbonate_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants):
    """Calculate carbonate ion from CO2 fugacity and bicarbonate ion."""
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    K2 = k_constants["carbonic_2"]
    return bicarbonate**2 * K2 / (K0 * fCO2 * K1)


def _bicarbonate_from_dic_H(dic, H, totals, k_constants):
    """Calculate bicarbonate ion from dissolved inorganic carbon and [H+]."""
    K1, K2 = k_constants["carbonic_1"], k_constants["carbonic_2"]
    return dic * K1 * H / (H**2 + K1 * H + K1 * K2)


def bicarbonate_from_dic_pH(dic, pH, totals, k_constants):
    """Calculate bicarbonate ion from dissolved inorganic carbon and pH."""
    H = 10.0**-pH
    return _bicarbonate_from_dic_H(dic, H, totals, k_constants)


def bicarbonate_from_alkalinity_pH(alkalinity, pH, totals, k_constants):
    """Calculate carbonate ion from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants)
    return bicarbonate_from_dic_pH(dic, pH, totals, k_constants)


def bicarbonate_from_pH_fCO2(pH, fCO2, totals, k_constants):
    """Calculate bicarbonate ion from pH and CO2 fugacity."""
    K0 = k_constants["CO2"]
    K1 = k_constants["carbonic_1"]
    H = 10.0**-pH
    return K0 * K1 * fCO2 / H


def bicarbonate_from_pH_carbonate(pH, carbonate, totals, k_constants):
    """Calculate bicarbonate ion from pH and carbonate ion."""
    H = 10.0**-pH
    return carbonate * H / k_constants["carbonic_2"]


def bicarbonate_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants):
    """Bicarbonate ion from CO2 fugacity and carbonate ion."""
    pH = pH_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants)
    return bicarbonate_from_pH_carbonate(pH, carbonate, totals, k_constants)


def _CO2_from_dic_H(dic, H, totals, k_constants):
    """Calculate aqueous CO2 from dissolved inorganic carbon and [H+]."""
    K1, K2 = k_constants["carbonic_1"], k_constants["carbonic_2"]
    return dic * H**2 / (H**2 + K1 * H + K1 * K2)
