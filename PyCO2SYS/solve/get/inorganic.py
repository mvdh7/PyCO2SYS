# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate one new carbonate system variable from various input pairs."""

from jax import numpy as np, lax
from ... import salts
from .. import delta, initialise, residual, speciate


def alkalinity_from_dic_pH(dic, pH, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and pH."""
    sw = speciate.inorganic(dic, pH, totals, k_constants)
    return sw["alkalinity"] * 1e6


def alkalinity_from_pH_fCO2(pH, fCO2, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and CO2 fugacity."""
    dic = dic_from_pH_fCO2(pH, fCO2, k_CO2, k_H2CO3, k_HCO3)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants)


def alkalinity_from_pH_carbonate(pH, carbonate, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and carbonate ion."""
    dic = dic_from_pH_CO3(pH, CO3, k_H2CO3, k_HCO3)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants)


def alkalinity_from_pH_bicarbonate(pH, bicarbonate, totals, k_constants):
    """Calculate total alkalinity from dissolved inorganic carbon and bicarbonate ion."""
    dic = dic_from_pH_HCO3(pH, HCO3, k_H2CO3, k_HCO3)
    return alkalinity_from_dic_pH(dic, pH, totals, k_constants)


def alkalinity_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants):
    """Total alkalinity from CO2 fugacity and carbonate ion."""
    pH = pH_from_fCO2_CO3(fCO2, CO3, k_CO2, k_H2CO3, kHCO3)
    return alkalinity_from_pH_fCO2(pH, fCO2, totals, k_constants)


def alkalinity_from_fCO2_bicarbonate(fCO2, bicarbonate, totals, k_constants):
    """Total alkalinity from CO2 fugacity and bicarbonate ion."""
    carbonate = CO3_from_fCO2_HCO3(fCO2, HCO3, k_CO2, k_H2CO3, k_HCO3)
    return alkalinity_from_fCO2_carbonate(fCO2, carbonate, totals, k_constants)


def alkalinity_from_carbonate_bicarbonate(carbonate, bicarbonate, totals, k_constants):
    """Total alkalinity from carbonate ion and carbonate ion."""
    pH = pH_from_CO3_HCO3(CO3, HCO3, k_HCO3)
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


def dic_from_pH_fCO2(pH, fCO2, k_CO2, k_H2CO3, k_HCO3):
    """Calculate dissolved inorganic carbon from pH and CO2 fugacity.
    Based on CalculateTCfrompHfCO2, version 01.02, 12-13-96, by Ernie Lewis.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    fCO2 : float
        Seawater fCO2 in µatm.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    K0, K1, K2 = k_CO2, k_H2CO3, k_HCO3
    H = 10.0**-pH
    return K0 * fCO2 * (H**2 + K1 * H + K1 * K2) / H**2


def dic_from_pH_CO3(pH, CO3, k_H2CO3, k_HCO3):
    """Calculate dissolved inorganic carbon from pH and carbonate ion.
    Follows ZW01 Appendix B (7).

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    K1, K2 = k_H2CO3, k_HCO3
    H = 10.0**-pH
    return CO3 * (1 + H / K2 + H**2 / (K1 * K2))


def dic_from_pH_HCO3(pH, HCO3, k_H2CO3, k_HCO3):
    """Calculate dissolved inorganic carbon from pH and bicarbonate ion.
    Follows ZW01 Appendix B (6).

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    K1, K2 = k_H2CO3, k_HCO3
    H = 10.0**-pH
    return HCO3 * (1 + H / K1 + K2 / H)


def dic_from_fCO2_CO3(fCO2, CO3, k_CO2, k_H2CO3, kHCO3):
    """Dissolved inorganic carbon from CO2 fugacity and carbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    pH = pH_from_fCO2_CO3(fCO2, CO3, k_CO2, k_H2CO3, kHCO3)
    return dic_from_pH_CO3(pH, CO3, k_H2CO3, k_HCO3)


def dic_from_fCO2_HCO3(fCO2, HCO3, k_CO2, k_H2CO3, k_HCO3):
    """Dissolved inorganic carbon from CO2 fugacity and bicarbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    CO3 = CO3_from_fCO2_HCO3(fCO2, HCO3, k_CO2, k_H2CO3, k_HCO3)
    return k_CO2 * fCO2 + HCO3 + CO3


def dic_from_CO3_HCO3(CO3, HCO3, k_H2CO3, k_HCO3):
    """Dissolved inorganic carbon from carbonate ion and carbonate ion.

    Parameters
    ----------
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    pH = pH_from_CO3_HCO3(CO3, HCO3, k_HCO3)
    return dic_from_pH_CO3(pH, CO3, k_H2CO3, k_HCO3)


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


def pH_from_dic_fCO2(dic, fCO2, k_CO2, k_H2CO3, k_CO3):
    """Calculate pH from dissolved inorganic carbon and CO2 fugacity.

    This calculates pH from TC and fCO2 using K0, K1, and K2 by solving the quadratic in
    H: fCO2*K0 = TC*H*H/(K1*H + H*H + K1*K2).
    If there is not a real root, then pH is returned as np.nan.

    Based on CalculatepHfromTCfCO2, version 02.02, 11-12-96, by Ernie Lewis.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    fCO2 : float
        Seawater fCO2 in µatm.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0, K1, K2 = k_CO2, k_H2CO3, k_HCO3
    RR = K0 * fCO2 / dic
    Discr = (K1 * RR) ** 2 + 4 * (1 - RR) * K1 * K2 * RR
    F = (RR >= 1) | (Discr <= 0)
    if np.any(F):
        print("Some input fCO2 values are impossibly high given the input DIC;")
        print("returning np.nan.")
    H = np.where(F, np.nan, 0.5 * (K1 * RR + np.sqrt(Discr)) / (1 - RR))
    pH = -np.log10(H)
    return pH


def pH_from_dic_CO3(dic, CO3, k_H2CO3, k_HCO3):
    """Calculate pH from dissolved inorganic carbon and carbonate ion.

    This calculates pH from Carbonate and TC using K1, and K2 by solving the
    quadratic in H: TC * K1 * K2= Carb * (H * H + K1 * H +  K1 * K2).

    Based on CalculatepHfromTCCarb, version 01.00, 06-12-2019, by Denis Pierrot.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K1, K2 = k_H2CO3, k_HCO3
    RR = 1 - dic / carbonate
    Discr = K1**2 - 4 * K1 * K2 * RR
    F = (carbonate >= dic) | (Discr <= 0)
    H = np.where(F, np.nan, (-K1 + np.sqrt(Discr)) / 2)
    return -np.log10(H)


def pH_from_dic_HCO3(dic, HCO3, k_H2CO3, k_HCO3):
    """Calculate pH from dissolved inorganic carbon and bicarbonate ion.

    Follows ZW01 Appendix B (12).

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    HCO3 : float
        Biarbonate ion content in µmol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K1, K2 = k_H2CO3, k_HCO3
    a = 1e-6 * bicarbonate / K1
    b = 1e-6 * (bicarbonate - dic)
    c = 1e-6 * bicarbonate * K2
    bsq_4ac = b**2 - 4 * a * c
    F = (bicarbonate >= dic) | (bsq_4ac <= 0)
    if np.any(F):
        print("Some input bicarbonate values are impossibly high given the input DIC;")
        print("returning np.nan.")
    H = np.where(F, np.nan, (-b + which_bicarbonate_root * np.sqrt(bsq_4ac)) / (2 * a))
    return -np.log10(H)


def pH_from_fCO2_CO3(fCO2, CO3, k_CO2, k_H2CO3, kHCO3):
    """Calculate pH from CO2 fugacity and carbonate ion.

    This calculates pH from Carbonate and fCO2 using K0, K1, and K2 by solving
    the equation in H: fCO2 * K0 * K1* K2 = Carb * H * H

    Based on CalculatepHfromfCO2Carb, version 01.00, 06-12-2019, by Denis Pierrot.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0, K1, K2 = k_CO2, k_H2CO3, k_HCO3
    H = np.sqrt(K0 * K1 * K2 * fCO2 / carbonate)
    return -np.log10(H)


def pH_from_fCO2_HCO3(fCO2, HCO3, k_CO2, k_H2CO3):
    """pH from CO2 fugacity and bicarbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3 : float
        First carbonic acid dissociation constant.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0, K1 = k_CO2, k_H2CO3
    H = K0 * K1 * fCO2 / bicarbonate
    return -np.log10(H)


def pH_from_CO3_HCO3(CO3, HCO3, k_HCO3):
    """pH from carbonate ion and carbonate ion.

    Parameters
    ----------
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_HCO3 : float
        Second carbonic acid dissociation constant.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    H = k_HCO3 * bicarbonate / carbonate
    return -np.log10(H)


def fCO2_from_CO3_HCO3(CO3, HCO3, k_CO2, k_H2CO3, k_HCO3):
    """Calculate CO2 fugacity from carbonate ion and bicarbonate ion.

    Parameters
    ----------
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0, K1, K2 = k_CO2, k_H2CO3, k_HCO3
    H = 1e-6 * bicarbonate**2 * K2 / (carbonate * K1 * K0)
    return -np.log10(H)


def fCO2_from_alkalinity_dic(alkalinity, dic, totals, k_constants):
    """Calculate CO2 fugacity from total alkalinity and dissolved inorganic carbon."""
    pH = pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants)
    return fCO2_from_dic_pH(dic, pH, k_CO2, k_H2CO3, k_HCO3)


def fCO2_from_alkalinity_pH(alkalinity, pH, totals, k_constants):
    """Calculate CO2 fugacity from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants)
    return fCO2_from_dic_pH(dic, pH, k_CO2, k_H2CO3, k_HCO3)


def fCO2_from_dic_pH(dic, pH, k_CO2, k_H2CO3, k_HCO3):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.

    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    K0, K1, K2 = k_CO2, k_H2CO3, k_HCO3
    H = 10.0**-pH
    return dic * H**2 / (H**2 + K1 * H + K1 * K2) / K0


def fCO2_from_pH_CO3(pH, CO3, k_CO2, k_H2CO3, k_HCO3):
    """Calculate CO2 fugacity from pH and carbonate ion.

    Based on CalculatefCO2frompHCarb, version 01.0, 06-12-2019, by Denis Pierrot.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    K0, K1, K2 = k_CO2, k_H2CO3, k_HCO3
    H = 10.0**-pH
    return carbonate * H**2 / (K0 * K1 * K2)


def fCO2_from_pH_HCO3(pH, HCO3, k_CO2, k_H2CO3):
    """Calculate CO2 fugacity from pH and bicarbonate ion.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3 : float
        First carbonic acid dissociation constant.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    K0, K1 = k_CO2, k_H2CO3
    H = 10.0**-pH
    return HCO3 * H / (K0 * K1)


def carbonate_from_alkalinity_dic(alkalinity, dic, totals, k_constants):
    """Calculate carbonate ion from total alkalinity and dissolved inorganic carbon."""
    pH = pH_from_alkalinity_dic(alkalinity, dic, totals, k_constants)
    return CO3_from_dic_pH(dic, H, k_H2CO3, k_HCO3)


def carbonate_from_alkalinity_pH(alkalinity, pH, totals, k_constants):
    """Calculate carbonate ion from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants)
    return CO3_from_dic_pH(dic, H, k_H2CO3, k_HCO3)


def CO3_from_dic_H(dic, H, k_H2CO3, k_HCO3):
    """Calculate carbonate ion from dissolved inorganic carbon and [H+].

    Based on CalculateCarbfromdicpH, version 01.0, 06-12-2019, by Denis Pierrot.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] in mol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    K1, K2 = k_H2CO3, k_HCO3
    return dic * K1 * K2 / (H**2 + K1 * H + K1 * K2)


def CO3_from_dic_pH(dic, pH, k_H2CO3, k_HCO3):
    """Calculate carbonate ion from dissolved inorganic carbon and pH.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    H = 10.0**-pH
    return CO3_from_dic_H(dic, H, k_H2CO3, k_HCO3)


def CO3_from_pH_fCO2(pH, fCO2, k_CO2, k_H2CO3, k_HCO3):
    """Calculate carbonate ion from pH and CO2 fugacity.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    fCO2 : float
        Seawater fCO2 in µatm.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    dic = dic_from_pH_fCO2(pH, fCO2, k_CO2, k_H2CO3, k_HCO3)
    return CO3_from_dic_pH(dic, H, k_H2CO3, k_HCO3)


def CO3_from_pH_HCO3(pH, HCO3, k_HCO3):
    """Calculate bicarbonate ion from pH and carbonate ion.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_HCO3 : float
        Second carbonic acid dissociation constant.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    H = 10.0**-pH
    return k_HCO3 * bicarbonate / H


def CO3_from_fCO2_HCO3(fCO2, HCO3, k_CO2, k_H2CO3, k_HCO3):
    """Calculate carbonate ion from CO2 fugacity and bicarbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    K0, K1, K2 = k_CO2, k_H2CO3, k_HCO3
    return HCO3**2 * K2 / (K0 * fCO2 * K1)


def HCO3_from_dic_H(dic, H, k_H2CO3, k_HCO3):
    """Calculate bicarbonate ion from dissolved inorganic carbon and [H+].

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] content in mol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    K1, K2 = k_H2CO3, k_HCO3
    return dic * K1 * H / (H**2 + K1 * H + K1 * K2)


def HCO3_from_dic_pH(dic, pH, k_H2CO3, k_HCO3):
    """Calculate bicarbonate ion from dissolved inorganic carbon and pH.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    H = 10.0**-pH
    return HCO3_from_dic_H(dic, H, k_H2CO3, k_HCO3)


def bicarbonate_from_alkalinity_pH(alkalinity, pH, totals, k_constants):
    """Calculate carbonate ion from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(alkalinity, pH, totals, k_constants)
    return HCO3_from_dic_pH(dic, pH, k_H2CO3, k_HCO3)


def HCO3_from_pH_fCO2(pH, fCO2, k_CO2, k_H2CO3):
    """Calculate bicarbonate ion from pH and CO2 fugacity.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    fCO2 : float
        Seawater fCO2 in µatm.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3 : float
        First carbonic acid dissociation constant.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    K0, K1 = k_CO2, k_H2CO3
    H = 10.0**-pH
    return K0 * K1 * fCO2 / H


def HCO3_from_pH_CO3(pH, CO3, k_HCO3):
    """Calculate bicarbonate ion from pH and carbonate ion.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    k_HCO3 : float
        Second carbonic acid dissociation constant.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    H = 10.0**-pH
    return carbonate * H / k_HCO3


def HCO3_from_fCO2_CO3(fCO2, CO3, k_CO2, k_H2CO3, kHCO3):
    """Bicarbonate ion from CO2 fugacity and carbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    k_CO2 : float
        Solubility constant for CO2.
    k_H2CO3, k_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    pH = pH_from_fCO2_CO3(fCO2, CO3, k_CO2, k_H2CO3, kHCO3)
    return HCO3_from_pH_CO3(pH, CO3, k_HCO3)


def CO2_from_dic_H(dic, H, k_H2CO3, k_HCO3):
    """Calculate aqueous CO2 from dissolved inorganic carbon and [H+].

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] in mol/kg-sw.
    k_H2CO3, kHCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Aqueous CO2 content in µmol/kg-sw.
    """
    K1, K2 = k_H2CO3, k_HCO3
    return dic * H**2 / (H**2 + K1 * H + K1 * K2)
