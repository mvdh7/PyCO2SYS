# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate one new carbonate system variable from various input pairs."""

import warnings

from jax import numpy as np

from . import delta, initialise, speciate


def alkalinity_from_dic_pH(
    dic,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    pk_H2O,
    pk_H2CO3,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate total alkalinity from dissolved inorganic carbon and pH."""
    H = 10.0**-pH
    H_free = speciate.get_H_free(H, opt_to_free)
    OH = speciate.get_OH(H, pk_H2O)
    HCO3 = HCO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3)
    CO3 = CO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3)
    BOH4 = speciate.get_BOH4(total_borate, H, pk_BOH3)
    HPO4 = speciate.get_HPO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    PO4 = speciate.get_PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    H3PO4 = speciate.get_H3PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    H3SiO4 = speciate.get_H3SiO4(total_silicate, H, pk_Si)
    NH3 = speciate.get_NH3(total_ammonia, H, pk_NH3)
    HS = speciate.get_HS(total_sulfide, H, pk_H2S)
    HSO4 = speciate.get_HSO4(total_sulfate, H_free, pk_HSO4_free)
    HF = speciate.get_HF(total_fluoride, H_free, pk_HF_free)
    HNO2 = speciate.get_HNO2(total_nitrite, H, pk_HNO2)
    return speciate.sum_alkalinity(
        H_free, OH, HCO3, CO3, BOH4, HPO4, PO4, H3PO4, H3SiO4, NH3, HS, HSO4, HF, HNO2
    )


def alkalinity_from_pH_fCO2(
    pH,
    fCO2,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    pk_CO2,
    pk_H2O,
    pk_H2CO3,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate total alkalinity from dissolved inorganic carbon and CO2 fugacity."""
    dic = dic_from_pH_fCO2(pH, fCO2, pk_CO2, pk_H2CO3, pk_HCO3)
    return alkalinity_from_dic_pH(
        dic,
        pH,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        total_nitrite,
        pk_H2O,
        pk_H2CO3,
        pk_HCO3,
        pk_BOH3,
        pk_H3PO4,
        pk_H2PO4,
        pk_HPO4,
        pk_Si,
        pk_NH3,
        pk_H2S,
        pk_HSO4_free,
        pk_HF_free,
        pk_HNO2,
    )


def dic_from_alkalinity_pH_speciated(
    alkalinity,
    pH,
    H_free,
    OH,
    BOH4,
    HPO4,
    PO4,
    H3PO4,
    H3SiO4,
    NH3,
    HS,
    HSO4,
    HF,
    HNO2,
    pk_H2CO3,
    pk_HCO3,
):
    """Calculate dissolved inorganic carbon from total alkalinity and pH.
    Based on CalculateTCfromTApH, version 02.03, 10-10-97, by Ernie Lewis.

    Parameters
    ----------
    alkalinity : float
        Total alkalinity in µmol/kg-sw.
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    H_free : float
        Hydrogen ion content on the free scale in µmol/kg-sw.
    OH : float
        Hydroxide ion content in µmol/kg-sw.
    BOH4 : float
        B(OH)4 content in µmol/kg-sw.
    HPO4 : float
        Hydrogen phosphate content in µmol/kg-sw.
    PO4 : float
        Phosphate content in µmol/kg-sw.
    H3PO4 : float
        Trihydrogen phosphate content in µmol/kg-sw.
    H3SiO4 : float
        H3SiO4 content in µmol/kg-sw.
    NH3 : float
        Ammonia content in µmol/kg-sw.
    HS : float
        Bisulfide content in µmol/kg-sw.
    HSO4 : float
        Bisulfate content in µmol/kg-sw.
    HF : float
        HF content in µmol/kg-sw.
    HNO2 : float
        Nitrous acid content in µmol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    alkalinity_with_zero_dic = speciate.sum_alkalinity(
        H_free, OH, 0, 0, BOH4, HPO4, PO4, H3PO4, H3SiO4, NH3, HS, HSO4, HF, HNO2
    )
    F = alkalinity_with_zero_dic > alkalinity
    if np.any(F):
        warnings.warn(
            "Some input pH values are impossibly high given the input alkalinity;"
            + " returning `np.nan` rather than negative DIC."
        )
    alkalinity_carbonate = np.where(F, np.nan, alkalinity - alkalinity_with_zero_dic)
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    H = 10**-pH
    dic = alkalinity_carbonate * (H**2 + K1 * H + K1 * K2) / (K1 * H + 2 * K1 * K2)
    return dic


def dic_from_alkalinity_pH(
    alkalinity,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    pk_H2O,
    pk_H2CO3,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    H = 10**-pH
    H_free = speciate.get_H_free(H, opt_to_free)
    OH = speciate.get_OH(H, pk_H2O)
    BOH4 = speciate.get_BOH4(total_borate, H, pk_BOH3)
    HPO4 = speciate.get_HPO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    PO4 = speciate.get_PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    H3PO4 = speciate.get_H3PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    H3SiO4 = speciate.get_H3SiO4(total_silicate, H, pk_Si)
    NH3 = speciate.get_NH3(total_ammonia, H, pk_NH3)
    HS = speciate.get_HS(total_sulfide, H, pk_H2S)
    HSO4 = speciate.get_HSO4(total_sulfate, H_free, pk_HSO4_free)
    HF = speciate.get_HF(total_fluoride, H_free, pk_HF_free)
    HNO2 = speciate.get_HNO2(total_nitrite, H, pk_HNO2)
    return dic_from_alkalinity_pH_speciated(
        alkalinity,
        pH,
        H_free,
        OH,
        BOH4,
        HPO4,
        PO4,
        H3PO4,
        H3SiO4,
        NH3,
        HS,
        HSO4,
        HF,
        HNO2,
        pk_H2CO3,
        pk_HCO3,
    )


def dic_from_pH_fCO2(pH, fCO2, pk_CO2, pk_H2CO3, pk_HCO3):
    """Calculate dissolved inorganic carbon from pH and CO2 fugacity.
    Based on CalculateTCfrompHfCO2, version 01.02, 12-13-96, by Ernie Lewis.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    fCO2 : float
        Seawater fCO2 in µatm.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    H = 10**-pH
    return K0 * fCO2 * (H**2 + K1 * H + K1 * K2) / H**2


def dic_from_pH_CO3(pH, CO3, pk_H2CO3, pk_HCO3):
    """Calculate dissolved inorganic carbon from pH and carbonate ion.
    Follows ZW01 Appendix B (7).

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    H = 10**-pH
    return CO3 * (1 + H / K2 + H**2 / (K1 * K2))


def dic_from_pH_HCO3(pH, HCO3, pk_H2CO3, pk_HCO3):
    """Calculate dissolved inorganic carbon from pH and bicarbonate ion.
    Follows ZW01 Appendix B (6).

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    H = 10**-pH
    return HCO3 * (1 + H / K1 + K2 / H)


def dic_from_fCO2_CO3(fCO2, CO3, pk_CO2, pk_H2CO3, pk_HCO3):
    """Dissolved inorganic carbon from CO2 fugacity and carbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    pH = pH_from_fCO2_CO3(fCO2, CO3, pk_CO2, pk_H2CO3, pk_HCO3)
    return dic_from_pH_CO3(pH, CO3, pk_H2CO3, pk_HCO3)


def dic_from_fCO2_HCO3(fCO2, HCO3, pk_CO2, pk_H2CO3, pk_HCO3):
    """Dissolved inorganic carbon from CO2 fugacity and bicarbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    CO3 = CO3_from_fCO2_HCO3(fCO2, HCO3, pk_CO2, pk_H2CO3, pk_HCO3)
    return pk_CO2 * fCO2 + HCO3 + CO3


def dic_from_CO3_HCO3(CO3, HCO3, pk_H2CO3, pk_HCO3):
    """Dissolved inorganic carbon from carbonate ion and carbonate ion.

    Parameters
    ----------
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        DIC in µmol/kg-sw.
    """
    pH = pH_from_CO3_HCO3(CO3, HCO3, pk_HCO3)
    return dic_from_pH_CO3(pH, CO3, pk_H2CO3, pk_HCO3)


def pH_from_alkalinity_dic(
    alkalinity,
    dic,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    opt_to_free,
    pk_H2O,
    pk_H2CO3,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate pH from total alkalinity and DIC using a Newton-Raphson iterative
    method.  Based on the CalculatepHfromTATC function, version 04.01, Oct 96, by Ernie
    Lewis.
    """
    # First guess inspired by M13/OE15, added v1.3.0:
    pH = initialise.from_dic(alkalinity, dic, total_borate, pk_H2CO3, pk_HCO3, pk_BOH3)
    pH_tolerance = 1e-8
    pH_delta = 1.0 + pH_tolerance
    while np.any(np.abs(pH_delta) >= pH_tolerance):
        pH_done = (
            np.abs(pH_delta) < pH_tolerance
        )  # check which ones don't need updating
        pH_delta = delta.pH_from_alkalinity_dic(
            pH,
            alkalinity,
            dic,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            total_nitrite,
            opt_to_free,
            pk_H2O,
            pk_H2CO3,
            pk_HCO3,
            pk_BOH3,
            pk_H3PO4,
            pk_H2PO4,
            pk_HPO4,
            pk_Si,
            pk_NH3,
            pk_H2S,
            pk_HSO4_free,
            pk_HF_free,
            pk_HNO2,
        )  # the pH jump
        # To keep the jump from being too big:
        # This is the default PyCO2SYS way - jump by 1 instead if `pH_delta` > 1
        pH_delta = np.where(np.abs(pH_delta) > 1.0, np.sign(pH_delta), pH_delta)
        pH = np.where(pH_done, pH, pH + pH_delta)  # only update rows that need it
    return pH


def pH_from_alkalinity_fCO2(
    alkalinity,
    fCO2,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    opt_to_free,
    pk_H2O,
    pk_CO2,
    pk_H2CO3,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate pH from total alkalinity and DIC using a Newton-Raphson iterative
    method.  Based on the CalculatepHfromTATC function, version 04.01, Oct 96, by Ernie
    Lewis.
    """
    # First guess inspired by M13/OE15, added v1.3.0:
    pH = initialise.from_fCO2(
        alkalinity, fCO2, total_borate, pk_CO2, pk_H2CO3, pk_HCO3, pk_BOH3
    )
    pH_tolerance = 1e-8
    pH_delta = 1.0 + pH_tolerance
    while np.any(np.abs(pH_delta) >= pH_tolerance):
        pH_done = (
            np.abs(pH_delta) < pH_tolerance
        )  # check which ones don't need updating
        pH_delta = delta.pH_from_alkalinity_fCO2(
            pH,
            alkalinity,
            fCO2,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            total_nitrite,
            opt_to_free,
            pk_H2O,
            pk_CO2,
            pk_H2CO3,
            pk_HCO3,
            pk_BOH3,
            pk_H3PO4,
            pk_H2PO4,
            pk_HPO4,
            pk_Si,
            pk_NH3,
            pk_H2S,
            pk_HSO4_free,
            pk_HF_free,
            pk_HNO2,
        )  # the pH jump
        # To keep the jump from being too big:
        # This is the default PyCO2SYS way - jump by 1 instead if `pH_delta` > 1
        pH_delta = np.where(np.abs(pH_delta) > 1.0, np.sign(pH_delta), pH_delta)
        pH = np.where(pH_done, pH, pH + pH_delta)  # only update rows that need it
    return pH


def pH_from_alkalinity_CO3(
    alkalinity,
    CO3,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    opt_to_free,
    pk_H2O,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate pH from total alkalinity and CO3 using a Newton-Raphson iterative
    method.  Based on the CalculatepHfromTATC function, version 04.01, Oct 96, by Ernie
    Lewis.
    """
    # First guess inspired by M13/OE15, added v1.3.0:
    pH = initialise.from_CO3(alkalinity, CO3, total_borate, pk_HCO3, pk_BOH3)
    pH_tolerance = 1e-8
    pH_delta = 1.0 + pH_tolerance
    while np.any(np.abs(pH_delta) >= pH_tolerance):
        pH_done = (
            np.abs(pH_delta) < pH_tolerance
        )  # check which ones don't need updating
        pH_delta = delta.pH_from_alkalinity_CO3(
            pH,
            alkalinity,
            CO3,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            total_nitrite,
            opt_to_free,
            pk_H2O,
            pk_HCO3,
            pk_BOH3,
            pk_H3PO4,
            pk_H2PO4,
            pk_HPO4,
            pk_Si,
            pk_NH3,
            pk_H2S,
            pk_HSO4_free,
            pk_HF_free,
            pk_HNO2,
        )  # the pH jump
        # To keep the jump from being too big:
        # This is the default PyCO2SYS way - jump by 1 instead if `pH_delta` > 1
        pH_delta = np.where(np.abs(pH_delta) > 1.0, np.sign(pH_delta), pH_delta)
        pH = np.where(pH_done, pH, pH + pH_delta)  # only update rows that need it
    return pH


def pH_from_alkalinity_HCO3(
    alkalinity,
    HCO3,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    opt_to_free,
    pk_H2O,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate pH from total alkalinity and HCO3 using a Newton-Raphson iterative
    method.  Based on the CalculatepHfromTATC function, version 04.01, Oct 96, by Ernie
    Lewis.
    """
    # First guess inspired by M13/OE15, added v1.3.0:
    pH = initialise.from_HCO3(alkalinity, HCO3, total_borate, pk_HCO3, pk_BOH3)
    pH_tolerance = 1e-8
    pH_delta = 1.0 + pH_tolerance
    while np.any(np.abs(pH_delta) >= pH_tolerance):
        pH_done = (
            np.abs(pH_delta) < pH_tolerance
        )  # check which ones don't need updating
        pH_delta = delta.pH_from_alkalinity_HCO3(
            pH,
            alkalinity,
            HCO3,
            total_borate,
            total_phosphate,
            total_silicate,
            total_ammonia,
            total_sulfide,
            total_sulfate,
            total_fluoride,
            total_nitrite,
            opt_to_free,
            pk_H2O,
            pk_HCO3,
            pk_BOH3,
            pk_H3PO4,
            pk_H2PO4,
            pk_HPO4,
            pk_Si,
            pk_NH3,
            pk_H2S,
            pk_HSO4_free,
            pk_HF_free,
            pk_HNO2,
        )  # the pH jump
        # To keep the jump from being too big:
        # This is the default PyCO2SYS way - jump by 1 instead if `pH_delta` > 1
        pH_delta = np.where(np.abs(pH_delta) > 1.0, np.sign(pH_delta), pH_delta)
        pH = np.where(pH_done, pH, pH + pH_delta)  # only update rows that need it
    return pH


def pH_from_dic_fCO2(dic, fCO2, pk_CO2, pk_H2CO3, pk_HCO3):
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
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    RR = K0 * fCO2 / dic
    Discr = (K1 * RR) ** 2 + 4 * (1 - RR) * K1 * K2 * RR
    F = (RR >= 1) | (Discr <= 0)
    if np.any(F):
        warnings.warn(
            "Some input fCO2 values are impossibly high given the input DIC;"
            + " returning np.nan."
        )
    H = np.where(F, np.nan, 0.5 * (K1 * RR + np.sqrt(Discr)) / (1 - RR))
    pH = -np.log10(H)
    return pH


def pH_from_dic_CO3(dic, CO3, pk_H2CO3, pk_HCO3):
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
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    RR = 1 - dic / CO3
    Discr = K1**2 - 4 * K1 * K2 * RR
    F = (CO3 >= dic) | (Discr <= 0)
    H = np.where(F, np.nan, (-K1 + np.sqrt(Discr)) / 2)
    return -np.log10(H)


def pH_from_dic_HCO3_hi(dic, HCO3, pk_H2CO3, pk_HCO3):
    """Calculate pH from dissolved inorganic carbon and bicarbonate ion, taking the
    high-pH root.  Used when opt_HCO3_root = 2.

    Follows ZW01 Appendix B (12).

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    HCO3 : float
        Biarbonate ion content in µmol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    a = 1e-6 * HCO3 / K1
    b = 1e-6 * (HCO3 - dic)
    c = 1e-6 * HCO3 * K2
    bsq_4ac = b**2 - 4 * a * c
    F = (HCO3 >= dic) | (bsq_4ac <= 0)
    if np.any(F):
        warnings.warn(
            "Some input HCO3 values are impossibly high given the input DIC;"
            + " returning np.nan."
        )
    H = np.where(F, np.nan, (-b - np.sqrt(bsq_4ac)) / (2 * a))
    return -np.log10(H)


def pH_from_dic_HCO3_lo(dic, HCO3, pk_H2CO3, pk_HCO3):
    """Calculate pH from dissolved inorganic carbon and bicarbonate ion, taking the
    low-pH root.  Used when opt_HCO3_root = 1.

    Follows ZW01 Appendix B (12).

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    HCO3 : float
        Biarbonate ion content in µmol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.
    HCO3_root : int
        Which root to take: -1 for the high-pH, +1 for the low-pH.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    a = 1e-6 * HCO3 / K1
    b = 1e-6 * (HCO3 - dic)
    c = 1e-6 * HCO3 * K2
    bsq_4ac = b**2 - 4 * a * c
    F = (HCO3 >= dic) | (bsq_4ac <= 0)
    if np.any(F):
        warnings.warn(
            "Some input HCO3 values are impossibly high given the input DIC;"
            + " returning np.nan."
        )
    H = np.where(F, np.nan, (-b + np.sqrt(bsq_4ac)) / (2 * a))
    return -np.log10(H)


def pH_from_fCO2_CO3(fCO2, CO3, pk_CO2, pk_H2CO3, pk_HCO3):
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
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    H = np.sqrt(K0 * K1 * K2 * fCO2 / CO3)
    return -np.log10(H)


def pH_from_fCO2_HCO3(fCO2, HCO3, pk_CO2, pk_H2CO3):
    """pH from CO2 fugacity and bicarbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3 : float
        First carbonic acid dissociation constant.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    H = K0 * K1 * fCO2 / HCO3
    return -np.log10(H)


def pH_from_CO3_HCO3(CO3, HCO3, pk_HCO3):
    """pH from carbonate ion and carbonate ion.

    Parameters
    ----------
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_HCO3 : float
        Second carbonic acid dissociation constant.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    H = 10**-pk_HCO3 * HCO3 / CO3
    return -np.log10(H)


def fCO2_from_CO3_HCO3(CO3, HCO3, pk_CO2, pk_H2CO3, pk_HCO3):
    """Calculate CO2 fugacity from carbonate ion and bicarbonate ion.

    Parameters
    ----------
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater pH on the scale indicated by opt_pH_scale.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    fCO2 = HCO3**2 * K2 / (CO3 * K1 * K0)
    return fCO2


def fCO2_from_alkalinity_pH(
    alkalinity,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    pk_CO2,
    pk_H2O,
    pk_H2CO3,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate CO2 fugacity from total alkalinity and pH."""
    H = 10**-pH
    H_free = speciate.get_H_free(H, opt_to_free)
    OH = speciate.get_OH(H, pk_H2O)
    BOH4 = speciate.get_BOH4(total_borate, H, pk_BOH3)
    HPO4 = speciate.get_HPO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    PO4 = speciate.get_PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    H3PO4 = speciate.get_H3PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4)
    H3SiO4 = speciate.get_H3SiO4(total_silicate, H, pk_Si)
    NH3 = speciate.get_NH3(total_ammonia, H, pk_NH3)
    HS = speciate.get_HS(total_sulfide, H, pk_H2S)
    HSO4 = speciate.get_HSO4(total_sulfate, H_free, pk_HSO4_free)
    HF = speciate.get_HF(total_fluoride, H_free, pk_HF_free)
    HNO2 = speciate.get_HNO2(total_nitrite, H, pk_HNO2)
    dic = dic_from_alkalinity_pH_speciated(
        alkalinity,
        pH,
        H_free,
        OH,
        BOH4,
        HPO4,
        PO4,
        H3PO4,
        H3SiO4,
        NH3,
        HS,
        HSO4,
        HF,
        HNO2,
        pk_H2CO3,
        pk_HCO3,
    )
    return fCO2_from_dic_pH(dic, pH, pk_CO2, pk_H2CO3, pk_HCO3)


def fCO2_from_dic_pH(dic, pH, pk_CO2, pk_H2CO3, pk_HCO3):
    """Calculate CO2 fugacity from dissolved inorganic carbon and pH.

    Based on CalculatefCO2fromTCpH, version 02.02, 12-13-96, by Ernie Lewis.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    H = 10**-pH
    return dic * H**2 / (H**2 + K1 * H + K1 * K2) / K0


def fCO2_from_pH_CO3(pH, CO3, pk_CO2, pk_H2CO3, pk_HCO3):
    """Calculate CO2 fugacity from pH and carbonate ion.

    Based on CalculatefCO2frompHCarb, version 01.0, 06-12-2019, by Denis Pierrot.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    H = 10**-pH
    return CO3 * H**2 / (K0 * K1 * K2)


def fCO2_from_pH_HCO3(pH, HCO3, pk_CO2, pk_H2CO3):
    """Calculate CO2 fugacity from pH and bicarbonate ion.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3 : float
        First carbonic acid dissociation constant.

    Returns
    -------
    float
        Seawater fCO2 in µatm.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    H = 10**-pH
    return HCO3 * H / (K0 * K1)


def CO3_from_alkalinity_pH(
    alkalinity,
    pH,
    opt_to_free,
    total_borate,
    total_phosphate,
    total_silicate,
    total_ammonia,
    total_sulfide,
    total_sulfate,
    total_fluoride,
    total_nitrite,
    pk_H2O,
    pk_H2CO3,
    pk_HCO3,
    pk_BOH3,
    pk_H3PO4,
    pk_H2PO4,
    pk_HPO4,
    pk_Si,
    pk_NH3,
    pk_H2S,
    pk_HSO4_free,
    pk_HF_free,
    pk_HNO2,
):
    """Calculate carbonate ion from total alkalinity and pH."""
    dic = dic_from_alkalinity_pH(
        alkalinity,
        pH,
        opt_to_free,
        total_borate,
        total_phosphate,
        total_silicate,
        total_ammonia,
        total_sulfide,
        total_sulfate,
        total_fluoride,
        total_nitrite,
        pk_H2O,
        pk_H2CO3,
        pk_HCO3,
        pk_BOH3,
        pk_H3PO4,
        pk_H2PO4,
        pk_HPO4,
        pk_Si,
        pk_NH3,
        pk_H2S,
        pk_HSO4_free,
        pk_HF_free,
        pk_HNO2,
    )
    return CO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3)


def CO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3):
    """Calculate carbonate ion from dissolved inorganic carbon and pH.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    H = 10**-pH
    return speciate.get_CO3(dic, H, pk_H2CO3, pk_HCO3)


def CO3_from_pH_fCO2(pH, fCO2, pk_CO2, pk_H2CO3, pk_HCO3):
    """Calculate carbonate ion from pH and CO2 fugacity.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    fCO2 : float
        Seawater fCO2 in µatm.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    dic = dic_from_pH_fCO2(pH, fCO2, pk_CO2, pk_H2CO3, pk_HCO3)
    return CO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3)


def CO3_from_pH_HCO3(pH, HCO3, pk_HCO3):
    """Calculate bicarbonate ion from pH and carbonate ion.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_HCO3 : float
        Second carbonic acid dissociation constant.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    H = 10**-pH
    return 10**-pk_HCO3 * HCO3 / H


def CO3_from_fCO2_HCO3(fCO2, HCO3, pk_CO2, pk_H2CO3, pk_HCO3):
    """Calculate carbonate ion from CO2 fugacity and bicarbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    return HCO3**2 * K2 / (K0 * fCO2 * K1)


def HCO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3):
    """Calculate bicarbonate ion from dissolved inorganic carbon and pH.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    H = 10**-pH
    return speciate.get_HCO3(dic, H, pk_H2CO3, pk_HCO3)


def HCO3_from_pH_fCO2(pH, fCO2, pk_CO2, pk_H2CO3):
    """Calculate bicarbonate ion from pH and CO2 fugacity.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    fCO2 : float
        Seawater fCO2 in µatm.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3 : float
        First carbonic acid dissociation constant.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    K0 = 10**-pk_CO2
    K1 = 10**-pk_H2CO3
    H = 10**-pH
    return K0 * K1 * fCO2 / H


def HCO3_from_pH_CO3(pH, CO3, pk_HCO3):
    """Calculate bicarbonate ion from pH and carbonate ion.

    Parameters
    ----------
    pH : float
        Seawater pH on the scale indicated by opt_pH_scale.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    pk_HCO3 : float
        Second carbonic acid dissociation constant.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    H = 10**-pH
    return CO3 * H / 10**-pk_HCO3


def HCO3_from_fCO2_CO3(fCO2, CO3, pk_CO2, pk_H2CO3, pk_HCO3):
    """Bicarbonate ion from CO2 fugacity and carbonate ion.

    Parameters
    ----------
    fCO2 : float
        Seawater fCO2 in µatm.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
    pk_CO2 : float
        Solubility constant for CO2.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    pH = pH_from_fCO2_CO3(fCO2, CO3, pk_CO2, pk_H2CO3, pk_HCO3)
    return HCO3_from_pH_CO3(pH, CO3, pk_HCO3)


def CO2_from_dic_H(dic, H, pk_H2CO3, pk_HCO3):
    """Calculate aqueous CO2 from dissolved inorganic carbon and [H+].

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] in mol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Aqueous CO2 content in µmol/kg-sw.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    return dic * H**2 / (H**2 + K1 * H + K1 * K2)
