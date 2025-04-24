# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for alkalinity-pH solvers."""

from . import inorganic, speciate


def pH_from_alkalinity_dic(
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
):
    """Calculate residual alkalinity from pH and DIC for solver
    `inorganic.pH_from_alkalinity_dic()`.
    """
    H = 10**-pH
    H_free = speciate.get_H_free(H, opt_to_free)
    OH = speciate.get_OH(H, pk_H2O)
    HCO3 = speciate.get_HCO3(dic, H, pk_H2CO3, pk_HCO3)
    CO3 = speciate.get_CO3(dic, H, pk_H2CO3, pk_HCO3)
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
    return (
        speciate.sum_alkalinity(
            H_free,
            OH,
            HCO3,
            CO3,
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
        )
        - alkalinity
    )


def pH_from_alkalinity_fCO2(
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
):
    """Calculate residual alkalinity from pH and DIC for solver
    `inorganic.pH_from_alkalinity_fCO2()`.
    """
    dic = inorganic.dic_from_pH_fCO2(pH, fCO2, pk_CO2, pk_H2CO3, pk_HCO3)
    H = 10**-pH
    H_free = speciate.get_H_free(H, opt_to_free)
    OH = speciate.get_OH(H, pk_H2O)
    HCO3 = inorganic.HCO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3)
    CO3 = inorganic.CO3_from_dic_pH(dic, pH, pk_H2CO3, pk_HCO3)
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
    return (
        speciate.sum_alkalinity(
            H_free,
            OH,
            HCO3,
            CO3,
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
        )
        - alkalinity
    )


def pH_from_alkalinity_CO3(
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
):
    """Calculate residual alkalinity from pH and CO3 for solver
    `inorganic.pH_from_alkalinity_CO3()`.
    """
    H = 10**-pH
    H_free = speciate.get_H_free(H, opt_to_free)
    OH = speciate.get_OH(H, pk_H2O)
    HCO3 = inorganic.HCO3_from_pH_CO3(pH, CO3, pk_HCO3)
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
    return (
        speciate.sum_alkalinity(
            H_free,
            OH,
            HCO3,
            CO3,
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
        )
        - alkalinity
    )


def pH_from_alkalinity_HCO3(
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
):
    """Calculate residual alkalinity from pH and HCO3 for solver
    `inorganic.pH_from_alkalinity_HCO3()`.
    """
    H = 10**-pH
    H_free = speciate.get_H_free(H, opt_to_free)
    OH = speciate.get_OH(H, pk_H2O)
    CO3 = inorganic.CO3_from_pH_HCO3(pH, HCO3, pk_HCO3)
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
    return (
        speciate.sum_alkalinity(
            H_free,
            OH,
            HCO3,
            CO3,
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
        )
        - alkalinity
    )
