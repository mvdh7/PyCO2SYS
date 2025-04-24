# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for alkalinity-pH solvers."""

import jax

from ..meta import egrad
from . import residual


@jax.jit
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
    """Calculate delta-pH for solver ``inorganic.pH_from_alkalinity_dic``."""
    args = (
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
    )
    alkalinity_residual = residual.pH_from_alkalinity_dic(*args)
    alkalinity_residual_grad = egrad(residual.pH_from_alkalinity_dic)(*args)
    return -(alkalinity_residual / alkalinity_residual_grad)


@jax.jit
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
    """Calculate delta-pH for solver ``inorganic.pH_from_alkalinity_fCO2``."""
    args = (
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
    )
    alkalinity_residual = residual.pH_from_alkalinity_fCO2(*args)
    alkalinity_residual_grad = egrad(residual.pH_from_alkalinity_fCO2)(*args)
    return -(alkalinity_residual / alkalinity_residual_grad)


@jax.jit
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
    """Calculate delta-pH for solver ``inorganic.pH_from_alkalinity_CO3``."""
    args = (
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
    )
    alkalinity_residual = residual.pH_from_alkalinity_CO3(*args)
    alkalinity_residual_grad = egrad(residual.pH_from_alkalinity_CO3)(*args)
    return -(alkalinity_residual / alkalinity_residual_grad)


@jax.jit
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
    """Calculate delta-pH for solver ``inorganic.pH_from_alkalinity_HCO3``."""
    args = (
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
    )
    alkalinity_residual = residual.pH_from_alkalinity_HCO3(*args)
    alkalinity_residual_grad = egrad(residual.pH_from_alkalinity_HCO3)(*args)
    return -(alkalinity_residual / alkalinity_residual_grad)
