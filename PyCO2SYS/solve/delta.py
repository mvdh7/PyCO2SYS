# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for alkalinity-pH solvers."""

import jax
from jax import numpy as np
from . import get, residual
from .. import egrad


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
    opt_to_free,
    k_H2O,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
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
        opt_to_free,
        k_H2O,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
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
    opt_to_free,
    k_H2O,
    k_CO2,
    k_H2CO3,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
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
        opt_to_free,
        k_H2O,
        k_CO2,
        k_H2CO3,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
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
    opt_to_free,
    k_H2O,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
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
        opt_to_free,
        k_H2O,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
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
    opt_to_free,
    k_H2O,
    k_HCO3,
    k_BOH3,
    k_H3PO4,
    k_H2PO4,
    k_HPO4,
    k_Si,
    k_NH3,
    k_H2S,
    k_HSO4_free,
    k_HF_free,
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
        opt_to_free,
        k_H2O,
        k_HCO3,
        k_BOH3,
        k_H3PO4,
        k_H2PO4,
        k_HPO4,
        k_Si,
        k_NH3,
        k_H2S,
        k_HSO4_free,
        k_HF_free,
    )
    alkalinity_residual = residual.pH_from_alkalinity_HCO3(*args)
    alkalinity_residual_grad = egrad(residual.pH_from_alkalinity_HCO3)(*args)
    return -(alkalinity_residual / alkalinity_residual_grad)
