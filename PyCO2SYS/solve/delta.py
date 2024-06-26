# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for alkalinity-pH solvers."""

import jax
from jax import numpy as np
from . import get, residual


def egrad(g):
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped


def pH_from_alkalinity_dic_with_grad(pH, alkalinity, dic, totals, k_constants):
    return jax.value_and_grad(residual.pH_from_alkalinity_dic)(
        pH, alkalinity, dic, totals, k_constants
    )


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
    """Calculate delta-pH from pH and DIC for solver
    `inorganic.pH_from_alkalinity_dic`.
    """
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


def pH_from_alkalinity_dic_zlp_with_grad(
    pH, alkalinity, dic, totals, k_constants, pzlp
):
    return jax.value_and_grad(residual.pH_from_alkalinity_dic_zlp)(
        pH, alkalinity, dic, totals, k_constants, pzlp
    )


def pH_from_alkalinity_dic_zlp(pH, alkalinity, dic, totals, k_constants, pzlp):
    """Calculate delta-pH from pH and DIC for solver
    `inorganic_zlp.pH_from_alkalinity_dic`.
    """
    residual, residual_grad = pH_from_alkalinity_dic_zlp_with_grad(
        pH, alkalinity, dic, totals, k_constants, pzlp
    )
    return -(residual / residual_grad)
