# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for alkalinity-pH solvers."""

import jax
from . import get, residual


def pH_from_alkalinity_dic_with_grad(pH, alkalinity, dic, totals, k_constants):
    return jax.value_and_grad(residual.pH_from_alkalinity_dic)(
        pH, alkalinity, dic, totals, k_constants
    )


def pH_from_alkalinity_dic(pH, alkalinity, dic, totals, k_constants):
    """Calculate delta-pH from pH and DIC for solver
    `inorganic.pH_from_alkalinity_dic`.
    """
    residual, residual_grad = pH_from_alkalinity_dic_with_grad(
        pH, alkalinity, dic, totals, k_constants
    )
    return -(residual / residual_grad)


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
