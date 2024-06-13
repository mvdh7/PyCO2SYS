# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Evaluate residuals for alkalinity-pH solvers."""

import jax
from jax import numpy as np
from .. import salts
from . import get, speciate


def pH_from_alkalinity_dic(pH, alkalinity, dic, totals, k_constants):
    """Calculate residual alkalinity from pH and DIC for solver
    `inorganic.pH_from_alkalinity_dic()`.
    """
    return (
        get.inorganic.alkalinity_from_dic_pH(dic, pH, totals, k_constants) - alkalinity
    )


def pH_from_alkalinity_dic_zlp(pH, alkalinity, dic, totals, k_constants, pzlp):
    """Calculate residual alkalinity from pH and DIC for solver
    `inorganic_zlp.H_from_alkalinity_dic()`.
    """
    return (
        get.inorganic_zlp.alkalinity_from_dic_pH(dic, pH, totals, k_constants, pzlp)
        - alkalinity
    )


def pH_from_alkalinity_dic_dom(
    pH__log10_chi, alkalinity, dic, totals, k_constants, nd_params
):
    """Calculate residual alkalinity and charge balance from pH and DIC for solver
    `inorganic_dom.pH_from_alkalinity_dic`.
    """
    pH, log10_chi = pH__log10_chi
    sw = speciate.inorganic_dom_chi(dic, pH, totals, k_constants, nd_params, log10_chi)
    c_ions, z_ions = dom.get_ions(sw, totals, nd_params["density"])
    ionic_strength = dom.get_ionic_strength(c_ions, z_ions)
    residual_alkalinity = sw["alkalinity"] - alkalinity * 1e-6
    residual_charge_balance = dom.charge_balance(
        log10_chi, c_ions, z_ions, ionic_strength, nd_params
    )
    return np.array([residual_alkalinity, residual_charge_balance])


def pH_from_alkalinity_dic_dom_jac(
    pH__log10_chi, alkalinity, dic, totals, k_constants, nd_params
):
    """Calculate the Jacobian of pH_from_alkalinity_dic_dom()."""
    return jax.jacfwd(pH_from_alkalinity_dic_dom)(
        pH__log10_chi, alkalinity, dic, totals, k_constants, nd_params
    )


def pH_from_alkalinity_dic_dom_metals(
    pH__log10_chi, alkalinity, dic, totals, k_constants, nd_params
):
    """Calculate residual alkalinity and charge balance from pH and DIC for solver
    `inorganic_dom_metals.pH_from_alkalinity_dic`.
    """
    pH, log10_chi = pH__log10_chi
    sw = speciate.inorganic_dom_metals_chi(
        dic, pH, totals, k_constants, nd_params, log10_chi
    )
    c_ions, z_ions = dom.get_ions(sw, totals, nd_params["density"])
    ionic_strength = dom.get_ionic_strength(c_ions, z_ions)
    residual_alkalinity = sw["alkalinity"] - alkalinity * 1e-6
    residual_charge_balance = dom.charge_balance_metals(
        log10_chi, c_ions, z_ions, ionic_strength, nd_params
    )
    return np.array([residual_alkalinity, residual_charge_balance])


def pH_from_alkalinity_dic_dom_metals_jac(
    pH__log10_chi, alkalinity, dic, totals, k_constants, nd_params
):
    """Calculate the Jacobian of pH_from_alkalinity_dic_dom_metals()."""
    return jax.jacfwd(pH_from_alkalinity_dic_dom_metals)(
        pH__log10_chi, alkalinity, dic, totals, k_constants, nd_params
    )
