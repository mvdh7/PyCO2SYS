# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2022  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Equations and parameters for modelling marine organic matter."""

from autograd import numpy as np
from .. import salts

# Bold values for fulvic/humic acids from Milne et al. (2001) Table 4
nd_fulvic = {
    "b": 0.57,
    "Qmax_H1": 5.88,
    "logK_H1": 2.34,
    "m1": 0.38,
    "Qmax_H2": 1.86,
    "logK_H2": 8.60,
    "m2": 0.53,
}
nd_humic = {
    "b": 0.49,
    "Qmax_H1": 3.15,
    "logK_H1": 2.93,
    "m1": 0.50,
    "Qmax_H2": 2.55,
    "logK_H2": 8.00,
    "m2": 0.26,
}


def nica(cH, chi, nd_params):
    """Calculate the NICA isotherm for the given parameters.

    This is the total amount of protons bound to DOM (Q_H) using the NICA equation,
    following Milne et al. (2001), eq. (1).

    Parameters
    ----------
    chi : array_like
        Boltzmann factor.
    cH : array_like
        Proton concentration in the bulk solution in mol/L.
    k_nica : dict
        NICA model parameters.

    Returns
    -------
    array_like
        Amount of bound protons in mol/kg-DOM.
    """
    cH_D = cH * chi  # proton concentration in the Donnan gel phase
    het1 = (10.0 ** nd_params["logK_H1"] * cH_D) ** nd_params["m1"]
    Q_H1 = nd_params["Qmax_H1"] * het1 / (1 + het1)
    het2 = (10.0 ** nd_params["logK_H2"] * cH_D) ** nd_params["m2"]
    Q_H2 = nd_params["Qmax_H2"] * het2 / (1 + het2)
    return Q_H1 + Q_H2


def nica_charge(cH, chi, nd_params):
    """Charge on DOM from the NICA equation."""
    return nica(cH, chi, nd_params) - nd_params["QmaxH1"] - nd_params["QmaxH2"]


def donnan_volume(ionic_strength, nd_params):
    """Donnan gel volume in L/kg-DOM, following Milne et al. (2001) eq. (2)."""
    logVD = nd_params["b"] * (1 - np.log10(ionic_strength)) - 1
    return 10.0**logVD


def donnan_charge(chi, c_ions, z_ions, ionic_strength, nd_params):
    """Charge balance of Donnan gel."""
    V_D = donnan_volume(ionic_strength, nd_params)
    total_charge = np.sum(z_ions * np.vstack(chi) ** z_ions * c_ions, axis=1)
    return V_D * total_charge


def charge_balance(log10_chi, c_ions, z_ions, ionic_strength, nd_params):
    """Overall charge balance to solve for log10(chi)."""
    chi = 10.0**log10_chi
    cH = c_ions[:, 0]  # assumes first ion column is always H!
    return nica_charge(cH, chi, nd_params) - donnan_charge(
        chi, c_ions, z_ions, ionic_strength, nd_params
    )


def solve_chi(
    c_ions, z_ions, ionic_strength, nd_params, log10_chi_guess=None, niter=100
):
    """Solve charge balance for log10(chi) with Newton-Raphson."""
    if log10_chi_guess is None:
        log10_chi = np.full_like(c_ions[:, 0], 0.0)  # first guess if needed
    else:
        log10_chi = log10_chi_guess * 1.0
    max_step = 1.0
    for i in range(niter):
        log10_chi_delta = -(
            charge_balance(log10_chi, c_ions, z_ions, ionic_strength, nd_params)
            / charge_balance_grad(log10_chi, c_ions, z_ions, ionic_strength, nd_params)
        )
        log10_chi_delta = np.where(
            np.abs(log10_chi_delta) > max_step,
            max_step * np.sign(log10_chi_delta),
            log10_chi_delta,
        )
        log10_chi = log10_chi + log10_chi_delta
    return log10_chi


def get_ions(sw, salinity):
    """Generate c_ions and z_ions arrays."""
    rc = salts.get_reference_composition(salinity)
    density = 1.01  # kg/L --------------------- NEEDS UPDATING TO A FUNCTION! USE GSW?
    c_ions = (
        np.array(
            [
                sw["Hfree"].ravel(),
                sw["OH"].ravel(),
                rc["Na"].ravel(),
                rc["Mg"].ravel(),
                rc["Ca"].ravel(),  # could be from sw?
                rc["K"].ravel(),
                rc["Sr"].ravel(),
                np.zeros_like(rc["Cl"].ravel()),
                sw["SO4"].ravel(),
                sw["HSO4"].ravel(),
                rc["Br"].ravel(),
                sw["F"].ravel(),
                sw["CO3"].ravel(),
                sw["HCO3"].ravel(),
                sw["BOH4"].ravel(),
                sw["H3SiO4"].ravel(),
                sw["H2PO4"].ravel(),
                sw["HPO4"].ravel(),
                sw["PO4"].ravel(),
                sw["NH4"].ravel(),
                sw["HS"].ravel(),
            ]
        ).transpose()
        * density
    )
    z_ions = np.array(
        [1, -1, 1, 2, 2, 1, 2, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -3, 1, -1]
    )  # order must match c_ions!
    # Enforce charge balance with Cl (because it is the most abundant ion)
    c_ions[:, 7] = np.sum(c_ions * z_ions, axis=1)
    return c_ions, z_ions


def get_dom_bound_protons(
    total_dom, sw, salinity, nd_params, log10_chi_guess=None, niter=100
):
    """Interface function to calculate amount of DOM-bound protons."""
    c_ions, z_ions = get_ions(sw, salinity)
    ionic_strength = 0.5 * np.sum(c_ions * z_ions**2, axis=1)
    log10_chi = solve_chi(
        c_ions,
        z_ions,
        ionic_strength,
        nd_params,
        log10_chi_guess=log10_chi_guess,
        niter=niter,
    )
    QH = nica(c_ions[:, 0], chi, nd_params)  # mol/kg-DOM
    return QH * total_dom * 1e-6  # assuming total_dom is in mg-DOM/kg-sw
