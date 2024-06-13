# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Equations and parameters for modelling marine organic matter."""

import jax
from jax import numpy as np
from .. import constants, properties, salts

# Bold values for fulvic/humic acids from Milne et al. (2001) Table 4.
# A "density" field must be added to these before they can be fully used.
nd_fulvic = {
    "b": 0.57,
    "Qmax_H1": 5.88,
    "logK_H1": 2.34,
    "m1": 0.38,
    "Qmax_H2": 1.86,
    "logK_H2": 8.60,
    "m2": 0.53,
    # Metals from Milne et al. (2003) Table 3
    "p1": 0.59,
    "p2": 0.70,
    "n_H1": 0.66,
    "n_H2": 0.76,
    "logK_Ca1": -2.13,
    "logK_Ca2": -3.0,
    "n_Ca1": 0.85,
    "n_Ca2": 0.80,
    "logK_Mg1": -2.1,
    "logK_Mg2": -2.4,
    "n_Mg1": 0.77,
    "n_Mg2": 0.59,
    "logK_Sr1": -2.5,
    "logK_Sr2": -4.6,
    "n_Sr1": 0.85,
    "n_Sr2": 0.70,
}
nd_humic = {
    "b": 0.49,
    "Qmax_H1": 3.15,
    "logK_H1": 2.93,
    "m1": 0.50,
    "Qmax_H2": 2.55,
    "logK_H2": 8.00,
    "m2": 0.26,
    # Metals from Milne et al. (2003) Table 3
    "p1": 0.62,
    "p2": 0.41,
    "n_H1": 0.81,
    "n_H2": 0.63,
    "logK_Ca1": -1.37,
    "logK_Ca2": -0.43,
    "n_Ca1": 0.78,
    "n_Ca2": 0.75,
    "logK_Mg1": -0.6,
    "logK_Mg2": 0.6,
    "n_Mg1": 0.77,
    "n_Mg2": 0.59,
    "logK_Sr1": -1.36,
    "logK_Sr2": -0.43,
    "n_Sr1": 0.78,
    "n_Sr2": 0.75,
}


def get_nd_params(
    nd_type="humic",
    temperature=25.0,
    salinity=35.0,
    pressure=0.0,
    latitude=0.0,
    longitude=0.0,
):
    """Get a set of NICA-Donnan model parameters to use with Aqualibrium functions.

    Parameters
    ----------
    nd_type : str, optional
        Which set of parameters to use, either 'humic' (default) or 'fulvic'.
    temperature : float, optional
        Temperature in °C (to calculate density), by default 25.
    salinity : float, optional
        Practical salinity (to calculate density), by default 35.
    pressure : float, optional
        In-water pressure in dbar (to calculate density), by default 0.
    latitude : float, optional
        Latitude in decimal °N (to calculate density), by default 0.
    longitude : float, optional
        Longitude in decimal °E (to calculate density), by default 0.
    """
    if nd_type == "humic":
        nd_params = nd_humic
    elif nd_type == "fulvic":
        nd_params = nd_fulvic
    else:
        print("nd_type '{}' not recognised, valid options are 'humic' or 'fulvic'.")
    nd_params["density"] = properties.get_density(
        salinity=salinity,
        temperature=temperature,
        pressure=pressure,
        latitude=latitude,
        longitude=longitude,
    )
    return nd_params


def nica(cH, chi, nd_params):
    """Calculate the NICA isotherm for the given parameters.

    This is the total amount of protons bound to DOM (Q_H) using the NICA equation,
    following Milne et al. (2001), eq. (1).

    Parameters
    ----------
    cH : float
        Proton concentration in the bulk solution in mol/l.
    chi : float
        Boltzmann factor.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Amount of bound protons in mol/kg-DOM.
    """
    cH_D = cH * chi  # proton concentration in the Donnan gel phase
    het1 = (10.0 ** nd_params["logK_H1"] * cH_D) ** nd_params["m1"]
    Q_H1 = nd_params["Qmax_H1"] * het1 / (1 + het1)
    het2 = (10.0 ** nd_params["logK_H2"] * cH_D) ** nd_params["m2"]
    Q_H2 = nd_params["Qmax_H2"] * het2 / (1 + het2)
    return Q_H1 + Q_H2


def nica_charge(cH, chi, nd_params):
    """Charge on DOM from the NICA equation.

    Parameters
    ----------
    cH : float
        Proton concentration in the bulk solution in mol/l.
    chi : float
        Boltzmann factor.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Charge on the NICA molecule in mol/kg-DOM.
    """
    return nica(cH, chi, nd_params) - nd_params["Qmax_H1"] - nd_params["Qmax_H2"]


def nica_metals(cH, cM, chi, nd_params):
    """Calculate the NICA isotherm for the given parameters, including competition from
    divalent metals.

    This is the total amount of protons bound to DOM (Q_H) using the NICA equation,
    following Milne et al. (2003), eq. (1).

    Parameters
    ----------
    cH : float
        Proton concentration in the bulk solution in mol/l.
    cM : dict
        Divalent metal ion concentrations in the bulk solution in mol/l.
    chi : float
        Boltzmann factor.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Amount of bound protons in mol/kg-DOM.
    """
    cH_D = cH * chi  # proton concentration in the Donnan gel phase
    # Calculate the first affinity distribution
    het1_H = (10.0 ** nd_params["logK_H1"] * cH_D) ** nd_params["n_H1"]
    het1_M = np.sum(
        np.array(
            [
                (10.0 ** nd_params["logK_{}1".format(m)] * c * chi)
                ** nd_params["n_{}1".format(m)]
                for m, c in cM.items()
            ]
        )
    )
    Q_H1 = (
        nd_params["Qmax_H1"]
        * het1_H
        * (het1_H + het1_M) ** nd_params["p1"]
        / ((het1_H + het1_M) * (1 + (het1_H + het1_M) ** nd_params["p1"]))
    )
    # Calculate the second affinity distribution
    het2_H = (10.0 ** nd_params["logK_H2"] * cH_D) ** nd_params["n_H2"]
    het2_M = np.sum(
        np.array(
            [
                (10.0 ** nd_params["logK_{}2".format(m)] * c * chi)
                ** nd_params["n_{}2".format(m)]
                for m, c in cM.items()
            ]
        )
    )
    Q_H2 = (
        nd_params["Qmax_H2"]
        * het2_H
        * (het2_H + het2_M) ** nd_params["p2"]
        / ((het2_H + het2_M) * (1 + (het2_H + het2_M) ** nd_params["p2"]))
    )
    return Q_H1 + Q_H2


def nica_specific_metal(metal, cH, cM, chi, nd_params):
    """Calculate the NICA isotherm for the given parameters, including competition from
    divalent metals, for a specific metal.

    This is the total amount of metals bound to DOM (Q_M) summed across all metals,
    using the NICA equation of Milne et al. (2003), eq. (1).

    Parameters
    ----------
    metal : str
        Which metal to calculate Q_M for.
    cH : float
        Proton concentration in the bulk solution in mol/l.
    cM : dict
        Divalent metal ion concentrations in the bulk solution in mol/l.
    chi : float
        Boltzmann factor.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Amount of the bound metals in mol/kg-DOM.
    """
    cH_D = cH * chi  # proton concentration in the Donnan gel phase
    # Calculate the first affinity distribution
    het1_H = (10.0 ** nd_params["logK_H1"] * cH_D) ** nd_params["n_H1"]
    het1_M = np.sum(
        np.array(
            [
                (10.0 ** nd_params["logK_{}1".format(m)] * c * chi)
                ** nd_params["n_{}1".format(m)]
                for m, c in cM.items()
            ]
        )
    )
    Q_H1 = (
        (nd_params["n_{}1".format(metal)] / nd_params["n_H1"])
        * nd_params["Qmax_H1"]
        * (10.0 ** nd_params["logK_{}1".format(metal)] * cM[metal] * chi)
        ** nd_params["n_{}1".format(metal)]
        * (het1_H + het1_M) ** nd_params["p1"]
        / ((het1_H + het1_M) * (1 + (het1_H + het1_M) ** nd_params["p1"]))
    )
    # Calculate the second affinity distribution
    het2_H = (10.0 ** nd_params["logK_H2"] * cH_D) ** nd_params["n_H2"]
    het2_M = np.sum(
        np.array(
            [
                (10.0 ** nd_params["logK_{}2".format(m)] * c * chi)
                ** nd_params["n_{}2".format(m)]
                for m, c in cM.items()
            ]
        )
    )
    Q_H2 = (
        (nd_params["n_{}2".format(metal)] / nd_params["n_H2"])
        * nd_params["Qmax_H2"]
        * (10.0 ** nd_params["logK_{}2".format(metal)] * cM[metal] * chi)
        ** nd_params["n_{}2".format(metal)]
        * (het2_H + het2_M) ** nd_params["p2"]
        / ((het2_H + het2_M) * (1 + (het2_H + het2_M) ** nd_params["p2"]))
    )
    return Q_H1 + Q_H2


def nica_metals_not_H(cH, cM, chi, nd_params):
    """Calculate the NICA isotherm for the given parameters, including competition from
    divalent metals.

    This is the total amount of metals bound to DOM (Q_M) summed across all metals,
    using the NICA equation of Milne et al. (2003), eq. (1).

    Parameters
    ----------
    cH : float
        Proton concentration in the bulk solution in mol/l.
    cM : dict
        Divalent metal ion concentrations in the bulk solution in mol/l.
    chi : float
        Boltzmann factor.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Amount of all bound metals in mol/kg-DOM.
    """
    cH_D = cH * chi  # proton concentration in the Donnan gel phase
    # Calculate the first affinity distribution
    het1_H = (10.0 ** nd_params["logK_H1"] * cH_D) ** nd_params["n_H1"]
    het1_M = np.sum(
        np.array(
            [
                (10.0 ** nd_params["logK_{}1".format(m)] * c * chi)
                ** nd_params["n_{}1".format(m)]
                for m, c in cM.items()
            ]
        )
    )
    Q_H1 = np.sum(
        np.array(
            [
                (nd_params["n_{}1".format(m)] / nd_params["n_H1"])
                * nd_params["Qmax_H1"]
                * (10.0 ** nd_params["logK_{}1".format(m)] * c * chi)
                ** nd_params["n_{}1".format(m)]
                * (het1_H + het1_M) ** nd_params["p1"]
                / ((het1_H + het1_M) * (1 + (het1_H + het1_M) ** nd_params["p1"]))
                for m, c in cM.items()
            ]
        )
    )
    # Calculate the second affinity distribution
    het2_H = (10.0 ** nd_params["logK_H2"] * cH_D) ** nd_params["n_H2"]
    het2_M = np.sum(
        np.array(
            [
                (10.0 ** nd_params["logK_{}2".format(m)] * c * chi)
                ** nd_params["n_{}2".format(m)]
                for m, c in cM.items()
            ]
        )
    )
    Q_H2 = np.sum(
        np.array(
            [
                (nd_params["n_{}2".format(m)] / nd_params["n_H2"])
                * nd_params["Qmax_H2"]
                * (10.0 ** nd_params["logK_{}2".format(m)] * c * chi)
                ** nd_params["n_{}2".format(m)]
                * (het2_H + het2_M) ** nd_params["p2"]
                / ((het2_H + het2_M) * (1 + (het2_H + het2_M) ** nd_params["p2"]))
                for m, c in cM.items()
            ]
        )
    )
    return Q_H1 + Q_H2


def nica_charge_metals(cH, cM, chi, nd_params):
    """Charge on DOM from the NICA equation, including competition from divalent metals.

    Parameters
    ----------
    cH : float
        Proton concentration in the bulk solution in mol/l.
    cM : dict
        Divalent metal ion concentrations in the bulk solution in mol/l.
    chi : float
        Boltzmann factor.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Charge on the NICA molecule in mol/kg-DOM.
    """
    return (
        nica_metals(cH, cM, chi, nd_params)
        + 2 * nica_metals_not_H(cH, cM, chi, nd_params)
        - nd_params["Qmax_H1"]
        - nd_params["Qmax_H2"]
    )


def donnan_volume(ionic_strength_vol, nd_params):
    """Donnan gel volume in l/kg-DOM, following Milne et al. (2001) eq. (2).

    Parameters
    ----------
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Donnan gel volume in l/kg-DOM.
    """
    logVD = nd_params["b"] * (1 - np.log10(ionic_strength_vol)) - 1
    return 10.0**logVD  # l/kg-DOM


def donnan_charge(chi, c_ions, z_ions, ionic_strength_vol, nd_params):
    """Charge balance of the Donnan gel.

    Parameters
    ----------
    chi : float
        Boltzmann factor.
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges on the ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Charge balance of the Donnan gel in mol/kg-DOM.
    """
    V_D = donnan_volume(ionic_strength_vol, nd_params)  # l/kg-DOM
    chi_vector = np.where(z_ions > 0, chi, 1.0)  # apply chi only to cations
    total_charge = np.sum(z_ions * chi_vector**z_ions * c_ions)  # mol/l
    return V_D * total_charge  # mol/kg-DOM


def charge_balance(log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params):
    """Overall charge balance to solve for log10(chi).  The first item of c_ions must
    be cH.

    Parameters
    ----------
    log10_chi : float
        Base-10 logarithm of the Boltzmann factor.
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Charge balance of the NICA molecule plus Donnan gel in mol/l.
    """
    chi = 10.0**log10_chi
    cH = c_ions[0]  # assumes first c_ions item is always cH!
    return nica_charge(cH, chi, nd_params) + donnan_charge(
        chi, c_ions, z_ions, ionic_strength_vol, nd_params
    )


def charge_balance_with_grad(log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params):
    """Overall charge balance and its gradient w.r.t. log10(chi).

    Parameters
    ----------
    log10_chi : float
        Base-10 logarithm of the Boltzmann factor.
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Charge balance of the NICA molecule plus Donnan gel in mol/l.
    float
        Gradient of the charge balance w.r.t. log10(chi).
    """
    return jax.value_and_grad(lambda *args: charge_balance(*args))(
        log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params
    )


def charge_balance_metals(log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params):
    """Overall charge balance to solve for log10(chi), including competition from
    divalent metals.  The first item of c_ions must be cH.

    Parameters
    ----------
    log10_chi : float
        Base-10 logarithm of the Boltzmann factor.
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Charge balance of the NICA molecule plus Donnan gel in mol/l.
    """
    chi = 10.0**log10_chi
    cH = c_ions[0]  # assumes first c_ions item is always cH!
    cM = {"Ca": c_ions[3], "Mg": c_ions[4], "Sr": c_ions[5]}
    return nica_charge_metals(cH, cM, chi, nd_params) + donnan_charge(
        chi, c_ions, z_ions, ionic_strength_vol, nd_params
    )


def charge_balance_metals_with_grad(
    log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params
):
    """Overall charge balance and its gradient w.r.t. log10(chi), including competition
    from divalent metals.

    Parameters
    ----------
    log10_chi : float
        Base-10 logarithm of the Boltzmann factor.
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.

    Returns
    -------
    float
        Charge balance of the NICA molecule plus Donnan gel in mol/l.
    float
        Gradient of the charge balance w.r.t. log10(chi).
    """
    return jax.value_and_grad(lambda *args: charge_balance_metals(*args))(
        log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params
    )


@jax.jit
def _solve_log10_chi(c_ions, z_ions, ionic_strength_vol, nd_params, log10_chi_guess):
    """Solve charge balance for log10(chi) with Newton-Raphson, jitted function.

    Parameters
    ----------
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.
    log10_chi_guess : float, optional
        First guess for log10_chi, by default 3.0.

    Returns
    -------
    float
        Base-10 logarithm of the Boltzmann factor.
    """

    def cond(targets):
        log10_chi = targets
        residuals = np.array(
            [charge_balance(log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params)]
        )
        return np.any(np.abs(residuals) > 1e-12)

    def body(targets):
        log10_chi = targets
        cb_value, cb_grad = charge_balance_with_grad(
            log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params
        )
        deltas = -(cb_value / cb_grad)
        deltas = np.where(deltas > 1, 1.0, deltas)
        deltas = np.where(deltas < -1, -1.0, deltas)
        return targets + deltas

    # Set tolerances and limits and prepare for loop
    targets = log10_chi_guess
    targets = jax.lax.while_loop(cond, body, targets)
    return np.squeeze(targets)


def solve_log10_chi(c_ions, z_ions, ionic_strength_vol, nd_params, log10_chi_guess=3.0):
    """Solve charge balance for log10(chi) with Newton-Raphson.

    Parameters
    ----------
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.
    log10_chi_guess : float, optional
        First guess for log10_chi, by default 3.0.

    Returns
    -------
    float
        Base-10 logarithm of the Boltzmann factor.
    """
    return _solve_log10_chi(
        c_ions, z_ions, ionic_strength_vol, nd_params, log10_chi_guess
    )


@jax.jit
def _solve_log10_chi_metals(
    c_ions, z_ions, ionic_strength_vol, nd_params, log10_chi_guess
):
    """Solve charge balance for log10(chi) with Newton-Raphson, included competition
    from divalent metals.  Jitted function.

    Parameters
    ----------
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.
    log10_chi_guess : float, optional
        First guess for log10_chi, by default 3.0.

    Returns
    -------
    float
        Base-10 logarithm of the Boltzmann factor.
    """

    def cond(targets):
        log10_chi = targets
        residuals = np.array(
            [
                charge_balance_metals(
                    log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params
                )
            ]
        )
        return np.any(np.abs(residuals) > 1e-12)

    def body(targets):
        log10_chi = targets
        cb_value, cb_grad = charge_balance_metals_with_grad(
            log10_chi, c_ions, z_ions, ionic_strength_vol, nd_params
        )
        deltas = -(cb_value / cb_grad)
        deltas = np.where(deltas > 1, 1.0, deltas)
        deltas = np.where(deltas < -1, -1.0, deltas)
        return targets + deltas

    # Set tolerances and limits and run loop
    targets = log10_chi_guess
    targets = jax.lax.while_loop(cond, body, targets)
    return np.squeeze(targets)


def solve_log10_chi_metals(
    c_ions, z_ions, ionic_strength_vol, nd_params, log10_chi_guess=3.0
):
    """Solve charge balance for log10(chi) with Newton-Raphson, including competition
    from divalent metals.

    Parameters
    ----------
    c_ions : array_like
        Concentrations of ions in mol/l, in the same order as z_ions.
    z_ions : array_like
        Charges of ions, in the same order as c_ions.
    ionic_strength_vol : float
        Ionic strength of the bulk solution in mol/l.
    nd_params : dict
        NICA model parameters.
    log10_chi_guess : float, optional
        First guess for log10_chi, by default 3.0.

    Returns
    -------
    float
        Base-10 logarithm of the Boltzmann factor.
    """
    return _solve_log10_chi_metals(
        c_ions, z_ions, ionic_strength_vol, nd_params, log10_chi_guess
    )


def psi_to_chi(psi, temperature):
    """Convert Boltzmann factor psi to chi.

    Parameters
    ----------
    psi : float
        Boltzmann factor as psi.
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        Boltzmann factor as chi.
    """
    return np.exp(
        -psi / (constants.k_boltzmann * (temperature + constants.temperature_zero))
    )


def chi_to_psi(chi, temperature):
    """Convert Boltzmann factor chi to psi.

    Parameters
    ----------
    chi : float
        Boltzmann factor as chi.
    temperature : float
        Temperature in °C.

    Returns
    -------
    float
        Boltzmann factor as psi.
    """
    return (
        -np.log(chi)
        * constants.k_boltzmann
        * (temperature + constants.temperature_zero)
    )


def get_ions(sw, totals, density):
    """Generate c_ions and z_ions arrays.

    The order is
        0: H (+1),
        1: all other other +1s,
        2: all -1s,
        3: Ca (+2),
        4: Mg (+2),
        5: Sr (+2),
        6: all -2s,
        7: all +3s.

    Parameters
    ----------
    sw : dict
        Substance contents of pH-sensitive ions, output from `aq.speciation`.
    rc : dict
        Substance contents of pH-insensitive ions, output from
        `aq.get_reference_composition`.
    density : float
        Seawater density in kg/l.

    Returns
    -------
    c_ions : array_like
        Concentrations of ions with various charges in mol/l.  The first item is H+.
    z_ions : array_like
        Charges on the ions, in the same order as c_ions.
    """
    # Assemble arrays
    c_ions = np.array(
        [
            sw["Hfree"],
            totals["sodium"] + totals["potassium"] + sw["NH4"],
            sw["OH"]
            + totals["chloride"]
            + totals["bromide"]
            + sw["F"]
            + sw["HSO4"]
            + sw["HCO3"]
            + sw["BOH4"]
            + sw["H3SiO4"]
            + sw["H2PO4"]
            + sw["HS"],
            totals["calcium"],
            totals["magnesium"],
            totals["strontium"],
            sw["SO4"] + sw["CO3"] + sw["HPO4"],
            sw["PO4"],
        ]
    )
    c_ions = c_ions * density  # convert from mol/kg-sw to mol/l
    z_ions = np.array([1, 1, -1, 2, 2, 2, -2, 3])  # order must match c_ions!
    # Enforce charge balance with +1 ions (i.e. Na and Cl, because they are probably
    # the most abundant)
    cb = np.sum(c_ions * z_ions)  # charge balance
    # Where charge balance is negative, add extra "Na"
    iCat1 = 1  # index of item in c_ions containing +1 cations (except H+)
    cCat1 = c_ions[iCat1]
    cCat1 = np.where(cb < 0, cCat1 - cb, cCat1)
    # Where charge balance is positive, add extra "Cl"
    iAni1 = 2  # index of item in c_ions containing -1 anions
    cAni1 = c_ions[iAni1]
    cAni1 = np.where(cb > 0, cAni1 + cb, cAni1)
    # Reconstruct c_ions with new columns
    c_ions = np.concatenate((np.array([c_ions[0], cCat1, cAni1]), c_ions[3:]))
    return c_ions, z_ions


def get_ionic_strength(c_ions, z_ions):
    """Calculate ionic strength from concentrations and charges.  Output units match
    whatever the input units are for c_ions.

    Parameters
    ----------
    c_ions : array_like
        Concentrations of the ions in e.g. mol/l.
    z_ions : array_like
        Charges of the ions, in the same order as the columns of c_ions.

    Returns
    -------
    float
        Ionic strength, in the same units as c_ions.
    """
    return 0.5 * np.sum(c_ions * z_ions**2)
