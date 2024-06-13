# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate chemical speciation."""

from jax import numpy as np
from . import get


def phosphoric(h_scale, totals, k_constants):
    """Calculate the components of total phosphate."""
    tPO4 = totals["phosphate"]
    KP1 = k_constants["phosphoric_1"]
    KP2 = k_constants["phosphoric_2"]
    KP3 = k_constants["phosphoric_3"]
    denom = h_scale**3 + KP1 * h_scale**2 + KP1 * KP2 * h_scale + KP1 * KP2 * KP3
    return {
        "PO4": tPO4 * KP1 * KP2 * KP3 / denom,
        "HPO4": tPO4 * KP1 * KP2 * h_scale / denom,
        "H2PO4": tPO4 * KP1 * h_scale**2 / denom,
        "H3PO4": tPO4 * h_scale**3 / denom,
    }


def inorganic(dic, pH, totals, k_constants):
    """Calculate the full inorganic chemical speciation of seawater given DIC and pH.
    Based on CalculateAlkParts by Ernie Lewis.
    """
    h_scale = 10.0**-pH  # on the pH scale declared by the user
    sw = {}
    # Carbonate
    sw["HCO3"] = (
        get.inorganic._bicarbonate_from_dic_H(dic, h_scale, totals, k_constants) * 1e-6
    )
    sw["CO3"] = (
        get.inorganic._carbonate_from_dic_H(dic, h_scale, totals, k_constants) * 1e-6
    )
    sw["CO2"] = get.inorganic._CO2_from_dic_H(dic, h_scale, totals, k_constants) * 1e-6
    # Borate
    sw["BOH4"] = sw["alk_borate"] = (
        totals["borate"] * k_constants["borate"] / (k_constants["borate"] + h_scale)
    )
    sw["BOH3"] = totals["borate"] * h_scale / (k_constants["borate"] + h_scale)
    # Water
    sw["OH"] = k_constants["water"] / h_scale
    sw["Hfree"] = h_scale * k_constants["pH_factor_to_Free"]
    # Phosphate
    sw.update(phosphoric(h_scale, totals, k_constants))
    sw["alk_phosphate"] = sw["HPO4"] + 2 * sw["PO4"] - sw["H3PO4"]  # Dickson
    # sw["alk_phosphate"] = sw["H2PO4"] + 2 * sw["HPO4"] + 3 * sw["PO4"]  # charge-balance
    # Silicate
    sw["H3SiO4"] = sw["alk_silicate"] = (
        totals["silicate"]
        * k_constants["silicate"]
        / (k_constants["silicate"] + h_scale)
    )
    sw["H4SiO4"] = totals["silicate"] * h_scale / (k_constants["silicate"] + h_scale)
    # Ammonium
    sw["NH3"] = sw["alk_ammonia"] = (
        totals["ammonia"] * k_constants["ammonia"] / (k_constants["ammonia"] + h_scale)
    )
    sw["NH4"] = totals["ammonia"] * h_scale / (k_constants["ammonia"] + h_scale)
    # Sulfide
    sw["HS"] = sw["alk_sulfide"] = (
        totals["sulfide"] * k_constants["sulfide"] / (k_constants["sulfide"] + h_scale)
    )
    sw["H2S"] = totals["sulfide"] * h_scale / (k_constants["sulfide"] + h_scale)
    # KSO4 and KF are always on the Free scale, so:
    # Sulfate
    sw["HSO4"] = (
        totals["sulfate"] * sw["Hfree"] / (sw["Hfree"] + k_constants["bisulfate"])
    )
    sw["SO4"] = (
        totals["sulfate"]
        * k_constants["bisulfate"]
        / (sw["Hfree"] + k_constants["bisulfate"])
    )
    # Fluoride
    sw["HF"] = (
        totals["fluoride"] * sw["Hfree"] / (sw["Hfree"] + k_constants["fluoride"])
    )
    sw["F"] = (
        totals["fluoride"]
        * k_constants["fluoride"]
        / (sw["Hfree"] + k_constants["fluoride"])
    )
    # Extra alkalinity components (added in v1.6.0)
    sw["alpha"] = (
        totals["alpha"] * k_constants["alpha"] / (k_constants["alpha"] + h_scale)
    )
    sw["alphaH"] = totals["alpha"] * h_scale / (k_constants["alpha"] + h_scale)
    sw["beta"] = totals["beta"] * k_constants["beta"] / (k_constants["beta"] + h_scale)
    sw["betaH"] = totals["beta"] * h_scale / (k_constants["beta"] + h_scale)
    zlp = 4.5  # pK of 'zero level of protons' [WZK07]
    sw["alk_alpha"] = np.where(
        -np.log10(k_constants["alpha"]) <= zlp, -sw["alphaH"], sw["alpha"]
    )
    sw["alk_beta"] = np.where(
        -np.log10(k_constants["beta"]) <= zlp, -sw["betaH"], sw["beta"]
    )
    # Total alkalinity
    sw["alkalinity"] = (
        sw["HCO3"]
        + 2 * sw["CO3"]
        + sw["alk_borate"]
        + sw["OH"]
        + sw["alk_phosphate"]
        + sw["alk_silicate"]
        + sw["alk_ammonia"]
        + sw["alk_sulfide"]
        - sw["Hfree"]
        - sw["HSO4"]
        - sw["HF"]
        + sw["alk_alpha"]
        + sw["alk_beta"]
    )
    return sw


def inorganic_zlp(dic, pH, totals, k_constants, pzlp):
    """Calculate the full inorganic chemical speciation of seawater given DIC and pH
    with a variable zero-level of protons (zlp).
    """
    zlp = 10.0**-pzlp
    # First calculate speciation and alkalinity with a standard ZLP of 4.5
    sw = inorganic(dic, pH, totals, k_constants)
    # Then overwrite alkalinity using the new ZLP
    # Water
    sw["alkalinity"] = sw["OH"] - sw["Hfree"]
    # Carbonate
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["carbonic_1"] <= zlp,
        sw["HCO3"] + 2 * sw["CO3"],
        np.where(
            k_constants["carbonic_2"] <= zlp,
            sw["CO3"] - sw["CO2"],
            -sw["HCO3"] - 2 * sw["CO2"],
        ),
    )
    # Borate
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["borate"] <= zlp, sw["BOH4"], -sw["BOH3"]
    )
    # Phosphate
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["phosphoric_1"] <= zlp,
        sw["H2PO4"] + 2 * sw["HPO4"] + 3 * sw["PO4"],
        np.where(
            k_constants["phosphoric_2"] <= zlp,
            -sw["H3PO4"] + sw["HPO4"] + 2 * sw["PO4"],
            np.where(
                k_constants["phosphoric_3"] <= zlp,
                -2 * sw["H3PO4"] - sw["H2PO4"] + sw["PO4"],
                -3 * sw["H3PO4"] - 2 * sw["H2PO4"] - sw["HPO4"],
            ),
        ),
    )
    # Silicate
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["silicate"] <= zlp, sw["H3SiO4"], -sw["H4SiO4"]
    )
    # Ammonia
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["ammonia"] <= zlp, sw["NH3"], -sw["NH4"]
    )
    # Sulfide
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["sulfide"] <= zlp, sw["HS"], -sw["H2S"]
    )
    # Sulfate
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["bisulfate"] <= zlp, sw["SO4"], -sw["HSO4"]
    )
    # Fluoride
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["fluoride"] <= zlp, sw["F"], -sw["HF"]
    )
    # Alpha
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["alpha"] <= zlp, -sw["alphaH"], sw["alpha"]
    )
    # Beta
    sw["alkalinity"] = sw["alkalinity"] + np.where(
        k_constants["beta"] <= zlp, -sw["betaH"], sw["beta"]
    )
    return sw


def inorganic_dom(dic, pH, totals, k_constants, nd_params):
    """Calculate the full chemical speciation of seawater given DIC and pH, including
    NICA-Donnan DOM, with the Donnan chi value unknown.
    """
    # Calculate inorganic speciation without DOM
    sw = inorganic(dic, pH, totals, k_constants)
    # NICA-Donnan for DOM
    c_ions, z_ions = dom.get_ions(sw, totals, nd_params["density"])
    ionic_strength = dom.get_ionic_strength(c_ions, z_ions)
    sw["log10_chi"] = dom.solve_log10_chi(c_ions, z_ions, ionic_strength, nd_params)
    chi = 10.0 ** sw["log10_chi"]
    cH = sw["Hfree"] * nd_params["density"]
    sw["Q_H"] = dom.nica(cH, chi, nd_params)
    sw["alk_dom"] = -(dom.nica_charge(cH, chi, nd_params) * totals["dom"] * 1e-6)
    sw["domH"] = sw["Q_H"] * totals["dom"] * 1e-6
    # Update total alkalinity with the DOM component
    sw["alkalinity"] = sw["alkalinity"] + sw["alk_dom"]
    return sw


def inorganic_dom_metals(dic, pH, totals, k_constants, nd_params):
    """Calculate the full chemical speciation of seawater given DIC and pH, including
    NICA-Donnan DOM and divalent metal competition, with the Donnan chi value unknown.
    """
    # Calculate inorganic speciation without DOM
    sw = inorganic(dic, pH, totals, k_constants)
    # Prepare for NICA-Donnan for DOM
    c_ions, z_ions = dom.get_ions(sw, totals, nd_params["density"])
    ionic_strength = dom.get_ionic_strength(c_ions, z_ions)
    # Solve for the Donnan factor (chi)
    sw["log10_chi"] = dom.solve_log10_chi_metals(
        c_ions, z_ions, ionic_strength, nd_params
    )
    chi = 10.0 ** sw["log10_chi"]
    # Convert contents to concentrations
    cH = sw["Hfree"] * nd_params["density"]
    cM = {
        "Ca": totals["calcium"] * nd_params["density"],
        "Mg": totals["magnesium"] * nd_params["density"],
        "Sr": totals["strontium"] * nd_params["density"],
    }
    # Calculate bound metals
    for m in cM:
        sw["Q_{}".format(m)] = dom.nica_specific_metal(m, cH, cM, chi, nd_params)
        sw["dom{}".format(m)] = sw["Q_{}".format(m)] * totals["dom"] * 1e-6
    sw["Q_M"] = sw["Q_Ca"] + sw["Q_Mg"] + sw["Q_Sr"]
    # Calculate bound protons
    sw["Q_H"] = dom.nica_metals(cH, cM, chi, nd_params)
    # Calculate DOM speciation
    sw["alk_dom"] = (
        (nd_params["Qmax_H1"] + nd_params["Qmax_H2"] - sw["Q_H"] - 2 * sw["Q_M"])
        * totals["dom"]
        * 1e-6
    )
    sw["domH"] = sw["Q_H"] * totals["dom"] * 1e-6
    # Update total alkalinity with the DOM components
    sw["alkalinity"] = (
        sw["alkalinity"] + sw["alk_dom"] + 2 * (sw["domCa"] + sw["domMg"] + sw["domSr"])
    )
    return sw


def inorganic_dom_chi(dic, pH, totals, k_constants, nd_params, log10_chi):
    """Calculate the full chemical speciation of seawater given DIC and pH, including
    NICA-Donnan DOM, with a known Donnan chi value.
    """
    # Calculate inorganic speciation without DOM
    sw = inorganic(dic, pH, totals, k_constants)
    # Calculate DOM-associated alkalinity
    sw["alk_dom"] = -(
        dom.nica_charge(sw["Hfree"] * nd_params["density"], 10.0**log10_chi, nd_params)
        * totals["dom"]
        * 1e-6
    )
    # Update total alkalinity with the DOM component
    sw["alkalinity"] = sw["alkalinity"] + sw["alk_dom"]
    return sw


def inorganic_dom_metals_chi(dic, pH, totals, k_constants, nd_params, log10_chi):
    """Calculate the full chemical speciation of seawater given DIC and pH, including
    NICA-Donnan DOM and divalent metal competition, with a known Donnan chi value.
    """
    # Calculate inorganic speciation without DOM
    sw = inorganic(dic, pH, totals, k_constants)
    # Calculate DOM-associated alkalinity
    cH = sw["Hfree"] * nd_params["density"]
    chi = 10.0**log10_chi
    cM = {
        "Ca": totals["calcium"] * nd_params["density"],
        "Mg": totals["magnesium"] * nd_params["density"],
        "Sr": totals["strontium"] * nd_params["density"],
    }
    sw["alk_dom"] = -(
        dom.nica_charge_metals(cH, cM, chi, nd_params) * totals["dom"] * 1e-6
    )
    # Calculate bound metals
    for m in cM:
        sw["Q_{}".format(m)] = dom.nica_specific_metal(m, cH, cM, chi, nd_params)
        sw["dom{}".format(m)] = sw["Q_{}".format(m)] * totals["dom"] * 1e-6
    sw["Q_M"] = sw["Q_Ca"] + sw["Q_Mg"] + sw["Q_Sr"]
    # Update total alkalinity with the DOM component
    sw["alkalinity"] = (
        sw["alkalinity"] + sw["alk_dom"] + 2 * (sw["domCa"] + sw["domMg"] + sw["domSr"])
    )
    return sw
