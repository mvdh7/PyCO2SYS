# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate chemical speciation."""

from jax import numpy as np
from . import get


def get_HCO3(dic, H, k_H2CO3, k_HCO3):
    """Calculate bicarbonate ion from dissolved inorganic carbon and [H+].

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] content in mol/kg-sw.
    k_H2CO3, k_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    return dic * k_H2CO3 * H / (H**2 + k_H2CO3 * H + k_H2CO3 * k_HCO3)


def get_CO3(dic, H, k_H2CO3, k_HCO3):
    """Calculate carbonate ion from dissolved inorganic carbon and [H+].

    Based on CalculateCarbfromdicpH, version 01.0, 06-12-2019, by Denis Pierrot.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] in mol/kg-sw.
    k_H2CO3, k_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    return dic * k_H2CO3 * k_HCO3 / (H**2 + k_H2CO3 * H + k_H2CO3 * k_HCO3)


def get_PO4(total_phosphate, H, k_H3PO4, k_H2PO4, k_HPO4):
    """Phosphate content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H3PO4, k_H2PO4, k_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [PO₄³⁻] in µmol/kg-sw.
    """
    KP1, KP2, KP3 = k_H3PO4, k_H2PO4, k_HPO4
    return (
        total_phosphate
        * KP1
        * KP2
        * KP3
        / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_HPO4(total_phosphate, H, k_H3PO4, k_H2PO4, k_HPO4):
    """Hydrogen phosphate content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H3PO4, k_H2PO4, k_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [HPO₄²⁻] in µmol/kg-sw.
    """
    KP1, KP2, KP3 = k_H3PO4, k_H2PO4, k_HPO4
    return (
        total_phosphate
        * KP1
        * KP2
        * H
        / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_H2PO4(total_phosphate, H, k_H3PO4, k_H2PO4, k_HPO4):
    """Dihydrogen phosphate content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H3PO4, k_H2PO4, k_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [H₂PO₄⁻] in µmol/kg-sw.
    """
    KP1, KP2, KP3 = k_H3PO4, k_H2PO4, k_HPO4
    return (
        total_phosphate
        * KP1
        * H**2
        / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_H3PO4(total_phosphate, H, k_H3PO4, k_H2PO4, k_HPO4):
    """Undissociated phosphoric acid content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H3PO4, k_H2PO4, k_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [H₃PO₄] in µmol/kg-sw.
    """
    KP1, KP2, KP3 = k_H3PO4, k_H2PO4, k_HPO4
    return (
        total_phosphate * H**3 / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_BOH4(total_borate, H, k_BOH3):
    """Tetrahydroxyborate content in µmol/kg-sw.

    Parameters
    ----------
    total_borate : float
        Total borate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_BOH3 : float
        Boric acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [B(OH)₄⁻] in µmol/kg-sw.
    """
    return total_borate * k_BOH3 / (k_BOH3 + H)


def get_BOH3(total_borate, H, k_BOH3):
    """Undissociated boric acid content in µmol/kg-sw.

    Parameters
    ----------
    total_borate : float
        Total borate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_BOH3 : float
        Boric acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [B(OH)₃] in µmol/kg-sw.
    """
    return total_borate * H / (k_BOH3 + H)


def get_OH(H, k_H2O):
    """Hydroxide ion content in µmol/kg-sw.

    Parameters
    ----------
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H2O : float
        Water dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [OH⁻] in µmol/kg-sw.
    """
    return 1e6 * k_H2O / H


def get_H_free(H, opt_to_free):
    """Hydrogen ion content in µmol/kg-sw.

    Parameters
    ----------
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H2O : float
        Water dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H⁺] in µmol/kg-sw.
    """
    return 1e6 * H * opt_to_free


def get_H3SiO4(total_silicate, H, k_Si):
    """Trihydrogen orthosilicate content in µmol/kg-sw.

    Parameters
    ----------
    total_silicate : float
        Total silicate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_Si : float
        Orthosilicic acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H₃SiO₄⁻] in µmol/kg-sw.
    """
    return total_silicate * k_Si / (k_Si + H)


def get_H4SiO4(total_silicate, H, k_Si):
    """Undissociated orthosilicic acid content in µmol/kg-sw.

    Parameters
    ----------
    total_silicate : float
        Total silicate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_Si : float
        Orthosilicic acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H₄SiO₄] in µmol/kg-sw.
    """
    return total_silicate * H / (k_Si + H)


def get_HSO4(total_sulfate, H_free, k_HSO4_free):
    """Bisulfate ion content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    k_HSO4_free : float
        Bisulfate dissociation constant on the free scale.

    Returns
    -------
    float
        [HSO₄⁻] in µmol/kg-sw.
    """
    return 1e-6 * total_sulfate * H_free / ((1e-6 * H_free) + k_HSO4_free)


def get_SO4(total_sulfate, H_free, k_HSO4_free):
    """Sulfate ion content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    k_HSO4_free : float
        Bisulfate dissociation constant on the free scale.

    Returns
    -------
    float
        [SO₄²⁻] in µmol/kg-sw.
    """
    return 1e-6 * total_sulfate * k_HSO4_free / ((1e-6 * H_free) + k_HSO4_free)


def get_HF(total_fluoride, H_free, k_HF_free):
    """Undissociated HF content in µmol/kg-sw.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.

    Returns
    -------
    float
        [HF] in µmol/kg-sw.
    """
    return 1e-6 * total_fluoride * H_free / ((1e-6 * H_free) + k_HF_free)


def get_F(total_fluoride, H_free, k_HF_free):
    """Fluoride ion content in µmol/kg-sw.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    k_HF_free : float
        HF dissociation constant on the free scale.

    Returns
    -------
    float
        [F⁻] in µmol/kg-sw.
    """
    return 1e-6 * total_fluoride * k_HF_free / ((1e-6 * H_free) + k_HF_free)


def get_NH3(total_ammonia, H, k_NH3):
    """Ammonia content in µmol/kg-sw.

    Parameters
    ----------
    total_ammonia : float
        Total fluoride in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_NH3_free : float
        Ammonia association constant on opt_pH_scale.

    Returns
    -------
    float
        [NH₃] in µmol/kg-sw.
    """
    return total_ammonia * k_NH3 / (H + k_NH3)


def get_NH4(total_ammonia, H, k_NH3):
    """Ammonium content in µmol/kg-sw.

    Parameters
    ----------
    total_ammonia : float
        Total fluoride in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_NH3_free : float
        Ammonia association constant on opt_pH_scale.

    Returns
    -------
    float
        [NH₄] in µmol/kg-sw.
    """
    return total_ammonia * H / (H + k_NH3)


def get_HS(total_sulfide, H, k_H2S):
    """Hydrogen sulfide content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfide : float
        Total fluoride in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H2S : float
        Hydrogen disulfide dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [HS⁻] in µmol/kg-sw.
    """
    return total_sulfide * k_H2S / (H + k_H2S)


def get_H2S(total_sulfide, H, k_H2S):
    """Undissociated hydrogen disulfide content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfide : float
        Total fluoride in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    k_H2S : float
        Hydrogen disulfide dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H₂S] in µmol/kg-sw.
    """
    return total_sulfide * H / (H + k_H2S)


def sum_alkalinity(
    H_free, OH, HCO3, CO3, BOH4, HPO4, PO4, H3PO4, H3SiO4, NH3, HS, HSO4, HF
):
    """Total alkalinity in µmol/kg-sw.

    Parameters
    ----------
    H_free : float
        Hydrogen ion content on the free scale in µmol/kg-sw.
    OH : float
        Hydroxide ion content in µmol/kg-sw.
    HCO3 : float
        Bicarbonate ion content in µmol/kg-sw.
    CO3 : float
        Carbonate ion content in µmol/kg-sw.
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

    Returns
    -------
    float
        Total alkalinity in µmol/kg-sw.
    """
    return (
        HCO3
        + 2 * CO3
        + BOH4
        + OH
        + HPO4
        + 2 * PO4
        - H3PO4
        + H3SiO4
        + NH3
        + HS
        - H_free
        - HSO4
        - HF
    )


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
