# PyCO2SYSv2 a.k.a. aqualibrium: marine carbonate system calculations in Python.
# Copyright (C) 2020--2023  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Calculate chemical speciation."""


def get_HCO3(dic, H, pk_H2CO3, pk_HCO3):
    """Calculate bicarbonate ion from dissolved inorganic carbon and [H+].

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] content in mol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Bicarbonate ion content in µmol/kg-sw.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    return dic * K1 * H / (H**2 + K1 * H + K1 * K2)


def get_CO3(dic, H, pk_H2CO3, pk_HCO3):
    """Calculate carbonate ion from dissolved inorganic carbon and [H+].

    Based on CalculateCarbfromdicpH, version 01.0, 06-12-2019, by Denis Pierrot.

    Parameters
    ----------
    dic : float
        DIC in µmol/kg-sw.
    H : float
        [H+] in mol/kg-sw.
    pk_H2CO3, pk_HCO3 : float
        Carbonic acid dissociation constants.

    Returns
    -------
    float
        Carbonate ion content in µmol/kg-sw.
    """
    K1 = 10**-pk_H2CO3
    K2 = 10**-pk_HCO3
    return dic * K1 * K2 / (H**2 + K1 * H + K1 * K2)


def get_PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4):
    """Phosphate content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H3PO4, pk_H2PO4, pk_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [PO₄³⁻] in µmol/kg-sw.
    """
    KP1 = 10**-pk_H3PO4
    KP2 = 10**-pk_H2PO4
    KP3 = 10**-pk_HPO4
    return (
        total_phosphate
        * KP1
        * KP2
        * KP3
        / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_HPO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4):
    """Hydrogen phosphate content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H3PO4, pk_H2PO4, pk_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [HPO₄²⁻] in µmol/kg-sw.
    """
    KP1 = 10**-pk_H3PO4
    KP2 = 10**-pk_H2PO4
    KP3 = 10**-pk_HPO4
    return (
        total_phosphate
        * KP1
        * KP2
        * H
        / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_H2PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4):
    """Dihydrogen phosphate content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H3PO4, pk_H2PO4, pk_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [H₂PO₄⁻] in µmol/kg-sw.
    """
    KP1 = 10**-pk_H3PO4
    KP2 = 10**-pk_H2PO4
    KP3 = 10**-pk_HPO4
    return (
        total_phosphate
        * KP1
        * H**2
        / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_H3PO4(total_phosphate, H, pk_H3PO4, pk_H2PO4, pk_HPO4):
    """Undissociated phosphoric acid content in µmol/kg-sw.

    Parameters
    ----------
    total_phosphate : float
        Total phosphate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H3PO4, pk_H2PO4, pk_HPO4 : float
        Phosphoric acid dissociation constants on opt_pH_scale.

    Returns
    -------
    float
        [H₃PO₄] in µmol/kg-sw.
    """
    KP1 = 10**-pk_H3PO4
    KP2 = 10**-pk_H2PO4
    KP3 = 10**-pk_HPO4
    return (
        total_phosphate * H**3 / (H**3 + KP1 * H**2 + KP1 * KP2 * H + KP1 * KP2 * KP3)
    )


def get_BOH4(total_borate, H, pk_BOH3):
    """Tetrahydroxyborate content in µmol/kg-sw.

    Parameters
    ----------
    total_borate : float
        Total borate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_BOH3 : float
        Boric acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [B(OH)₄⁻] in µmol/kg-sw.
    """
    return total_borate * 10**-pk_BOH3 / (10**-pk_BOH3 + H)


def get_BOH3(total_borate, H, pk_BOH3):
    """Undissociated boric acid content in µmol/kg-sw.

    Parameters
    ----------
    total_borate : float
        Total borate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_BOH3 : float
        Boric acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [B(OH)₃] in µmol/kg-sw.
    """
    return total_borate * H / (10**-pk_BOH3 + H)


def get_OH(H, pk_H2O):
    """Hydroxide ion content in µmol/kg-sw.

    Parameters
    ----------
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H2O : float
        Water dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [OH⁻] in µmol/kg-sw.
    """
    return 1e6 * 10**-pk_H2O / H


def get_H_free(H, opt_to_free):
    """Hydrogen ion content in µmol/kg-sw.

    Parameters
    ----------
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H2O : float
        Water dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H⁺] in µmol/kg-sw.
    """
    return 1e6 * H * 10**-opt_to_free


def get_H3SiO4(total_silicate, H, pk_Si):
    """Trihydrogen orthosilicate content in µmol/kg-sw.

    Parameters
    ----------
    total_silicate : float
        Total silicate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_Si : float
        Orthosilicic acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H₃SiO₄⁻] in µmol/kg-sw.
    """
    return total_silicate * 10**-pk_Si / (10**-pk_Si + H)


def get_H4SiO4(total_silicate, H, pk_Si):
    """Undissociated orthosilicic acid content in µmol/kg-sw.

    Parameters
    ----------
    total_silicate : float
        Total silicate in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_Si : float
        Orthosilicic acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H₄SiO₄] in µmol/kg-sw.
    """
    return total_silicate * H / (10**-pk_Si + H)


def get_HSO4(total_sulfate, H_free, pk_HSO4_free):
    """Bisulfate ion content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    pk_HSO4_free : float
        Bisulfate dissociation constant on the free scale.

    Returns
    -------
    float
        [HSO₄⁻] in µmol/kg-sw.
    """
    H_free_molkg = H_free * 1e-6
    return total_sulfate * H_free_molkg / (H_free_molkg + 10**-pk_HSO4_free)


def get_SO4(total_sulfate, H_free, pk_HSO4_free):
    """Sulfate ion content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfate : float
        Total sulfate in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    pk_HSO4_free : float
        Bisulfate dissociation constant on the free scale.

    Returns
    -------
    float
        [SO₄²⁻] in µmol/kg-sw.
    """
    H_free_molkg = H_free * 1e-6
    return total_sulfate * 10**-pk_HSO4_free / (H_free_molkg + 10**-pk_HSO4_free)


def get_HF(total_fluoride, H_free, pk_HF_free):
    """Undissociated HF content in µmol/kg-sw.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.

    Returns
    -------
    float
        [HF] in µmol/kg-sw.
    """
    H_free_molkg = H_free * 1e-6
    return total_fluoride * H_free_molkg / (H_free_molkg + 10**-pk_HF_free)


def get_F(total_fluoride, H_free, pk_HF_free):
    """Fluoride ion content in µmol/kg-sw.

    Parameters
    ----------
    total_fluoride : float
        Total fluoride in µmol/kg-sw.
    H_free : float
        [H⁺] on the free scale in µmol/kg-sw.
    pk_HF_free : float
        HF dissociation constant on the free scale.

    Returns
    -------
    float
        [F⁻] in µmol/kg-sw.
    """
    H_free_molkg = H_free * 1e-6
    return total_fluoride * 10**-pk_HF_free / (H_free_molkg + 10**-pk_HF_free)


def get_NH3(total_ammonia, H, pk_NH3):
    """Ammonia content in µmol/kg-sw.

    Parameters
    ----------
    total_ammonia : float
        Total fluoride in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_NH3_free : float
        Ammonia association constant on opt_pH_scale.

    Returns
    -------
    float
        [NH₃] in µmol/kg-sw.
    """
    return total_ammonia * 10**-pk_NH3 / (H + 10**-pk_NH3)


def get_NH4(total_ammonia, H, pk_NH3):
    """Ammonium content in µmol/kg-sw.

    Parameters
    ----------
    total_ammonia : float
        Total fluoride in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_NH3_free : float
        Ammonia association constant on opt_pH_scale.

    Returns
    -------
    float
        [NH₄] in µmol/kg-sw.
    """
    return total_ammonia * H / (H + 10**-pk_NH3)


def get_HS(total_sulfide, H, pk_H2S):
    """Hydrogen sulfide content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfide : float
        Total sulfide in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H2S : float
        Hydrogen disulfide dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [HS⁻] in µmol/kg-sw.
    """
    return total_sulfide * 10**-pk_H2S / (H + 10**-pk_H2S)


def get_H2S(total_sulfide, H, pk_H2S):
    """Undissociated hydrogen disulfide content in µmol/kg-sw.

    Parameters
    ----------
    total_sulfide : float
        Total sulfide in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H2S : float
        Hydrogen disulfide dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [H₂S] in µmol/kg-sw.
    """
    return total_sulfide * H / (H + 10**-pk_H2S)


def get_NO2(total_nitrite, H, pk_HNO2):
    """Nitrite content in µmol/kg-sw.

    Parameters
    ----------
    total_nitrite : float
        Total nitrite in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_H2S : float
        Nitrous acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [NO₂⁻] in µmol/kg-sw.
    """
    return total_nitrite * 10**-pk_HNO2 / (H + 10**-pk_HNO2)


def get_HNO2(total_nitrite, H, pk_HNO2):
    """Undissociated nitrous acide content in µmol/kg-sw.

    Parameters
    ----------
    total_nitrite : float
        Total nitrite in µmol/kg-sw.
    H : float
        [H⁺] on opt_pH_scale in mol/kg-sw.
    pk_HNO2 : float
        Nitrous acid dissociation constant on opt_pH_scale.

    Returns
    -------
    float
        [HNO₂] in µmol/kg-sw.
    """
    return total_nitrite * H / (H + 10**-pk_HNO2)


def sum_alkalinity(
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
    HNO2 : float
        Nitrous acid content in µmol/kg-sw.

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
        - HNO2
    )
