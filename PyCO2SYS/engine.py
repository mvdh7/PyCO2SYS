# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
# ruff: noqa: C408
from inspect import signature

import networkx as nx
from jax import numpy as np

from . import (
    bio,
    buffers,
    constants,
    convert,
    equilibria,
    gas,
    salts,
    solubility,
    solve,
    upsilon,
)
from .classes.function_graph import (
    FunctionGraph,
    ShortcutDotDict,
    ShortcutsDict,
)
from .meta import PyCO2SYSError, warn
from .uncertainty import covmx


citations = {
    "opt_pH_scale": {
        1: "total pH scale",
        2: "seawater pH scale",
        3: "free pH scale",
        4: "NBS pH scale",
    },
    "opt_k_carbonic": {
        1: "Roy et al. (1993)",
        2: "Goyet & Poisson (1989)",
        3: "Hansson (1973) refit by Dickson & Millero (1987)",
        4: "Mehrbach et al. (1987) refit by Dickson & Millero (1987)",
        5: "Hansson (1973) and Mehrbach et al. (1987) refit by Dickson & Millero (1987)",
        6: "Mehrbach et al. (1973)",
        7: "Mehrbach et al. (1973)",
        8: "Millero (1979), freshwater",
        9: "Cai & Wang (1998)",
        10: "Lueker et al. (2000)",
        11: "Mojica Prieto & Millero (2002)",
        12: "Millero et al. (2002)",
        13: "Millero et al. (2006)",
        14: "Millero (2010)",
        15: "Waters & Millero (2013) corrected by Waters et al. (2014)",
        16: "Sulpis et al. (2020)",
        17: "Schockman & Byrne (2021)",
        18: "Papadimitriou et al. (2018)",
        19: "Martin-Mayor et al. (2025)",
    },
    "opt_total_borate": {
        1: "Uppström (1974)",
        2: "Lee et al. (2010)",
        3: "Kuliński et al. (2018)",
    },
    "opt_Ca": {
        1: "Riley & Tongudai (1967)",
        2: "Culkin (1965)",
    },
    "opt_k_HSO4": {
        1: "Dickson (1990a)",
        2: "Khoo et al. (1977)",
        3: "Waters & Millero (2013) corrected by Waters et al. (2014)",
    },
    "opt_k_HF": {
        1: "Dickson & Riley (1979)",
        2: "Perez & Fraga (1987)",
    },
    "opt_k_BOH3": {
        1: "Dickson (1990b)",
        2: "Li et al. (1969)",
    },
    "opt_k_phosphate": {
        1: "Yao & Millero (1995)",
        2: "Kester & Pytkowicz (1967)",
    },
    "opt_k_NH3": {
        1: "Clegg & Whitfield (1995)",
        2: "Yao & Millero (1995)",
    },
    "opt_k_Si": {
        1: "Yao & Millero (1995)",
        2: "Sillén et al. (1964)",
    },
    "opt_k_calcite": {
        1: "Mucci (1983)",
        2: "Ingle (1975)",
    },
    "opt_k_aragonite": {
        1: "Mucci (1983)",
        2: "Ingle et al. (1973)",
    },
    "opt_k_H2O": {
        1: "Millero (1995)",
        2: "Millero (1979)",
        3: "Harned & Owen (1958) refit by Millero (1979), freshwater",
    },
    "opt_k_HNO2": {
        1: "Borer et al. (2024)",
        2: "Borer et al. (2024), freshwater",
    },
    "opt_factor_k_H2CO3": {
        1: "Millero (1995)",
        2: "Edmond & Gieskes (1970)",
        3: "Millero (1983), freshwater",
    },
    "opt_factor_k_HCO3": {
        1: "Millero (1995)",
        2: "Edmond & Gieskes (1970)",
        3: "Millero (1983), freshwater",
    },
    "opt_factor_k_BOH3": {
        1: "Millero (1979)",
        2: "Edmond & Gieskes (1970)",
    },
    "opt_factor_k_H2O": {
        1: "Millero (1995)",
        2: "Millero (1983), freshwater",
    },
    "opt_gas_constant": {
        1: "DOEv2",
        2: "DOEv3",
        3: "2018 CODATA",
    },
    "opt_fugacity_factor": {
        1: "pCO2 ≠ fCO2",
        2: "pCO2 = fCO2",
    },
    "opt_HCO3_root": {
        1: "find low-pH root with DIC-HCO3 known pair",
        2: "find high-pH root with DIC-HCO3 known pair",
    },
    "method_fCO2": {
        1: "Humphreys (2024), parameterised υ_h",
        2: "Humphreys (2024), constant υ_h fitted to Takahashi et al. (1993) dataset",
        3: "Humphreys (2024), constant theoretical υ_h",
        4: "Humphreys (2024), user provided b_h",
        5: "Takahashi et al. (1993), linear fit",
        6: "Takahashi et al. (1993), quadratic fit",
    },
    "which_fCO2_insitu": {
        1: "pre-adjustment values are in situ",
        2: "adjusted values are in situ",
    },
}

# Define functions for calculations that depend neither on icase nor opts:
get_funcs = {
    # Total salt contents
    "ionic_strength": salts.ionic_strength_DOE94,
    "total_fluoride": salts.total_fluoride_R65,
    "total_sulfate": salts.total_sulfate_MR66,
    # Equilibrium constants at 1 atm and on reported pH scale
    "pk_CO2_1atm": equilibria.p1atm.pk_CO2_W74,
    "pk_H2S_total_1atm": equilibria.p1atm.pk_H2S_total_YM95,
    # pH scale conversion factors at 1 atm
    "free_to_sws_1atm": lambda total_fluoride, total_sulfate, pk_HF_free_1atm, pk_HSO4_free_1atm: (
        convert.pH_free_to_sws(
            total_fluoride, total_sulfate, pk_HF_free_1atm, pk_HSO4_free_1atm
        )
    ),
    "nbs_to_sws": convert.pH_nbs_to_sws,  # because fH doesn't get pressure-corrected
    "tot_to_sws_1atm": lambda total_fluoride, total_sulfate, pk_HF_free_1atm, pk_HSO4_free_1atm: (
        convert.pH_tot_to_sws(
            total_fluoride, total_sulfate, pk_HF_free_1atm, pk_HSO4_free_1atm
        )
    ),
    # Equilibrium constants at 1 atm and on the seawater pH scale
    "pk_H2S_sws_1atm": lambda pk_H2S_total_1atm, tot_to_sws_1atm: (
        pk_H2S_total_1atm + tot_to_sws_1atm
    ),
    # Pressure correction factors for equilibrium constants
    "factor_k_HSO4": equilibria.pcx.factor_k_HSO4,
    "factor_k_HF": equilibria.pcx.factor_k_HF,
    "factor_k_H2S": equilibria.pcx.factor_k_H2S,
    "factor_k_H3PO4": equilibria.pcx.factor_k_H3PO4,
    "factor_k_H2PO4": equilibria.pcx.factor_k_H2PO4,
    "factor_k_HPO4": equilibria.pcx.factor_k_HPO4,
    "factor_k_Si": equilibria.pcx.factor_k_Si,
    "factor_k_NH3": equilibria.pcx.factor_k_NH3,
    "factor_k_CO2": equilibria.pcx.factor_k_CO2,
    "factor_k_HNO2": equilibria.pcx.factor_k_HNO2,
    # Equilibrium constants at pressure and on the free pH scale
    "pk_HF_free": lambda pk_HF_free_1atm, factor_k_HF: (
        pk_HF_free_1atm - np.log10(factor_k_HF)
    ),
    "pk_HSO4_free": lambda pk_HSO4_free_1atm, factor_k_HSO4: (
        pk_HSO4_free_1atm - np.log10(factor_k_HSO4)
    ),
    # Equilibrium constants at pressure and on the seawater pH scale
    "pk_BOH3_sws": lambda pk_BOH3_sws_1atm, factor_k_BOH3: (
        pk_BOH3_sws_1atm - np.log10(factor_k_BOH3)
    ),
    "pk_H2O_sws": lambda pk_H2O_sws_1atm, factor_k_H2O: (
        pk_H2O_sws_1atm - np.log10(factor_k_H2O)
    ),
    "pk_H2S_sws": lambda pk_H2S_sws_1atm, factor_k_H2S: (
        pk_H2S_sws_1atm - np.log10(factor_k_H2S)
    ),
    "pk_H3PO4_sws": lambda pk_H3PO4_sws_1atm, factor_k_H3PO4: (
        pk_H3PO4_sws_1atm - np.log10(factor_k_H3PO4)
    ),
    "pk_H2PO4_sws": lambda pk_H2PO4_sws_1atm, factor_k_H2PO4: (
        pk_H2PO4_sws_1atm - np.log10(factor_k_H2PO4)
    ),
    "pk_HPO4_sws": lambda pk_HPO4_sws_1atm, factor_k_HPO4: (
        pk_HPO4_sws_1atm - np.log10(factor_k_HPO4)
    ),
    "pk_Si_sws": lambda pk_Si_sws_1atm, factor_k_Si: (
        pk_Si_sws_1atm - np.log10(factor_k_Si)
    ),
    "pk_NH3_sws": lambda pk_NH3_sws_1atm, factor_k_NH3: (
        pk_NH3_sws_1atm - np.log10(factor_k_NH3)
    ),
    "pk_H2CO3_sws": lambda pk_H2CO3_sws_1atm, factor_k_H2CO3: (
        pk_H2CO3_sws_1atm - np.log10(factor_k_H2CO3)
    ),
    "pk_HCO3_sws": lambda pk_HCO3_sws_1atm, factor_k_HCO3: (
        pk_HCO3_sws_1atm - np.log10(factor_k_HCO3)
    ),
    "pk_HNO2_sws": lambda pk_HNO2_sws_1atm, factor_k_HNO2: (
        pk_HNO2_sws_1atm - np.log10(factor_k_HNO2)
    ),
    # Equilibrium constants at pressure and on the requested pH scale
    "pk_CO2": lambda pk_CO2_1atm, factor_k_CO2: (
        pk_CO2_1atm - np.log10(factor_k_CO2)
    ),
    "pk_BOH3": lambda sws_to_opt, pk_BOH3_sws: sws_to_opt + pk_BOH3_sws,
    "pk_H2O": lambda sws_to_opt, pk_H2O_sws: sws_to_opt + pk_H2O_sws,
    "pk_H2S": lambda sws_to_opt, pk_H2S_sws: sws_to_opt + pk_H2S_sws,
    "pk_H3PO4": lambda sws_to_opt, pk_H3PO4_sws: sws_to_opt + pk_H3PO4_sws,
    "pk_H2PO4": lambda sws_to_opt, pk_H2PO4_sws: sws_to_opt + pk_H2PO4_sws,
    "pk_HPO4": lambda sws_to_opt, pk_HPO4_sws: sws_to_opt + pk_HPO4_sws,
    "pk_Si": lambda sws_to_opt, pk_Si_sws: sws_to_opt + pk_Si_sws,
    "pk_NH3": lambda sws_to_opt, pk_NH3_sws: sws_to_opt + pk_NH3_sws,
    "pk_H2CO3": lambda sws_to_opt, pk_H2CO3_sws: sws_to_opt + pk_H2CO3_sws,
    "pk_HCO3": lambda sws_to_opt, pk_HCO3_sws: sws_to_opt + pk_HCO3_sws,
    "pk_HNO2": lambda sws_to_opt, pk_HNO2_sws: sws_to_opt + pk_HNO2_sws,
    # Gasses
    "vp_factor": gas.vpfactor,
    # Mg-calcite solubility
    "acf_Ca": solubility.get_activity_coefficient_Ca,
    "acf_Mg": solubility.get_activity_coefficient_Mg,
    "acf_CO3": solubility.get_activity_coefficient_CO3,
    "pk_Mg_calcite_1atm": solubility.get_pk_Mg_calcite_1atm,
    "pk_Mg_calcite": solubility.get_pk_Mg_calcite,
    "Mg": salts.Mg_reference_composition,
}

# Define functions for calculations that depend on icase:
get_funcs_core = {}
for i in [0, 3, 4, 5, 6, 8, 9, 10, 11]:
    get_funcs_core[i] = {}
# alkalinity and DIC
get_funcs_core[102] = {
    "pH": solve.inorganic.pH_from_alkalinity_dic,
    "fCO2": solve.inorganic.fCO2_from_dic_pH,
    "CO3": solve.inorganic.CO3_from_dic_pH,
    "HCO3": solve.inorganic.HCO3_from_dic_pH,
}
# alkalinity and pH
get_funcs_core[103] = {
    "dic": solve.inorganic.dic_from_alkalinity_pH_speciated,
    "fCO2": solve.inorganic.fCO2_from_dic_pH,
    "CO3": solve.inorganic.CO3_from_dic_pH,
    "HCO3": solve.inorganic.HCO3_from_dic_pH,
}
# alkalinity and pCO2, fCO2, CO2, xCO2
for i in [104, 105, 108, 109]:
    get_funcs_core[i] = {
        "pH": solve.inorganic.pH_from_alkalinity_fCO2,
        "dic": solve.inorganic.dic_from_pH_fCO2,
        "HCO3": solve.inorganic.HCO3_from_pH_fCO2,
        "CO3": solve.inorganic.CO3_from_dic_pH,
    }
# alkalinity and CO3, omega
for i in [106, 110, 111]:
    get_funcs_core[i] = {
        "pH": solve.inorganic.pH_from_alkalinity_CO3,
        "dic": solve.inorganic.dic_from_pH_CO3,
        "HCO3": solve.inorganic.HCO3_from_pH_CO3,
        "fCO2": solve.inorganic.fCO2_from_pH_CO3,
    }
# alkalinity and HCO3
get_funcs_core[107] = {
    "pH": solve.inorganic.pH_from_alkalinity_HCO3,
    "dic": solve.inorganic.dic_from_pH_HCO3,
    "CO3": solve.inorganic.CO3_from_pH_HCO3,
    "fCO2": solve.inorganic.fCO2_from_pH_HCO3,
}
# DIC and pH
get_funcs_core[203] = {
    "fCO2": solve.inorganic.fCO2_from_dic_pH,
    "CO3": solve.inorganic.CO3_from_dic_pH,
    "HCO3": solve.inorganic.HCO3_from_dic_pH,
    "alkalinity": solve.speciate.sum_alkalinity,
}
# DIC and pCO2, fCO2, CO2, xCO2
for i in [204, 205, 208, 209]:
    get_funcs_core[i] = {
        "pH": solve.inorganic.pH_from_dic_fCO2,
        "HCO3": solve.inorganic.HCO3_from_pH_fCO2,
        "CO3": solve.inorganic.CO3_from_dic_pH,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# DIC and CO3, omega
for i in [206, 210, 211]:
    get_funcs_core[i] = {
        "pH": solve.inorganic.pH_from_dic_CO3,
        "HCO3": solve.inorganic.HCO3_from_pH_CO3,
        "fCO2": solve.inorganic.fCO2_from_pH_CO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# DIC and HCO3
get_funcs_core[207] = {
    # pH is taken care of by opt_HCO3_root
    "CO3": solve.inorganic.CO3_from_pH_HCO3,
    "fCO2": solve.inorganic.fCO2_from_pH_HCO3,
    "alkalinity": solve.speciate.sum_alkalinity,
}
# pH and pCO2, fCO2, CO2, xCO2
for i in [304, 305, 308, 309]:
    get_funcs_core[i] = {
        "dic": solve.inorganic.dic_from_pH_fCO2,
        "HCO3": solve.inorganic.HCO3_from_pH_fCO2,
        "CO3": solve.inorganic.CO3_from_dic_pH,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# pH and CO3, omega
for i in [306, 310, 311]:
    get_funcs_core[i] = {
        "dic": solve.inorganic.dic_from_pH_CO3,
        "HCO3": solve.inorganic.HCO3_from_pH_CO3,
        "fCO2": solve.inorganic.fCO2_from_pH_CO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# pH and HCO3
get_funcs_core[307] = {
    "dic": solve.inorganic.dic_from_pH_HCO3,
    "CO3": solve.inorganic.CO3_from_pH_HCO3,
    "fCO2": solve.inorganic.fCO2_from_pH_HCO3,
    "alkalinity": solve.speciate.sum_alkalinity,
}
# CO3, omega and pCO2, fCO2, CO2, xCO2
for i in [406, 506, 608, 609, 410, 510, 810, 910, 411, 511, 811, 911]:
    get_funcs_core[i] = {
        "pH": solve.inorganic.pH_from_fCO2_CO3,
        "dic": solve.inorganic.dic_from_pH_CO3,
        "HCO3": solve.inorganic.HCO3_from_pH_CO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# HCO3 and pCO2, fCO2, CO2, xCO2
for i in [407, 507, 708, 709]:
    get_funcs_core[i] = {
        "pH": solve.inorganic.pH_from_fCO2_HCO3,
        "dic": solve.inorganic.dic_from_pH_HCO3,
        "CO3": solve.inorganic.CO3_from_pH_HCO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# CO3, omega and HCO3
for i in [607, 710, 711]:
    get_funcs_core[i] = {
        "pH": solve.inorganic.pH_from_CO3_HCO3,
        "fCO2": solve.inorganic.fCO2_from_CO3_HCO3,
        "dic": solve.inorganic.dic_from_pH_CO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }

# Add p-f-x-CO2 interconversions
for k, fc in get_funcs_core.items():
    if "fCO2" in fc or k in [5, 105, 205, 305, 506, 507, 510, 511]:
        fc.update(
            {
                "pCO2": convert.fCO2_to_pCO2,
                "CO2": convert.fCO2_to_CO2aq,
                "xCO2": convert.fCO2_to_xCO2,
            }
        )
    elif k in [4, 104, 204, 304, 406, 407, 410, 411]:
        fc.update(
            {
                "fCO2": convert.pCO2_to_fCO2,
                "CO2": convert.fCO2_to_CO2aq,
                "xCO2": convert.fCO2_to_xCO2,
            }
        )
    elif k in [8, 108, 208, 308, 608, 708, 810, 811]:
        fc.update(
            {
                "fCO2": convert.CO2aq_to_fCO2,
                "pCO2": convert.fCO2_to_pCO2,
                "xCO2": convert.fCO2_to_xCO2,
            }
        )
    elif k in [9, 109, 209, 309, 609, 709, 910, 911]:
        fc.update(
            {
                "fCO2": convert.xCO2_to_fCO2,
                "pCO2": convert.fCO2_to_pCO2,
                "CO2": convert.fCO2_to_CO2aq,
            }
        )

# Add CO3-saturation state interconversions
for k, fc in get_funcs_core.items():
    if "CO3" in fc or k in [6, 106, 206, 306, 406, 506, 607, 608, 609]:
        fc.update(
            {
                "saturation_aragonite": solubility.OA_from_CO3,
                "saturation_calcite": solubility.OC_from_CO3,
                "saturation_Mg_calcite": solubility.OMgCaCO3_from_CO3,
            }
        )
    elif k in [10, 110, 210, 310, 410, 510, 710, 810, 910]:
        fc.update(
            {
                "CO3": solubility.CO3_from_OC,
                "saturation_aragonite": solubility.OA_from_CO3,
                "saturation_Mg_calcite": solubility.OMgCaCO3_from_CO3,
            }
        )
    elif k in [11, 111, 211, 311, 411, 511, 711, 811, 911]:
        fc.update(
            {
                "CO3": solubility.CO3_from_OA,
                "saturation_calcite": solubility.OC_from_CO3,
                "saturation_Mg_calcite": solubility.OMgCaCO3_from_CO3,
            }
        )

# Add buffers and similar
for k, fc in get_funcs_core.items():
    if k > 100:
        fc.update(
            {
                "substrate_inhibitor_ratio": bio.substrate_inhibitor_ratio,
                "gamma_dic": buffers.gamma_dic,
                "gamma_alkalinity": buffers.gamma_alkalinity,
                "beta_dic": buffers.beta_dic,
                "beta_alkalinity": buffers.beta_alkalinity,
                "omega_dic": buffers.omega_dic,
                "omega_alkalinity": buffers.omega_alkalinity,
                "Q_isocap": buffers.Q_isocap,
                "Q_isocap_approx": buffers.Q_isocap_approx,
                "psi": buffers.psi,
                "revelle_factor": buffers.revelle_factor,
                "d_lnOmega__d_CO3": buffers.d_lnOmega__d_CO3,
                "d_CO3__d_pH__alkalinity": buffers.d_CO3__d_pH__alkalinity,
                "d_CO3__d_pH__dic": buffers.d_CO3__d_pH__dic,
                "d_dic__d_pH__alkalinity": buffers.d_dic__d_pH__alkalinity,
                "d_alkalinity__d_pH__dic": buffers.d_alkalinity__d_pH__dic,
                "d_lnCO2__d_pH__alkalinity": buffers.d_lnCO2__d_pH__alkalinity,
                "d_lnCO2__d_pH__dic": buffers.d_lnCO2__d_pH__dic,
                "d_alkalinity__d_pH__fCO2": buffers.d_alkalinity__d_pH__fCO2,
                "d_dic__d_pH__fCO2": buffers.d_dic__d_pH__fCO2,
                "d_fCO2__d_pH__alkalinity": buffers.d_fCO2__d_pH__alkalinity,
            }
        )

# Chemical speciation functions can only be used if there is a pH value
funcs_chemspec = {
    "H": lambda pH: 10**-pH,
    "H3PO4": solve.speciate.get_H3PO4,
    "H2PO4": solve.speciate.get_H2PO4,
    "HPO4": solve.speciate.get_HPO4,
    "PO4": solve.speciate.get_PO4,
    "BOH4": solve.speciate.get_BOH4,
    "BOH3": solve.speciate.get_BOH3,
    "OH": solve.speciate.get_OH,
    "H_free": solve.speciate.get_H_free,
    "H3SiO4": solve.speciate.get_H3SiO4,
    "H4SiO4": solve.speciate.get_H4SiO4,
    "HSO4": solve.speciate.get_HSO4,
    "SO4": solve.speciate.get_SO4,
    "HF": solve.speciate.get_HF,
    "F": solve.speciate.get_F,
    "NH3": solve.speciate.get_NH3,
    "NH4": solve.speciate.get_NH4,
    "H2S": solve.speciate.get_H2S,
    "HS": solve.speciate.get_HS,
    "HNO2": solve.speciate.get_HNO2,
    "NO2": solve.speciate.get_NO2,
}
for k, fc in get_funcs_core.items():
    if k > 100 or k == 3:
        fc.update(funcs_chemspec)

# Define functions for calculations that depend on opts:
# (unlike in previous versions, each opt may only affect one parameter)
get_funcs_opts = {}
get_coeffs_opts = {}
get_funcs_opts["opt_gas_constant"] = {
    1: dict(),
    2: dict(),
    3: dict(),
}
get_coeffs_opts["opt_gas_constant"] = {
    1: dict(gas_constant=constants.RGasConstant_DOEv2),
    2: dict(gas_constant=constants.RGasConstant_DOEv3),
    3: dict(gas_constant=constants.RGasConstant_CODATA2018),
}
get_funcs_opts["opt_factor_k_BOH3"] = {
    1: dict(factor_k_BOH3=equilibria.pcx.factor_k_BOH3_M79),
    2: dict(factor_k_BOH3=equilibria.pcx.factor_k_BOH3_GEOSECS),
}
get_funcs_opts["opt_factor_k_H2CO3"] = {
    1: dict(factor_k_H2CO3=equilibria.pcx.factor_k_H2CO3),
    2: dict(factor_k_H2CO3=equilibria.pcx.factor_k_H2CO3_GEOSECS),
    3: dict(factor_k_H2CO3=equilibria.pcx.factor_k_H2CO3_fw),
}
get_funcs_opts["opt_factor_k_HCO3"] = {
    1: dict(factor_k_HCO3=equilibria.pcx.factor_k_HCO3),
    2: dict(factor_k_HCO3=equilibria.pcx.factor_k_HCO3_GEOSECS),
    3: dict(factor_k_HCO3=equilibria.pcx.factor_k_HCO3_fw),
}
get_funcs_opts["opt_factor_k_H2O"] = {
    1: dict(factor_k_H2O=equilibria.pcx.factor_k_H2O),
    2: dict(factor_k_H2O=equilibria.pcx.factor_k_H2O_fw),
}
get_funcs_opts["opt_fH"] = {
    1: dict(fH=convert.fH_TWB82),
    2: dict(fH=convert.fH_PTBO87),
    3: dict(fH=lambda: 1.0),
}
get_funcs_opts["opt_k_carbonic"] = {
    1: dict(
        pk_H2CO3_total_1atm=equilibria.p1atm.pk_H2CO3_total_RRV93,
        pk_HCO3_total_1atm=equilibria.p1atm.pk_HCO3_total_RRV93,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_total_1atm, tot_to_sws_1atm: (
            pk_H2CO3_total_1atm + tot_to_sws_1atm
        ),
        pk_HCO3_sws_1atm=lambda pk_HCO3_total_1atm, tot_to_sws_1atm: (
            pk_HCO3_total_1atm + tot_to_sws_1atm
        ),
    ),
    2: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_GP89,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_GP89,
    ),
    3: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_H73_DM87,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_H73_DM87,
    ),
    4: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_MCHP73_DM87,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_MCHP73_DM87,
    ),
    5: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_HM_DM87,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_HM_DM87,
    ),
    6: dict(
        pk_H2CO3_nbs_1atm=equilibria.p1atm.pk_H2CO3_nbs_MCHP73,
        pk_HCO3_nbs_1atm=equilibria.p1atm.pk_HCO3_nbs_MCHP73,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_nbs_1atm, nbs_to_sws: (
            pk_H2CO3_nbs_1atm + nbs_to_sws
        ),
        pk_HCO3_sws_1atm=lambda pk_HCO3_nbs_1atm, nbs_to_sws: (
            pk_HCO3_nbs_1atm + nbs_to_sws
        ),
    ),
    # 7: same as 6; see note at end
    8: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_M79,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_M79,
    ),
    9: dict(
        pk_H2CO3_nbs_1atm=equilibria.p1atm.pk_H2CO3_nbs_CW98,
        pk_HCO3_nbs_1atm=equilibria.p1atm.pk_HCO3_nbs_CW98,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_nbs_1atm, nbs_to_sws: (
            pk_H2CO3_nbs_1atm + nbs_to_sws
        ),
        pk_HCO3_sws_1atm=lambda pk_HCO3_nbs_1atm, nbs_to_sws: (
            pk_HCO3_nbs_1atm + nbs_to_sws
        ),
    ),
    10: dict(
        pk_H2CO3_total_1atm=equilibria.p1atm.pk_H2CO3_total_LDK00,
        pk_HCO3_total_1atm=equilibria.p1atm.pk_HCO3_total_LDK00,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_total_1atm, tot_to_sws_1atm: (
            pk_H2CO3_total_1atm + tot_to_sws_1atm
        ),
        pk_HCO3_sws_1atm=lambda pk_HCO3_total_1atm, tot_to_sws_1atm: (
            pk_HCO3_total_1atm + tot_to_sws_1atm
        ),
    ),
    11: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_MM02,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_MM02,
    ),
    12: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_MPL02,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_MPL02,
    ),
    13: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_MGH06,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_MGH06,
    ),
    14: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_M10,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_M10,
    ),
    15: dict(
        pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_WMW14,
        pk_HCO3_sws_1atm=equilibria.p1atm.pk_HCO3_sws_WMW14,
    ),
    16: dict(
        pk_H2CO3_total_1atm=equilibria.p1atm.pk_H2CO3_total_SLH20,
        pk_HCO3_total_1atm=equilibria.p1atm.pk_HCO3_total_SLH20,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_total_1atm, tot_to_sws_1atm: (
            pk_H2CO3_total_1atm + tot_to_sws_1atm
        ),
        pk_HCO3_sws_1atm=lambda pk_HCO3_total_1atm, tot_to_sws_1atm: (
            pk_HCO3_total_1atm + tot_to_sws_1atm
        ),
    ),
    17: dict(
        # pk_H2CO3_sws_1atm=equilibria.p1atm.pk_H2CO3_sws_WMW14,
        # ^ although the above should work, it gives slightly different answers
        #   than he conversion below, and below is consistent with the MATLAB
        #   implementation
        pk_H2CO3_total_1atm=equilibria.p1atm.pk_H2CO3_total_WMW14,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_total_1atm, tot_to_sws_1atm: (
            pk_H2CO3_total_1atm + tot_to_sws_1atm
        ),
        pk_HCO3_total_1atm=equilibria.p1atm.pk_HCO3_total_SB21,
        pk_HCO3_sws_1atm=lambda pk_HCO3_total_1atm, tot_to_sws_1atm: (
            pk_HCO3_total_1atm + tot_to_sws_1atm
        ),
    ),
    18: dict(
        pk_H2CO3_total_1atm=equilibria.p1atm.pk_H2CO3_total_PLR18,
        pk_HCO3_total_1atm=equilibria.p1atm.pk_HCO3_total_PLR18,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_total_1atm, tot_to_sws_1atm: (
            pk_H2CO3_total_1atm + tot_to_sws_1atm
        ),
        pk_HCO3_sws_1atm=lambda pk_HCO3_total_1atm, tot_to_sws_1atm: (
            pk_HCO3_total_1atm + tot_to_sws_1atm
        ),
    ),
    19: dict(
        pk_H2CO3_total_1atm=equilibria.p1atm.pk_H2CO3_total_WMW14,
        pk_HCO3_total_1atm=equilibria.p1atm.pk_HCO3_total_MMB25,
        pk_H2CO3_sws_1atm=lambda pk_H2CO3_total_1atm, tot_to_sws_1atm: (
            pk_H2CO3_total_1atm + tot_to_sws_1atm
        ),
        pk_HCO3_sws_1atm=lambda pk_HCO3_total_1atm, tot_to_sws_1atm: (
            pk_HCO3_total_1atm + tot_to_sws_1atm
        ),
    ),
}
# For historical reasons, these are the same as each other (one also gets the
# Peng "correction", but that's handled elsewhere):
gfo = get_funcs_opts
get_funcs_opts["opt_k_carbonic"][7] = gfo["opt_k_carbonic"][6].copy()
get_coeffs_opts["opt_k_carbonic"] = {
    1: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_total_RRV93(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_total_RRV93(),
    ),
    2: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_GP89(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_GP89(),
    ),
    3: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_H73_DM87(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_H73_DM87(),
    ),
    4: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_MCHP73_DM87(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_MCHP73_DM87(),
    ),
    5: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_HM_DM87(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_HM_DM87(),
    ),
    6: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_nbs_MCHP73(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_nbs_MCHP73(),
    ),
    7: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_nbs_MCHP73(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_nbs_MCHP73(),
    ),
    8: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_M79(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_M79(),
    ),
    9: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_nbs_CW98(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_nbs_CW98(),
    ),
    10: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_total_LDK00(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_total_LDK00(),
    ),
    11: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_MM02(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_MM02(),
    ),
    12: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_MPL02(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_MPL02(),
    ),
    13: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_MGH06(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_MGH06(),
    ),
    14: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_M10(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_M10(),
    ),
    15: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_sws_WMW14(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_sws_WMW14(),
    ),
    16: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_total_SLH20(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_total_SLH20(),
    ),
    17: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_total_WMW14(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_total_SB21(),
    ),
    18: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_total_PLR18(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_total_PLR18(),
    ),
    19: dict(
        coeffs_pk_H2CO3=equilibria.p1atm.coeffs_pk_H2CO3_total_WMW14(),
        coeffs_pk_HCO3=equilibria.p1atm.coeffs_pk_HCO3_total_MMB25(),
    ),
}
get_funcs_opts["opt_k_phosphate"] = {
    1: dict(
        pk_H3PO4_sws_1atm=equilibria.p1atm.pk_H3PO4_sws_YM95,
        pk_H2PO4_sws_1atm=equilibria.p1atm.pk_H2PO4_sws_YM95,
        pk_HPO4_sws_1atm=equilibria.p1atm.pk_HPO4_sws_YM95,
    ),
    2: dict(
        pk_H3PO4_sws_1atm=equilibria.p1atm.pk_H3PO4_sws_KP67,
        pk_H2PO4_nbs_1atm=equilibria.p1atm.pk_H2PO4_nbs_KP67,
        pk_H2PO4_sws_1atm=lambda pk_H2PO4_nbs_1atm, nbs_to_sws: (
            pk_H2PO4_nbs_1atm + nbs_to_sws
        ),
        pk_HPO4_nbs_1atm=equilibria.p1atm.pk_HPO4_nbs_KP67,
        pk_HPO4_sws_1atm=lambda pk_HPO4_nbs_1atm, nbs_to_sws: (
            pk_HPO4_nbs_1atm + nbs_to_sws
        ),
    ),
}
get_coeffs_opts["opt_k_phosphate"] = {
    1: dict(
        coeffs_pk_H3PO4=equilibria.p1atm.coeffs_pk_H3PO4_sws_YM95(),
        coeffs_pk_H2PO4=equilibria.p1atm.coeffs_pk_H2PO4_sws_YM95(),
        coeffs_pk_HPO4=equilibria.p1atm.coeffs_pk_HPO4_sws_YM95(),
    ),
    2: dict(
        coeffs_pk_H3PO4=equilibria.p1atm.coeffs_pk_H3PO4_sws_KP67(),
        coeffs_pk_H2PO4=equilibria.p1atm.coeffs_pk_H2PO4_nbs_KP67(),
        coeffs_pk_HPO4=equilibria.p1atm.coeffs_pk_HPO4_nbs_KP67(),
    ),
}
get_funcs_opts["opt_k_BOH3"] = {
    1: dict(
        pk_BOH3_total_1atm=equilibria.p1atm.pk_BOH3_total_D90b,
        pk_BOH3_sws_1atm=lambda pk_BOH3_total_1atm, tot_to_sws_1atm: (
            pk_BOH3_total_1atm + tot_to_sws_1atm
        ),
    ),
    2: dict(
        pk_BOH3_nbs_1atm=equilibria.p1atm.pk_BOH3_nbs_LTB69,
        pk_BOH3_sws_1atm=lambda pk_BOH3_nbs_1atm, nbs_to_sws: (
            pk_BOH3_nbs_1atm + nbs_to_sws
        ),
    ),
    3: dict(
        pk_BOH3_total_1atm=equilibria.p1atm.pk_BOH3_total_MMB26,
        pk_BOH3_sws_1atm=lambda pk_BOH3_total_1atm, tot_to_sws_1atm: (
            pk_BOH3_total_1atm + tot_to_sws_1atm
        ),
    ),
}
get_coeffs_opts["opt_k_BOH3"] = {
    1: dict(coeffs_pk_BOH3=equilibria.p1atm.coeffs_pk_BOH3_total_D90b()),
    2: dict(coeffs_pk_BOH3=equilibria.p1atm.coeffs_pk_BOH3_nbs_LTB69()),
    3: dict(coeffs_pk_BOH3=equilibria.p1atm.coeffs_pk_BOH3_total_MMB26()),
}
get_funcs_opts["opt_k_H2O"] = {
    1: dict(pk_H2O_sws_1atm=equilibria.p1atm.pk_H2O_sws_M95),
    2: dict(pk_H2O_sws_1atm=equilibria.p1atm.pk_H2O_sws_M79),
    3: dict(pk_H2O_sws_1atm=equilibria.p1atm.pk_H2O_sws_HO58_M79),
}
get_coeffs_opts["opt_k_H2O"] = {
    1: dict(coeffs_pk_H2O=equilibria.p1atm.coeffs_pk_H2O_sws_M95()),
    2: dict(coeffs_pk_H2O=equilibria.p1atm.coeffs_pk_H2O_sws_M79()),
    3: dict(coeffs_pk_H2O=equilibria.p1atm.coeffs_pk_H2O_sws_HO58_M79()),
}
get_funcs_opts["opt_k_HF"] = {
    1: dict(pk_HF_free_1atm=equilibria.p1atm.pk_HF_free_DR79),
    2: dict(pk_HF_free_1atm=equilibria.p1atm.pk_HF_free_PF87),
}
get_coeffs_opts["opt_k_HF"] = {
    1: dict(coeffs_pk_HF=equilibria.p1atm.coeffs_pk_HF_free_DR79()),
    2: dict(coeffs_pk_HF=equilibria.p1atm.coeffs_pk_HF_free_PF87()),
}
get_funcs_opts["opt_k_HSO4"] = {
    1: dict(pk_HSO4_free_1atm=equilibria.p1atm.pk_HSO4_free_D90a),
    2: dict(pk_HSO4_free_1atm=equilibria.p1atm.pk_HSO4_free_KRCB77),
    3: dict(pk_HSO4_free_1atm=equilibria.p1atm.pk_HSO4_free_WM13),
}
get_coeffs_opts["opt_k_HSO4"] = {
    1: dict(coeffs_pk_HSO4=equilibria.p1atm.coeffs_pk_HSO4_free_D90a()),
    2: dict(coeffs_pk_HSO4=equilibria.p1atm.coeffs_pk_HSO4_free_KRCB77()),
    3: dict(coeffs_pk_HSO4=equilibria.p1atm.coeffs_pk_HSO4_free_WM13()),
}
get_funcs_opts["opt_k_NH3"] = {
    1: dict(
        pk_NH3_total_1atm=equilibria.p1atm.pk_NH3_total_CW95,
        pk_NH3_sws_1atm=lambda pk_NH3_total_1atm, tot_to_sws_1atm: (
            pk_NH3_total_1atm + tot_to_sws_1atm
        ),
    ),
    2: dict(pk_NH3_sws_1atm=equilibria.p1atm.pk_NH3_sws_YM95),
}
get_coeffs_opts["opt_k_NH3"] = {
    1: dict(coeffs_pk_NH3=equilibria.p1atm.coeffs_pk_NH3_total_CW95()),
    2: dict(coeffs_pk_NH3=equilibria.p1atm.coeffs_pk_NH3_sws_YM95()),
}
get_funcs_opts["opt_k_Si"] = {
    1: dict(pk_Si_sws_1atm=equilibria.p1atm.pk_Si_sws_YM95),
    2: dict(
        pk_Si_nbs_1atm=equilibria.p1atm.pk_Si_nbs_SMB64,
        pk_Si_sws_1atm=lambda pk_Si_nbs_1atm, nbs_to_sws: (
            pk_Si_nbs_1atm + nbs_to_sws
        ),
    ),
}
get_coeffs_opts["opt_k_Si"] = {
    1: dict(coeffs_pk_Si=equilibria.p1atm.coeffs_pk_Si_sws_YM95()),
    2: dict(coeffs_pk_Si=equilibria.p1atm.coeffs_pk_Si_nbs_SMB64()),
}
get_funcs_opts["opt_k_HNO2"] = {
    1: dict(
        pk_HNO2_total_1atm=equilibria.p1atm.pk_HNO2_total_BBWB24,
        pk_HNO2_sws_1atm=lambda pk_HNO2_total_1atm, tot_to_sws_1atm: (
            pk_HNO2_total_1atm + tot_to_sws_1atm
        ),
    ),
    2: dict(
        pk_HNO2_nbs_1atm=equilibria.p1atm.pk_HNO2_nbs_BBWB24_freshwater,
        pk_HNO2_sws_1atm=lambda pk_HNO2_nbs_1atm, nbs_to_sws: (
            pk_HNO2_nbs_1atm + nbs_to_sws
        ),
    ),
}
get_coeffs_opts["opt_k_HNO2"] = {
    1: dict(coeffs_pk_HNO2=equilibria.p1atm.coeffs_pk_HNO2_total_BBWB24()),
    2: dict(
        coeffs_pk_HNO2=equilibria.p1atm.coeffs_pk_HNO2_nbs_BBWB24_freshwater()
    ),
}
get_funcs_opts["opt_pH_scale"] = {
    1: dict(  # total
        sws_to_opt=convert.pH_sws_to_tot,
        opt_to_free=convert.pH_tot_to_free,
        opt_to_sws=convert.pH_tot_to_sws,
        opt_to_nbs=convert.pH_tot_to_nbs,
    ),
    2: dict(  # sws
        sws_to_opt=lambda: 0,
        opt_to_free=convert.pH_sws_to_free,
        opt_to_tot=convert.pH_sws_to_tot,
        opt_to_nbs=convert.pH_sws_to_nbs,
    ),
    3: dict(  # free
        sws_to_opt=convert.pH_sws_to_free,
        opt_to_free=lambda: 0,
        opt_to_tot=convert.pH_free_to_tot,
        opt_to_sws=convert.pH_free_to_sws,
        opt_to_nbs=convert.pH_free_to_nbs,
    ),
    4: dict(  # nbs
        sws_to_opt=convert.pH_sws_to_nbs,
        opt_to_free=convert.pH_nbs_to_free,
        opt_to_tot=convert.pH_nbs_to_tot,
        opt_to_sws=convert.pH_nbs_to_sws,
    ),
}
for o, funcs in get_funcs_opts["opt_pH_scale"].items():
    if o == 1:
        funcs.update(dict(pH_total=lambda pH: pH))
    if o == 2:
        funcs.update(dict(pH_sws=lambda pH: pH))
    if o == 3:
        funcs.update(dict(pH_free=lambda pH: pH))
    if o == 4:
        funcs.update(dict(pH_nbs=lambda pH: pH))
    if o in [2, 3, 4]:
        funcs.update(dict(pH_total=lambda pH, opt_to_tot: pH + opt_to_tot))
    if o in [1, 3, 4]:
        funcs.update(dict(pH_sws=lambda pH, opt_to_sws: pH + opt_to_sws))
    if o in [1, 2, 4]:
        funcs.update(dict(pH_free=lambda pH, opt_to_free: pH + opt_to_free))
    if o in [1, 2, 3]:
        funcs.update(dict(pH_nbs=lambda pH, opt_to_nbs: pH + opt_to_nbs))
get_coeffs_opts["opt_total_borate"] = {
    1: dict(coeffs_total_borate=salts.coeffs_total_borate_U74()),
    2: dict(coeffs_total_borate=salts.coeffs_total_borate_LKB10()),
    3: dict(coeffs_total_borate=salts.coeffs_total_borate_KSK18()),
    4: dict(coeffs_total_borate=salts.coeffs_total_borate_C65()),
}
get_funcs_opts["opt_total_borate"] = {
    1: dict(total_borate=salts.total_borate_U74),
    2: dict(total_borate=salts.total_borate_LKB10),
    3: dict(total_borate=salts.total_borate_KSK18),
    4: dict(total_borate=salts.total_borate_C65),
}
get_funcs_opts["opt_Ca"] = {
    1: dict(Ca=salts.Ca_RT67),
    2: dict(Ca=salts.Ca_C65),
}
get_coeffs_opts["opt_Ca"] = {
    1: dict(coeffs_Ca=salts.coeffs_Ca_RT67()),
    2: dict(coeffs_Ca=salts.coeffs_Ca_C65()),
}
get_funcs_opts["opt_fugacity_factor"] = {
    1: dict(fugacity_factor=gas.fugacity_factor),
    2: dict(fugacity_factor=lambda: 1.0),  # for GEOSECS
}
get_funcs_opts["opt_HCO3_root"] = {  # only added if icase == 207
    1: dict(pH=solve.inorganic.pH_from_dic_HCO3_lo),
    2: dict(pH=solve.inorganic.pH_from_dic_HCO3_hi),  # for typical seawater
}
get_coeffs_opts["opt_k_calcite"] = {
    1: dict(coeffs_pk_calcite=solubility.coeffs_pk_calcite_M83()),
    2: dict(coeffs_pk_calcite=solubility.coeffs_pk_calcite_I75()),
}
get_funcs_opts["opt_k_calcite"] = {
    1: dict(pk_calcite=solubility.pk_calcite_M83),
    2: dict(pk_calcite=solubility.pk_calcite_I75),  # for GEOSECS
}
get_coeffs_opts["opt_k_aragonite"] = {
    1: dict(coeffs_pk_aragonite=solubility.coeffs_pk_aragonite_M83()),
    2: dict(coeffs_pk_aragonite=solubility.coeffs_pk_aragonite_GEOSECS()),
}
get_funcs_opts["opt_k_aragonite"] = {
    1: dict(pk_aragonite=solubility.pk_aragonite_M83),
    2: dict(pk_aragonite=solubility.pk_aragonite_GEOSECS),  # for GEOSECS
}
get_funcs_opts["opt_Mg_calcite_type"] = {
    1: dict(
        pkt_Mg_calcite_25C_1atm=solubility.get_pkt_Mg_calcite_25C_1atm_minprep
    ),
    2: dict(
        pkt_Mg_calcite_25C_1atm=solubility.get_pkt_Mg_calcite_25C_1atm_biogenic
    ),
    3: dict(
        pkt_Mg_calcite_25C_1atm=solubility.get_pkt_Mg_calcite_25C_1atm_synthetic
    ),
}
get_funcs_opts["opt_Mg_calcite_kt_Tdep"] = {
    1: dict(pkt_Mg_calcite_1atm=solubility.get_pkt_Mg_calcite_1atm_idealmix),
    2: dict(pkt_Mg_calcite_1atm=solubility.get_pkt_Mg_calcite_1atm_PB82),
    3: dict(pkt_Mg_calcite_1atm=solubility.get_pkt_Mg_calcite_1atm_vantHoff),
}


def icase_to_params(icase):
    if icase > 100:
        p1 = int(np.floor(icase / 100))
        p2 = int(icase - p1 * 100)
        par1 = parameters_core[p1 - 1]
        par2 = parameters_core[p2 - 1]
        return par1, par2
    elif icase > 0:
        return [parameters_core[icase - 1]]


def make_positional(get_value_of):
    assert hasattr(get_value_of, "args_list")

    def func_positional(*args):
        kwargs = {k: v for k, v in zip(get_value_of.args_list, args)}
        return get_value_of(**kwargs)

    func_positional.__doc__ = (
        get_value_of.__doc__.replace("kwargs", "args")
        .replace("dict", "tuple")
        .replace("Key-value pairs for", "Values of")
    )
    func_positional.args_list = get_value_of.args_list
    return func_positional


# DO NOT CHANGE THE ORDER OF THE ITEMS IN THIS TUPLE!!!
parameters_core = (
    "alkalinity",  # 1
    "dic",  # 2
    "pH",  # 3
    "pCO2",  # 4
    "fCO2",  # 5
    "CO3",  # 6
    "HCO3",  # 7
    "CO2",  # 8
    "xCO2",  # 9
    "saturation_calcite",  # 10
    "saturation_aragonite",  # 11
)

values_default = {
    "Mg_fraction": 0.0,
    "pressure_atmosphere": 1.0,  # atm
    "pressure": 0.0,  # dbar
    "salinity": 35.0,
    "temperature": 25.0,  # °C
    "total_ammonia": 0.0,  # µmol/kg-sw
    "total_phosphate": 0.0,  # µmol/kg-sw
    "total_silicate": 0.0,  # µmol/kg-sw
    "total_sulfide": 0.0,  # µmol/kg-sw
    "total_nitrite": 0.0,  # µmol/kg-sw
    "coeffs_pk_CO2": equilibria.p1atm.coeffs_pk_CO2_W74(),
    "coeffs_pk_H2S": equilibria.p1atm.coeffs_pk_H2S_total_YM95(),
    "coeffs_total_fluoride": salts.coeffs_total_fluoride_R65(),
    "coeffs_total_sulfate": salts.coeffs_total_sulfate_MR66(),
    "coeffs_Mg": salts.coeffs_Mg_reference_composition(),
}

opts_default = {
    "opt_Ca": 1,
    "opt_factor_k_BOH3": 1,
    "opt_factor_k_H2CO3": 1,
    "opt_factor_k_H2O": 1,
    "opt_factor_k_HCO3": 1,
    # "opt_fCO2_temperature": 1,
    "opt_fH": 1,
    "opt_fugacity_factor": 1,
    "opt_gas_constant": 3,
    "opt_HCO3_root": 2,
    "opt_k_aragonite": 1,
    "opt_k_BOH3": 1,
    "opt_k_calcite": 1,
    "opt_k_carbonic": 10,
    "opt_k_H2O": 1,
    "opt_k_HF": 1,
    "opt_k_HSO4": 1,
    "opt_k_NH3": 1,
    "opt_k_phosphate": 1,
    "opt_k_Si": 1,
    "opt_k_HNO2": 1,
    "opt_Mg_calcite_kt_Tdep": 1,
    "opt_Mg_calcite_type": 2,
    "opt_pH_scale": 1,
    "opt_total_borate": 1,
}

# Parameters that do not change between input and output conditions
condition_independent = (
    "alkalinity",
    "Ca",
    "dic",
    "gas_constant",
    "ionic_strength",
    "Mg_fraction",
    "pressure_atmosphere",
    "salinity",
    "total_ammonia",
    "total_borate",
    "total_fluoride",
    "total_phosphate",
    "total_silicate",
    "total_sulfate",
    "total_sulfide",
    "total_nitrite",
    "coeffs_pk_CO2",
    "coeffs_pk_H2S",
    "coeffs_pk_HF",
    "coeffs_pk_H2O",
    "coeffs_pk_HSO4",
    "coeffs_pk_BOH3",
    "coeffs_pk_NH3",
    "coeffs_pk_Si",
    "coeffs_pk_HNO2",
    "coeffs_pk_H2CO3",
    "coeffs_pk_HCO3",
    "coeffs_pk_H3PO4",
    "coeffs_pk_H2PO4",
    "coeffs_pk_HPO4",
    "coeffs_pk_calcite",
    "coeffs_pk_aragonite",
    "coeffs_total_borate",
    "coeffs_total_fluoride",
    "coeffs_total_sulfate",
    "coeffs_Mg",
    "coeffs_Ca",
)

# Define labels for parameter plotting
# NOTE This dict's keys are also used as the basis for the shortcuts,
#      so every parameter that isn't all lowercase should appear here.
#      (except those with __pre suffixes - they're added automatically).
node_labels = {
    "acf_Ca": r"$\gamma_{\mathrm{Ca}^{2+}}$",
    "acf_CO3": r"$\gamma_{\mathrm{CO}_3^{2–}}$",
    "acf_Mg": r"$\gamma_{\mathrm{Mg}^{2+}}$",
    "alkalinity": r"$A_\mathrm{T}$",
    "aq": "$a_q$",
    "beta_alkalinity": r"$\beta_{A_\mathrm{T}}$",
    "beta_dic": r"$\beta_{C_\mathrm{T}}$",
    "bh": "$b_h$",
    "bl": "$b_l$",
    "BOH3": r"$[\mathrm{B(OH)}_3]$",
    "BOH4": r"$[\mathrm{B(OH)}_4^–]$",
    "bq": "$b_q$",
    "Ca": r"$[\mathrm{Ca}^{2+}]$",
    "CO2": r"$[\mathrm{CO}_2(\mathrm{aq})]$",
    "CO3": "[CO$_3^{2–}$]",
    "d_lnOmega__d_CO3": "dlnΩ/d[CO$_3^{2-}$]",
    "dic": r"$T_\mathrm{C}$",
    "exp_upsilon": r"$e^\Upsilon$",
    "F": r"$[\mathrm{F}^-]$",
    "factor_k_BOH3": r"$P_\mathrm{B}$",
    "factor_k_CO2": "$P_0$",
    "factor_k_H2CO3": "$P_1$",
    "factor_k_H2O": r"$P_w$",
    "factor_k_H2PO4": r"$P_\mathrm{P2}$",
    "factor_k_H2S": r"$P_\mathrm{H_2S}$",
    "factor_k_H3PO4": r"$P_\mathrm{P1}$",
    "factor_k_HCO3": "$P_2$",
    "factor_k_HF": r"$P_\mathrm{HF}$",
    "factor_k_HNO2": r"$P_{\mathrm{HNO}_2}$",
    "factor_k_HPO4": r"$P_\mathrm{P3}$",
    "factor_k_HSO4": r"$P_\mathrm{SO_4}$",
    "factor_k_NH3": r"$P_\mathrm{NH_3}$",
    "factor_k_Si": r"$P_\mathrm{Si}$",
    "fCO2": "fCO$_2$",
    "fH": r"$\gamma_\mathrm{H}$(NBS)",
    "fugacity_factor": "$ƒ$",
    "gamma_alkalinity": r"$\gamma_{A_\mathrm{T}}$",
    "gamma_dic": r"$\gamma_{C_\mathrm{T}}$",
    "gas_constant": "$R$",
    "H_free": r"$[\mathrm{H}^+]^\mathrm{F}$",
    "H": r"$[\mathrm{H}^+]^*$",
    "H2PO4": r"$[\mathrm{H}_2\mathrm{PO}_4^–]$",
    "H2S": r"$[\mathrm{H_2S}]$",
    "H3PO4": r"$[\mathrm{H}_3\mathrm{PO}_4]$",
    "H3SiO4": r"$[\mathrm{H}_3\mathrm{SiO}_4^–]$",
    "H4SiO4": r"$[\mathrm{H}_4\mathrm{SiO}_4]$",
    "HCO3": "[HCO$_3^–$]",
    "HF": "[HF]",
    "HPO4": r"$[\mathrm{HPO}_4^{2–}]$",
    "HNO2": r"$[\mathrm{HNO}_2]$",
    "HS": r"$[\mathrm{HS}^–]$",
    "HSO4": r"$[\mathrm{HSO}_4^–]$",
    "ionic_strength": "$I$",
    "NO2": r"$[\mathrm{NO}_2^-]$",
    "pk_aragonite": r"p$K_\mathrm{a}^*$",
    "pk_BOH3_sws_1atm": r"p$K_\mathrm{B}^\mathrm{S0}$",
    "pk_BOH3_sws": r"p$K_\mathrm{B}^\mathrm{S}$",
    "pk_BOH3_total_1atm": r"p$K_\mathrm{B}^\mathrm{T0}$",
    "pk_BOH3": r"p$K_\mathrm{B}^*$",
    "pk_calcite": r"p$K_\mathrm{c}^*$",
    "pk_CO2_1atm": "p$K_0′^0$",
    "pk_CO2": "p$K_0′$",
    "pk_H2CO3_sws_1atm": r"p$K_1^\mathrm{S0}$",
    "pk_H2CO3_sws": r"p$K_1^\mathrm{S}$",
    "pk_H2CO3_total_1atm": r"p$K_1^\mathrm{T0}$",
    "pk_H2CO3": "p$K_1^*$",
    "pk_H2O_sws_1atm": r"p$K_w^\mathrm{S0}$",
    "pk_H2O_sws": r"p$K_w^\mathrm{S}$",
    "pk_H2O": "p$K_w^*$",
    "pk_H2PO4_sws_1atm": r"p$K_\mathrm{P2}^\mathrm{S0}$",
    "pk_H2PO4_sws": r"p$K_\mathrm{P2}^\mathrm{S}$",
    "pk_H2PO4": r"p$K_\mathrm{P2}^*$",
    "pk_H2S_sws_1atm": r"p$K_\mathrm{H_2S}^\mathrm{S0}$",
    "pk_H2S_sws": r"p$K_\mathrm{H_2S}^\mathrm{S}$",
    "pk_H2S_total_1atm": r"p$K_\mathrm{H_2S}^\mathrm{T0}$",
    "pk_H2S": r"p$K_\mathrm{H_2S}^*$",
    "pk_H3PO4_sws_1atm": r"p$K_\mathrm{P1}^\mathrm{S0}$",
    "pk_H3PO4_sws": r"p$K_\mathrm{P1}^\mathrm{S}$",
    "pk_H3PO4": r"p$K_\mathrm{P1}^*$",
    "pk_HCO3_sws_1atm": r"p$K_2^\mathrm{S0}$",
    "pk_HCO3_sws": r"p$K_2^\mathrm{S}$",
    "pk_HCO3_total_1atm": r"p$K_2^\mathrm{T0}$",
    "pk_HCO3": "p$K_2^*$",
    "pk_HF_free_1atm": r"p$K_\mathrm{HF}^\mathrm{F0}$",
    "pk_HF_free": r"p$K_\mathrm{HF}^\mathrm{F}$",
    "pk_HNO2_sws_1atm": r"p$K_\mathrm{HNO_2}^\mathrm{S0}$",
    "pk_HNO2_sws": r"p$K_\mathrm{HNO_2}^\mathrm{S}$",
    "pk_HNO2_total_1atm": r"p$K_\mathrm{HNO_2}^\mathrm{T0}$",
    "pk_HNO2": r"p$K_\mathrm{HNO_2}^*$",
    "pk_HPO4_sws_1atm": r"p$K_\mathrm{P3}^\mathrm{S0}$",
    "pk_HPO4_sws": r"p$K_\mathrm{P3}^\mathrm{S}$",
    "pk_HPO4": r"p$K_\mathrm{P3}^*$",
    "pk_HSO4_free_1atm": r"p$K_\mathrm{HSO_4}^\mathrm{F0}$",
    "pk_HSO4_free": r"p$K_\mathrm{HSO_4}^\mathrm{F}$",
    "pk_NH3_sws_1atm": r"p$K_\mathrm{NH_3}^\mathrm{S0}$",
    "pk_NH3_sws": r"p$K_\mathrm{NH_3}^\mathrm{S}$",
    "pk_NH3_total_1atm": r"p$K_\mathrm{NH_3}^\mathrm{T0}$",
    "pk_NH3": r"p$K_\mathrm{NH_3}^*$",
    "pk_Si_sws_1atm": r"p$K_\mathrm{Si}^\mathrm{S0}$",
    "pk_Si_sws": r"p$K_\mathrm{Si}^\mathrm{S}$",
    "pk_Si": r"p$K_\mathrm{Si}^*$",
    "Mg_fraction": "Mg fraction",
    "Mg": r"$[\mathrm{Mg}^{2+}]$",
    "NH3": r"$[\mathrm{NH}_3]$",
    "NH4": r"$[\mathrm{NH}_4^+]$",
    "OH": r"$[\mathrm{OH}^–]$",
    "omega_alkalinity": r"$\omega_{A_\mathrm{T}}$",
    "omega_dic": r"$\omega_{C_\mathrm{T}}$",
    "pCO2": r"$p\mathrm{CO}_2$",
    "pH": "pH",
    "pH_free": r"pH$_\mathrm{F}$",
    "pH_nbs": r"pH$_\mathrm{N}$",
    "pH_sws": r"pH$_\mathrm{S}$",
    "pH_total": r"pH$_\mathrm{T}$",
    "PO4": r"$[\mathrm{PO}_4^{3–}]$",
    "pressure_atmosphere": r"$p_\mathrm{atm}$",
    "pressure": "$p$",
    "psi": r"$\psi$",
    "Q_isocap_approx": "$Q_x$",
    "Q_isocap": "$Q$",
    "revelle_factor": r"$R_\mathrm{F}$",
    "salinity": "$S$",
    "saturation_aragonite": r"$Ω_\mathrm{a}$",
    "saturation_calcite": r"$Ω_\mathrm{c}$",
    "saturation_Mg_calcite": r"$Ω_\mathrm{c(Mg)}$",
    "SO4": r"$[\mathrm{SO}_4^{2–}]$",
    "substrate_inhibitor_ratio": "SIR",
    "temperature": "$t$",
    "total_ammonia": r"$T_\mathrm{NH_3}$",
    "total_borate": r"$T_\mathrm{B}$",
    "total_fluoride": r"$T_\mathrm{F}$",
    "total_nitrite": r"$T_\mathrm{HNO_2}$",
    "total_phosphate": r"$T_\mathrm{P}$",
    "total_silicate": r"$T_\mathrm{Si}$",
    "total_sulfate": r"$T_\mathrm{SO_4}$",
    "total_sulfide": r"$T_\mathrm{H_2S}$",
    "upsilon": r"$\upsilon$",
    "vp_factor": "$v$",
    "xCO2": r"$x\mathrm{CO}_2$",
    # pH scale conversions
    "free_to_opt": r"$_\mathrm{F}Y$",
    "free_to_sws_1atm": r"$_\mathrm{F}^\mathrm{S}Y^0$",
    "nbs_to_free": r"$_\mathrm{N}^\mathrm{F}Y$",
    "nbs_to_opt": r"$_\mathrm{N}Y$",
    "nbs_to_sws": r"$_\mathrm{N}^\mathrm{S}Y$",
    "nbs_to_tot": r"$_\mathrm{N}^\mathrm{T}Y$",
    "opt_to_free": r"$^\mathrm{F}Y$",
    "opt_to_nbs": r"$^\mathrm{N}Y$",
    "opt_to_sws": r"$^\mathrm{S}Y$",
    "opt_to_tot": r"$^\mathrm{T}Y$",
    "sws_to_free": r"$_\mathrm{S}^\mathrm{F}Y$",
    "sws_to_nbs": r"$_\mathrm{S}^\mathrm{N}Y$",
    "sws_to_opt": r"$_\mathrm{S}Y$",
    "sws_to_tot": r"$_\mathrm{S}^\mathrm{T}Y$",
    "tot_to_free": r"$_\mathrm{T}^\mathrm{F}Y$",
    "tot_to_nbs": r"$_\mathrm{T}^\mathrm{N}Y$",
    "tot_to_opt": r"$_\mathrm{T}Y$",
    "tot_to_sws_1atm": r"$_\mathrm{T}^\mathrm{S}Y^0$",
    "tot_to_sws": r"$_\mathrm{T}^\mathrm{S}Y$",
    # Coefficients
    "coeffs_bh": "$c$[$b_h$]",
    # TODO below not formatted
    "pk_Mg_calcite_1atm": "pk_Mg_calcite_1atm",
    "pkt_Mg_calcite_1atm": "pkt_Mg_calcite_1atm",
    "pk_Mg_calcite": "pk_Mg_calcite",
    "pkt_Mg_calcite_25C_1atm": "pkt_Mg_calcite_25C_1atm",
    "d_dic__d_pH__alkalinity": "d_dic__d_pH__alkalinity",
    "d_lnCO2__d_pH__alkalinity": "d_lnCO2__d_pH__alkalinity",
    "d_alkalinity__d_pH__dic": "d_alkalinity__d_pH__dic",
    "d_lnCO2__d_pH__dic": "d_lnCO2__d_pH__dic",
    "d_CO3__d_pH__alkalinity": "d_CO3__d_pH__alkalinity",
    "d_CO3__d_pH__dic": "d_CO3__d_pH__dic",
    "d_alkalinity__d_pH__fCO2": "d_alkalinity__d_pH__fCO2",
    "d_dic__d_pH__fCO2": "d_dic__d_pH__fCO2",
    "d_fCO2__d_pH__alkalinity": "d_fCO2__d_pH__alkalinity",
    "d_fCO2__d_pH__dic": "d_fCO2__d_pH__dic",
    "coeffs_pk_CO2": "coeffs_pk_CO2",
    "coeffs_pk_H2S": "coeffs_pk_H2S",
    "coeffs_pk_HF": "coeffs_pk_HF",
    "coeffs_pk_H2O": "coeffs_pk_H2O",
    "coeffs_pk_HSO4": "coeffs_pk_HSO4",
    "coeffs_pk_BOH3": "coeffs_pk_BOH3",
    "coeffs_pk_NH3": "coeffs_pk_NH3",
    "coeffs_pk_Si": "coeffs_pk_Si",
    "coeffs_pk_HNO2": "coeffs_pk_HNO2",
    "coeffs_pk_H2CO3": "coeffs_pk_H2CO3",
    "coeffs_pk_HCO3": "coeffs_pk_HCO3",
    "coeffs_pk_H3PO4": "coeffs_pk_H3PO4",
    "coeffs_pk_H2PO4": "coeffs_pk_H2PO4",
    "coeffs_pk_HPO4": "coeffs_pk_HPO4",
    "coeffs_pk_calcite": "coeffs_pk_calcite",
    "coeffs_pk_aragonite": "coeffs_pk_aragonite",
    "coeffs_total_borate": "coeffs_total_borate",
    "coeffs_total_fluoride": "coeffs_total_fluoride",
    "coeffs_total_sulfate": "coeffs_total_sulfate",
    "coeffs_Mg": "coeffs_Mg",
    "coeffs_Ca": "coeffs_Ca",
}
node_labels.update(
    {
        k + "__pre": r"$^\pi$" + v
        for k, v in node_labels.items()
        if k not in condition_independent
    }
)

# This is the set of parameters that will NOT be stored internally when
# store_steps == 1
exclude_on_store_steps_1 = {
    "factor_k_BOH3",
    "factor_k_CO2",
    "factor_k_H2CO3",
    "factor_k_H2O",
    "factor_k_H2PO4",
    "factor_k_H2S",
    "factor_k_H3PO4",
    "factor_k_HCO3",
    "factor_k_HF",
    "factor_k_HNO2",
    "factor_k_HPO4",
    "factor_k_HSO4",
    "factor_k_NH3",
    "factor_k_Si",
    "free_to_sws_1atm",
    "nbs_to_opt",
    "opt_to_free",
    "opt_to_nbs",
    "opt_to_sws",
    "pk_BOH3_sws_1atm",
    "pk_BOH3_sws",
    "pk_BOH3_total_1atm",
    "pk_CO2_1atm",
    "pk_H2CO3_sws_1atm",
    "pk_H2CO3_sws",
    "pk_H2CO3_total_1atm",
    "pk_H2O_sws_1atm",
    "pk_H2O_sws",
    "pk_H2PO4_sws_1atm",
    "pk_H2PO4_sws",
    "pk_H2S_sws_1atm",
    "pk_H2S_sws",
    "pk_H2S_total_1atm",
    "pk_H3PO4_sws_1atm",
    "pk_H3PO4_sws",
    "pk_HCO3_sws_1atm",
    "pk_HCO3_sws",
    "pk_HCO3_total_1atm",
    "pk_HF_free_1atm",
    "pk_HNO2_sws_1atm",
    "pk_HNO2_sws",
    "pk_HNO2_total_1atm",
    "pk_HPO4_sws_1atm",
    "pk_HPO4_sws",
    "pk_HSO4_free_1atm",
    "pk_Mg_calcite_1atm",
    "pk_NH3_sws_1atm",
    "pk_NH3_sws",
    "pk_NH3_total_1atm",
    "pk_Si_sws_1atm",
    "pk_Si_sws",
    "pkt_Mg_calcite_1atm",
    "pkt_Mg_calcite_25C_1atm",
    "sws_to_opt",
    "tot_to_opt",
    "tot_to_sws_1atm",
}

# Define shortcuts, the keys for which must all be lowercase
shortcuts = {k.lower(): k for k in node_labels if k.lower() != k}
shortcuts.update({k.lower(): k for k in opts_default if k.lower() != k})
shortcuts.update(
    {
        "tco2": "dic",
        "talk": "alkalinity",
        "alk": "alkalinity",
        "ta": "alkalinity",
        "ss_calc": "saturation_calcite",
        "ss_arag": "saturation_aragonite",
        "oc": "saturation_calcite",
        "oa": "saturation_aragonite",
        "ammonia": "total_ammonia",
        "borate": "total_borate",
        "fluoride": "total_fluoride",
        "nitrite": "total_nitrite",
        "phosphate": "total_phosphate",
        "silicate": "total_silicate",
        "sulfate": "total_sulfate",
        "sulfide": "total_sulfide",
        "tnh3": "total_ammonia",
        "tb": "total_borate",
        "tf": "total_fluoride",
        "tno2": "total_nitrite",
        "tp": "total_phosphate",
        "tsi": "total_silicate",
        "tso4": "total_sulfate",
        "th2s": "total_sulfide",
        "sir": "substrate_inhibitor_ratio",
        "pk0": "pk_CO2",
        "pk1": "pk_H2CO3",
        "pk2": "pk_HCO3",
        "pkw": "pk_H2O",
        "pkb": "pk_BOH3",
        "method_fco2": "method_fCO2",
        "which_fco2_insitu": "which_fCO2_insitu",
        "sal": "salinity",
        "temp": "temperature",
        "pres": "pressure",
        "s": "salinity",
        "t": "temperature",
        "p": "pressure",
        "revelle": "revelle_factor",
        "q": "Q_isocap",
    }
)
# Add any missing shortcuts
for k in get_funcs:
    if k != k.lower() and k.lower() not in shortcuts:
        shortcuts[k.lower()] = k
for v in get_funcs_core.values():
    for l in v:
        if l != l.lower() and l.lower() not in shortcuts:
            shortcuts[l.lower()] = l
for k in funcs_chemspec:
    if k != k.lower() and k.lower() not in shortcuts:
        shortcuts[k.lower()] = k
for k, v in get_coeffs_opts.items():
    if k != k.lower() and k.lower() not in shortcuts:
        shortcuts[k.lower()] = k
    for w in v.values():
        for m in w:
            if m != m.lower() and m.lower() not in shortcuts:
                shortcuts[m.lower()] = m
for k, v in get_funcs_opts.items():
    if k != k.lower() and k.lower() not in shortcuts:
        shortcuts[k.lower()] = k
    for w in v.values():
        for m in w:
            if m != m.lower() and m.lower() not in shortcuts:
                shortcuts[m.lower()] = m
# This needs to be the final step of constructing `shortcuts`:
# append "__pre" to all shortcuts that need it and don't yet have it
for k, v in shortcuts.copy().items():
    if (
        not k.endswith("__pre")
        and k + "__pre" not in shortcuts
        and k not in condition_independent
    ):
        shortcuts[k + "__pre"] = v + "__pre"
shortcuts = ShortcutsDict(**shortcuts)


def da_to_array(da, xr_dims):
    """Convert an xarray `DataArray` `da` into a NumPy `array`.

    The NumPy `array` will have as many dimensions as `len(xr_dims)` and the
    dimensions will be in the same order as indicated in `xr_dims`.

    If `da` does not contain a dimension from `xr_dims`, a new singleton
    dimension will be added in the appropriate position.

    `da` is not allowed to contain any dimensions that are not in `xr_dims`.

    Parameters
    ----------
    da : xarray.DataArray
        The `DataArray` to be converted.
    xr_dims : iterable
        The full list of dimension names in the correct order for the output
        NumPy array.  Can be obtained from an xarray `Dataset` (`ds`) as
        `ds.sizes`.

    Returns
    -------
    numpy.array
        The converted `array`.
    """
    # Get `DataArray` info
    da_dims = list(da.sizes)
    try:
        da_data = da.data.astype(float)
    except ValueError:
        return None
    # Prepare for loop through `xr_dims`
    move_from = []
    extra_dims = 0
    for d in xr_dims:
        if d in da_dims:
            # If the dimension is in `da`, just append the appropriate position
            # to `move_from`
            move_from.append(da_dims.index(d))
        else:
            # If the dimension is not in `da`, we need to create it at the end
            move_from.append(len(da_dims) + extra_dims)
            da_data = np.expand_dims(da_data, -1)
            extra_dims += 1  # increment offset, for adding multiple new dims
    # Move axes around to the shape matching `xr_dims`
    return np.moveaxis(da_data, move_from, range(len(xr_dims)))


class OptsDict(ShortcutDotDict):
    def __init__(self, shortcuts):
        super().__init__(shortcuts)

    def __repr__(self):
        text = "CO2System settings."
        opts_sections = {
            "Equilibrium constants": [
                "opt_pH_scale",
                "opt_k_carbonic",
                "opt_k_HSO4",
                "opt_k_HF",
                "opt_k_BOH3",
                "opt_k_phosphate",
                "opt_k_NH3",
                "opt_k_Si",
                "opt_k_calcite",
                "opt_k_aragonite",
                "opt_k_H2O",
                "opt_k_HNO2",
            ],
            "Pressure correction factors": [
                "opt_factor_k_H2CO3",
                "opt_factor_k_HCO3",
                "opt_factor_k_BOH3",
                "opt_factor_k_H2O",
            ],
            "Total salt contents": [
                "opt_total_borate",
                "opt_Ca",
            ],
            "Other settings": [
                "opt_HCO3_root",  # needs to not be last in this list
                "opt_gas_constant",
                "opt_fugacity_factor",
            ],
        }
        sections = list(opts_sections.keys())
        for section, opts in opts_sections.items():
            if section == sections[-1]:
                text += f"\n└─ {section.upper()}:"
            else:
                text += f"\n├─ {section.upper()}:"
            opts = [opt for opt in opts if opt in self.data]
            len_opts_max = max([len(opt) for opt in opts])
            for opt in opts:
                if section == sections[-1]:
                    if opt == opts[-1]:
                        text += "\n   └─"
                    else:
                        text += "\n   ├─"
                else:
                    if opt == opts[-1]:
                        text += "\n│  └─"
                    else:
                        text += "\n│  ├─"
                text += "─" * (
                    len_opts_max - len(opt)
                ) + " {}[{:>2.0f}]: {}.".format(  # noqa: UP032
                    opt,
                    self.data[opt],
                    citations[opt][self.data[opt]],
                )
        text += "\nOnly parameterisations with multiple options are included."
        return text


class CO2System(FunctionGraph):
    """An equilibrium model of the marine carbonate system.

    Methods
    -------
    adjust
        Adjust the system to a different temperature and/or pressure.
    get_grads
        Calculate derivatives of parameters with respect to each other.
    get_jacs
        Calculate Jacobian matrices of derivatives.
    keys_all
        Return a tuple of all possible results keys, including those that have
        not yet been solved for.
    propagate
        Propagate independent uncertainties through the calculations.
    solve
        Calculate parameter(s) and store them internally.
    to_pandas
        Return parameters as a pandas `Series` or `DataFrame`.
    to_xarray
        Return parameters as an xarray `DataArray` or `Dataset`.

    Attributes
    ----------
    grads : dict
        Derivatives of parameters with respect to each other, calculated with
        get_grads.
    ignored : list
        Which kwargs or keys in data were ignored.
    opts : dict
        The optional settings being used for calculations.  Constructed when
        the CO2System is initalised; subsequent changes will not affect any
        calculations.
    uncertainty or u : Uncertainties
        Uncertainties in parameters with respect to each other, calculated with
        propagate (or prop).
    validity or v : Valids
        Validity of parameters with respect to each other.

    In addition to the methods listed above, all of the methods usually
    available for a dict can be used.  Methods such as keys, values and
    items will run only over parameters that have already been solved for.

    Advanced attributes
    -------------------
    data : dict
        The known parameters (either user-provided or solved for).
        This is **not** related to the sys function data argument.
    graph : nx.DiGraph
        The graph of calculations.
    shortcuts : dict
        Alternative key mapper.
    _adjusted : bool
        Whether this system was generated using adjust.
    _icase : int
        Which known core parameters were provided.
    _method_fCO2 : int
        Which method was used to adjust fCO2 in an adjusted system.
    _nodes_defaults : set
        Which parameters took the default values.
    _nodes_original : set
        Which parameters were user-provided or took the default values.
    _nodes_user : set
        Which parameters were user-provided.
    _pd_index : pd.Index
        If data was a pandas DataFrame, this contains its index.
    _requested : list
        Which parameters have been directly requested for solving.
    _which_fCO2_insitu : int
        If method_fCO2 == 1, whether pre-adjustment or adjusted fCO2
        represents in situ conditions.
    _xr_dims : tuple
        If data was an xarray Dataset, this contains all its dimensions.
    _xr_shape : tuple
        If data was an xarray Dataset, this contains its fullest shape.
    """

    from .uncertainty import set_u_OEDG18

    def __init__(
        self,
        defaults: dict | None = None,
        graph: nx.DiGraph | None = None,
        funcs: dict | None = None,
        shortcuts: dict | None = None,
        no_store: set | None = None,
        icase: int | None = None,
        opts: dict | None = None,
        pd_index=None,
        xr_dims=None,
        xr_shape=None,
    ):
        super().__init__(
            defaults=defaults,
            graph=graph,
            funcs=funcs,
            shortcuts=shortcuts,
            no_store=no_store,
        )
        self._adjusted = False
        self._method_fCO2 = None
        self._which_fCO2_insitu = None
        self._icase = icase
        self.opts = OptsDict(self.shortcuts)
        self.opts.update(opts)
        self._pd_index = pd_index
        if xr_dims is not None:
            assert xr_shape is not None
            assert len(xr_dims) == len(xr_shape)
        else:
            assert xr_shape is None
        self._xr_dims = xr_dims
        self._xr_shape = xr_shape

    def __repr__(self):
        text = "CO2System"
        if self._adjusted:
            text += " (adjusted)"
        if self._icase == 0:
            text += " with no known CO2 parameters."
        elif self._icase < 100:
            known = parameters_core[self._icase - 1]
            text += f" with known {known}."
        else:
            text += " with known {} and {}.".format(
                *icase_to_params(self._icase)
            )
        text += "\n├─ User-defined parameters:"
        if len(self._nodes_user) == 0:
            text += "\n    None."
        else:
            params_user = list(self._nodes_user)
            params_user.sort()
            text += "\n│  └─ "
            for i, p in enumerate(params_user):
                text += p
                if i < len(params_user) - 1:
                    text += ", "
                else:
                    text += "."
            if self._adjusted:
                text += (
                    "\n│     (__pre suffix indicates pre-adjustment values)"
                )
        if self._adjusted and self._method_fCO2 is not None:
            text += "\n├─ Temperature-sensitivity of fCO2:"
            if self._method_fCO2 == 1:
                text += "\n│  ├─────── method_fCO2[{:>2.0f}]: {}.".format(
                    self._method_fCO2,
                    citations["method_fCO2"][self._method_fCO2],
                )
                text += "\n│  └─ which_fCO2_insitu[{:>2.0f}]: {}.".format(
                    self._which_fCO2_insitu,
                    citations["which_fCO2_insitu"][self._which_fCO2_insitu],
                )
            else:
                text += "\n│  └─ method_fCO2[{:>2.0f}]: {}.".format(
                    self._method_fCO2,
                    citations["method_fCO2"][self._method_fCO2],
                )
        text += "\n└─ Parameterisations and options:"
        opts = ["opt_pH_scale", "opt_k_carbonic", "opt_total_borate"]
        len_opts_max = max([len(opt) for opt in opts])
        for opt in opts:
            text += (
                "\n   ├─"
                + "─" * (len_opts_max - len(opt))
                + " {}[{:>2.0f}]: {}.".format(  # noqa: UP032
                    opt,
                    self.opts[opt],
                    citations[opt][self.opts[opt]],
                )
            )
        text += (
            "\n   └─"
            + "─" * (len_opts_max - 2)
            + " Others: see CO2System.opts."
        )
        return text

    def solve(
        self,
        parameters: list[str] | str | None = None,
        store_steps: int = 1,
    ):
        """Calculate parameter(s) and store them internally.

        Parameters
        ----------
        parameters : str or list of str, optional
            Which parameter(s) to calculate and store, by default `None`, in
            which case all possible parameters are calculated and stored
            internally.  The full list of possible parameters is provided
            below.
        store_steps : int, optional
            Whether/which non-requested parameters calculated during
            intermediate calculation steps should be stored, by default `1`.
            The options are
                0 - store only the specifically requested parameters,
                1 - store the most used set of intermediate parameters, or
                2 - store the complete set of parameters.

        Returns
        -------
        CO2System
            The original `CO2System` including the newly solved parameters.

        PARAMETERS THAT CAN BE SOLVED FOR
        =================================
        Note that some parameters may be available only for certain
        combinations of core carbonate system parameters optional settings.

        pH on different scales
        ----------------------
             Key | Description
        -------: | :-----------------------------------------------------------
              pH | pH on the scale specified by `opt_pH_scale`.
        pH_total | pH on the total scale.
          pH_sws | pH on the seawater scale.
         pH_free | pH on the free scale.
          pH_nbs | pH on the NBS scale.
              fH | H+ activity coefficient for conversions to/from NBS scale.

        Chemical speciation
        -------------------
        All are substance contents in units of µmol/kg.

           Key | Description
        -----: | :-------------------------------------------------------------
        H_free | "Free" protons.
            OH | Hydroxide ion.
           CO3 | Carbonate ion.
          HCO3 | Bicarbonate ion.
           CO2 | Aqueous CO2.
          BOH4 | Tetrahydroxyborate.
          BOH3 | Boric acid.
         H3PO4 | Phosphoric acid.
         H2PO4 | Dihydrogen phosphate.
          HPO4 | Monohydrogen phosphate.
           PO4 | Phosphate.
        H4SiO4 | Orthosilicic acid.
        H3SiO4 | Trihydrogen orthosilicate.
           NH3 | Ammonia.
           NH4 | Ammonium.
            HS | Bisulfide.
           H2S | Hydrogen sulfide.
          HSO4 | Bisulfate.
           SO4 | Sulfate.
            HF | Hydrofluoric acid.
             F | Fluoride.
          HNO2 | Nitrous acid.
           NO2 | Nitrite.

        Chemical buffer factors
        -----------------------
                              Key | Description
        ------------------------: | :------------------------------------------
                   revelle_factor | Revelle factor.
                              psi | Psi of FCG94.
                        gamma_dic | Buffer factors from ESM10.
                         beta_dic | Buffer factors from ESM10.
                        omega_dic | Buffer factors from ESM10.
                 gamma_alkalinity | Buffer factors from ESM10.
                  beta_alkalinity | Buffer factors from ESM10.
                 omega_alkalinity | Buffer factors from ESM10.
                         Q_isocap | Isocapnic quotient from HDW18.
                  Q_isocap_approx | Approximate isocapnic quotient from HDW18.
                       dlnfCO2_dT | temperature sensitivity of ln(fCO2).
                       dlnpCO2_dT | temperature sensitivity of ln(pCO2).
        substrate_inhibitor_ratio | HCO3/H_free, substrate:inhibitor from B15.

        Equilibrium constants
        ---------------------
        All are returned on the pH scale specified by `opt_pH_scale`.

                 Key | Description
        -----------: | :-------------------------------------------------------
              pk_CO2 | Henry's constant for CO2.
            pk_H2CO3 | First dissociation constant for carbonic acid.
             pk_HCO3 | Second dissociation constant for carbonic acid.
              pk_H2O | Water dissociation constant.
             pk_BOH3 | Boric acid equilibrium constant.
          pk_HF_free | HF dissociation constant (always free scale).
        pk_HSO4_free | Bisulfate dissociation constant (always free scale).
            pk_H3PO4 | First dissociation constant for phosphoric acid.
            pk_H2PO4 | Second dissociation constant for phosphoric acid.
             pk_HPO4 | Third dissociation constant for phosphoric acid.
               pk_Si | Silicic acid dissociation constant.
              pk_NH3 | Ammonia equilibrium constant.
              pk_H2S | Hydrogen sulfide dissociation constant.
             pk_HNO2 | Nitrous acid dissociation constant.

        Other results
        -------------
                    Key | Description (unit)
        --------------: | :----------------------------------------------------
                upsilon | Temperature-sensitivity of fCO2 (%/°C)
        fugacity_factor | Converts between pCO2 and fCO2.
              vp_factor | Vapour pressure factor, converts pCO2 and xCO2.
           gas_constant | Universal gas constant (J/mol/K).
        """
        return super().solve(parameters, store_steps=store_steps)

    def to_pandas(self, parameters=None):
        """Return parameters as a pandas `Series` or `DataFrame`.  All
        parameters should be scalar or one-dimensional vectors of the same
        size.

        Parameters
        ----------
        parameters : str or list of str, optional
            The parameter(s) to return.  These are solved for if not already
            available. If `None`, then all parameters that have already been
            solved for are returned.

        Returns
        -------
        pd.Series or pd.DataFrame
            The parameter(s) as a `pd.Series` (if `parameters` is a `str`) or
            as a `pd.DataFrame` (if `parameters` is a `list`) with the original
            pandas index passed into the `CO2System` as `data`.  If `data` was
            not a `pd.DataFrame` then the default index will be used.
        """
        try:
            import pandas as pd

            if parameters is None:
                parameters = self.keys()
            self.solve(parameters=parameters)
            if isinstance(parameters, str):
                return pd.Series(data=self[parameters], index=self._pd_index)
            else:
                return pd.DataFrame(
                    {
                        p: pd.Series(
                            data=self[p] * np.ones(self._pd_index.shape),
                            index=self._pd_index,
                        )
                        for p in parameters
                    }
                )
        except ImportError:
            warn("pandas could not be imported.", stacklevel=3)

    def _get_xr_ndims(self, parameter):
        ndims = []
        if not np.isscalar(self[parameter]):
            for i, vs in enumerate(self[parameter].shape):
                if vs == self._xr_shape[i]:
                    ndims.append(self._xr_dims[i])
        return ndims

    def to_xarray(self, parameters=None):
        """Return parameters as an xarray `DataArray` or `Dataset`.

        Parameters
        ----------
        parameters : str or list of str, optional
            The parameter(s) to return.  These are solved for if not already
            available. If `None`, then all parameters that have already been
            solved for are returned.

        Returns
        -------
        xr.DataArray or xr.Dataset
            The parameter(s) as a `xr.DataArray` (if `parameters` is a `str`)
            or as a `xr.Dataset` (if `parameters` is a `list`) with the
            original xarray dimensions passed into the `CO2System` as `data`.
            If `data` was not an `xr.Dataset` then this function will not work.
        """
        assert self._xr_dims is not None and self._xr_shape is not None, (
            "`data` was not provided as an `xr.Dataset` "
            + "when creating this `CO2System`."
        )
        try:
            import xarray as xr

            if parameters is None:
                parameters = self.keys()
            self.solve(parameters=parameters)
            if isinstance(parameters, str):
                ndims = self._get_xr_ndims(parameters)
                return xr.DataArray(np.squeeze(self[parameters]), dims=ndims)
            else:
                return xr.Dataset(
                    {
                        p: xr.DataArray(
                            np.squeeze(self[p]), dims=self._get_xr_ndims(p)
                        )
                        for p in parameters
                    }
                )
        except ImportError:
            warn("xarray could not be imported.", stacklevel=3)

    def _get_expUps(
        self,
        method_fCO2,
        temperature,
        bh_upsilon=None,
        which_fCO2_insitu=1,
    ):
        if method_fCO2 in [1, 2, 3, 4]:
            self.solve("gas_constant")
        match method_fCO2:
            case 1:
                self.solve("fCO2")
                fCO2 = self.fCO2
                assert which_fCO2_insitu in [1, 2]
                if which_fCO2_insitu == 2:
                    # If the output conditions are the environmental ones, then
                    # we need to provide an estimate of output fCO2 in order to
                    # use the bh parameterisation; we get this using the
                    # method_fCO2=2 approach:
                    fCO2 = fCO2 * upsilon.expUps_TOG93_H24(
                        self.data["temperature"],
                        temperature,
                        self.data["gas_constant"],
                    )
                return upsilon.expUps_parameterised_H24(
                    self.data["temperature"],
                    temperature,
                    self.data["salinity"],
                    fCO2,
                    self.data["gas_constant"],
                    which_fCO2_insitu=which_fCO2_insitu,
                )
            case 2:
                return upsilon.expUps_TOG93_H24(
                    self.data["temperature"],
                    temperature,
                    self.data["gas_constant"],
                )
            case 3:
                return upsilon.expUps_enthalpy_H24(
                    self.data["temperature"],
                    temperature,
                    self.data["gas_constant"],
                )
            case 4:
                assert bh_upsilon is not None, (
                    "A bh_upsilon value must be provided for method_fCO2=4."
                )
                return upsilon.expUps_Hoff_H24(
                    self.data["temperature"],
                    temperature,
                    self.data["gas_constant"],
                    bh_upsilon,
                )
            case 5:
                return upsilon.expUps_linear_TOG93(
                    self.data["temperature"],
                    temperature,
                )
            case 6:
                return upsilon.expUps_quadratic_TOG93(
                    self.data["temperature"],
                    temperature,
                )

    def _adjust_prep(self, param):
        # Convert temperature and/or pressure from pandas Series to NumPy
        # arrays, if necessary.  The checks to see if they are Series are
        # not foolproof, but they do avoid needing to import pandas.
        if all(hasattr(param, a) for a in ["index", "values", "dtype"]):
            assert self._pd_index is not None, (
                "Parameters cannot be provided as a pandas Series"
                + " because this CO2System was not constructed"
                + " from an pandas DataFrame."
            )
            assert self._pd_index.equals(param.index), (
                "Cannot use this pandas Series for the adjust-to value"
                + " because its index does not match that used to construct"
                + " this CO2System."
            )
            param = param.to_numpy().astype(float)
        # Convert temperature and/or pressure from xarray DataArrays to NumPy
        # arrays, if necessary.  The checks to see if they are DataArrays are
        # not foolproof, but they do avoid needing to import xarray.
        if all(hasattr(param, a) for a in ["data", "dims", "coords"]):
            assert self._xr_dims is not None, (
                "Parameters cannot be provided as an xarray DataArray"
                + " because this CO2System was not constructed"
                + " from an xarray Dataset."
            )
            param = da_to_array(param, self._xr_dims)
        return param

    def _adjust_alkalinity_dic(self, temperature=None, pressure=None):
        temperature = self._adjust_prep(temperature)
        pressure = self._adjust_prep(pressure)
        kwargs_adjust = {}
        if temperature is not None:
            kwargs_adjust["temperature"] = temperature
        if pressure is not None:
            kwargs_adjust["pressure"] = pressure
        data_pre = {
            k: self.data[k] for k in self._nodes_user if k not in kwargs_adjust
        }
        co2a = CO2System(
            graph=self.graph,
            defaults=self.defaults,
            shortcuts=self.shortcuts,
            icase=self._icase,
            opts=self.opts,
            pd_index=self._pd_index,
            xr_dims=self._xr_dims,
            xr_shape=self._xr_shape,
        ).set_data(**data_pre, **kwargs_adjust)
        return co2a

    def _adjust_2p(self, temperature=None, pressure=None):
        temperature = self._adjust_prep(temperature)
        pressure = self._adjust_prep(pressure)
        kwargs_adjust = {}
        if temperature is not None:
            kwargs_adjust["temperature"] = temperature
        if pressure is not None:
            kwargs_adjust["pressure"] = pressure
        # To adjust to a different temperature/pressure, we need to know
        # alkalinity and DIC for the original system.  First, we get the
        # subgraph from the original system that contains just alkalinity, DIC
        # and all their ancestors.
        graph_pre = self.graph.subgraph(
            nx.ancestors(self.graph, "alkalinity")
            | nx.ancestors(self.graph, "dic")
            | {"alkalinity", "dic"}
        )
        # All of the nodes in graph_pre that are not condition-independent are
        # now renamed with "__pre" appended, to keep them distinct from the
        # same nodes under the adjusted conditions.  Temperature and pressure
        # are considered to be condition-independent if they were not adjusted.
        no_pre = [*condition_independent]
        for p in ["temperature", "pressure"]:
            if p not in kwargs_adjust:
                no_pre.append(p)
        graph_pre = nx.relabel_nodes(
            graph_pre,
            {n: n if n in no_pre else n + "__pre" for n in graph_pre.nodes},
        )
        args = {}
        for node, attrs in graph_pre.nodes.items():
            if "func" in attrs:
                args[node] = [
                    k if k in no_pre else k + "__pre"
                    for k in signature(attrs["func"]).parameters
                ]
        nx.set_node_attributes(graph_pre, args, name="args")
        # graph_pre can now be merged with a new graph to compute everything
        # from alkalinity and DIC.  The original system's `opts` are retained.
        funcs_adj = get_funcs | get_funcs_core[102]
        for opt, v in self.opts.items():
            # opt_HCO3_root is available only for icase == 207
            if opt != "opt_HCO3_root":
                funcs_adj.update(get_funcs_opts[opt][v])
        graph_adj = nx.compose(graph_pre, FunctionGraph.get_graph(funcs_adj))
        # The new system will have the same set of user-provided parameter
        # values as the original, but the ones that are condition-dependent get
        # renamed with "__pre" appended.
        data_pre = self[list(self._nodes_original)]
        for k, v in data_pre.copy().items():
            if k not in no_pre:
                data_pre[k + "__pre"] = data_pre.pop(k)
        co2a = CO2System(
            graph=graph_adj,
            defaults=self.defaults,
            shortcuts=self.shortcuts,
            icase=self._icase,
            opts=self.opts,
            pd_index=self._pd_index,
            xr_dims=self._xr_dims,
            xr_shape=self._xr_shape,
        ).set_data(**data_pre, **kwargs_adjust)
        # Parameters that have already been solved for in the original system
        # are copied across, so that they don't need solving for again.
        for k, v in self.data.items():
            if k not in co2a:
                if k in no_pre:
                    co2a.data[k] = v
                else:
                    co2a.data[k + "__pre"] = v
        # Uncertainties that were assigned in the original system are copied
        # across.
        uncertainty_pre = {}
        for k, v in self.uncertainty.assigned.items():
            if k in no_pre:
                uncertainty_pre[k] = v
            else:
                uncertainty_pre[k + "__pre"] = v
        co2a.set_uncertainty(**uncertainty_pre)
        co2a.solve(self._requested)
        return co2a

    def _adjust_1p(
        self,
        temperature=None,
        bh=None,
        method_fCO2=1,
        which_fCO2_insitu=1,
    ):
        temperature = self._adjust_prep(temperature)
        bh = self._adjust_prep(bh)
        assert method_fCO2 in [1, 2, 3, 4, 5, 6]
        # To adjust to a different temperature/pressure, we need to know fCO2
        # for the original system.  First, we get the subgraph from the
        # original system that contains only fCO2 and all its ancestors.
        graph_pre = self.graph.subgraph(
            nx.ancestors(self.graph, "fCO2") | {"fCO2"}
        )
        # All of the nodes in graph_pre that are not condition-independent are
        # now renamed with "__pre" appended, to keep the distinct from the same
        # nodes under the adjusted conditions.  Pressure is also considered to
        # be condition-independent as it cannot currently be adjusted.
        no_pre = [*condition_independent, "pressure"]
        graph_pre = nx.relabel_nodes(
            graph_pre,
            {n: n if n in no_pre else n + "__pre" for n in graph_pre.nodes},
        )
        args = {}
        for node, attrs in graph_pre.nodes.items():
            if "func" in attrs:
                args[node] = [
                    k if k in no_pre else k + "__pre"
                    for k in signature(attrs["func"]).parameters
                ]
        nx.set_node_attributes(graph_pre, args, name="args")
        # graph_pre can now be merged with a new graph to compute everything
        # from fCO2.  The original system's `opts` are retained.
        funcs_adj = get_funcs | get_funcs_core[5]
        for opt, v in self.opts.items():
            # opt_HCO3_root is available only for icase == 207
            if opt != "opt_HCO3_root":
                funcs_adj.update(get_funcs_opts[opt][v])
        graph_adj = nx.compose(graph_pre, FunctionGraph.get_graph(funcs_adj))
        # The new system will have the same set of user-provided parameter
        # values as the original, but the ones that are condition-dependent get
        # renamed with "__pre" appended.
        data_pre = self[list(self._nodes_original)]
        for k, v in data_pre.copy().items():
            if k not in no_pre:
                data_pre[k + "__pre"] = data_pre.pop(k)
        # Here we add the functions that convert fCO2 across temperatures to
        # `graph_adj`, depending on the conversion option.
        cfuncs = {
            "fCO2": lambda fCO2__pre, exp_upsilon: fCO2__pre * exp_upsilon
        }
        if method_fCO2 == 1:
            assert which_fCO2_insitu in [1, 2]
            if which_fCO2_insitu == 1:
                cfuncs["bh"] = (
                    lambda temperature__pre, salinity, fCO2__pre, coeffs_bh: (
                        upsilon.get_bh_H24(
                            temperature__pre, salinity, fCO2__pre, coeffs_bh
                        )
                    )
                )
            elif which_fCO2_insitu == 2:
                cfuncs["bh"] = (
                    lambda temperature__pre, temperature, salinity, fCO2__pre, coeffs_bh, gas_constant: (
                        upsilon.get_bh_H24(
                            temperature__pre,
                            salinity,
                            fCO2__pre
                            * upsilon.expUps_TOG93_H24(
                                temperature__pre,
                                temperature,
                                gas_constant,
                            ),
                            coeffs_bh,
                        )
                    )
                )
            cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
        elif method_fCO2 == 2:
            cfuncs["bh"] = lambda: upsilon.bh_TOG93_H24
            cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
        elif method_fCO2 == 3:
            cfuncs["bh"] = lambda: upsilon.bh_enthalpy_H24
            cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
        elif method_fCO2 == 4:
            assert bh is not None, (
                "A `bh` value must be provided for `method_fCO2=4`."
            )
            data_pre["bh"] = bh
            no_pre.append("bh")
            cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
        elif method_fCO2 == 5:
            cfuncs["exp_upsilon"] = upsilon.expUps_linear_TOG93
        elif method_fCO2 == 6:
            cfuncs["exp_upsilon"] = upsilon.expUps_quadratic_TOG93
        for k, func in cfuncs.items():
            for f in signature(func).parameters:
                graph_adj.add_edge(f, k)
        nx.set_node_attributes(graph_adj, cfuncs, name="func")
        args = {}
        for node, attrs in graph_adj.nodes.items():
            if node in cfuncs:
                args[node] = list(
                    signature(attrs["func"]).parameters
                )  # could come from graph args, not function signature?
        nx.set_node_attributes(graph_adj, args, name="args")
        # Now we can create the new CO2System
        defaults = self.defaults
        if method_fCO2 == 1:
            defaults = defaults.copy()
            defaults["coeffs_bh"] = upsilon.coeffs_bh_H24()
        elif method_fCO2 == 5:
            defaults = defaults.copy()
            defaults["bl"] = upsilon.bl_TOG93
        co2a = CO2System(
            graph=graph_adj,
            defaults=defaults,
            shortcuts=self.shortcuts,
            icase=self._icase,
            opts=self.opts,
            pd_index=self._pd_index,
            xr_dims=self._xr_dims,
            xr_shape=self._xr_shape,
        ).set_data(**data_pre, temperature=temperature)
        # Parameters that have already been solved for in the original system
        # are copied across, so that they don't need solving for again.
        for k, v in self.data.items():
            if k not in co2a:
                if k in no_pre:
                    co2a.data[k] = v
                else:
                    co2a.data[k + "__pre"] = v
        # Uncertainties that were assigned in the original system are
        # copied across
        uncertainty_pre = {}
        for k, v in self.uncertainty.assigned.items():
            if k in no_pre:
                uncertainty_pre[k] = v
            else:
                uncertainty_pre[k + "__pre"] = v
        co2a.set_uncertainty(**uncertainty_pre)
        co2a.solve(self._requested)
        co2a._method_fCO2 = method_fCO2
        # For method_fCO2 == 1 only (H24 parameterisation), we also need
        # to store the which_fCO2_insitu value
        if method_fCO2 == 1:
            co2a._which_fCO2_insitu = which_fCO2_insitu
        # Finally, assign uncertainties based on H24
        if method_fCO2 == 1:  # H24 parameterisation
            nx.set_node_attributes(
                co2a.graph,
                {"coeffs_bh": True},
                name="coeffs",
            )
            co2a.set_uncertainty(coeffs_bh=covmx.bh_H24())
        elif method_fCO2 == 5:
            nx.set_node_attributes(
                co2a.graph,
                {"bl": False},
                name="coeffs",
            )
            co2a.set_uncertainty(bl=upsilon.u_bl_TOG93**2)
        return co2a

    def adjust(self, **kwargs):
        """Adjust the CO2System to a different temperature and/or
        pressure.

        Works differently depending on whether one or two core marine
        carbonate system (MCS) parameters are known.

        If the original CO2System was created from a pandas DataFrame or
        xarray Dataset using the data kwarg, then the temperature and
        pressure provided to adjust can be pandas Series or xarray
        DataArrays, as long as their index or dimensions are consistent
        with the original data.

        Any other system properties (e.g. salinity, total salt contents,
        optional settings) must be defined when creating the original,
        unadjusted CO2System.  They cannot be added in during the adjust
        step.

        Parameters when two core MCS parameters are known
        -------------------------------------------------
        temperature : array-like, optional
            The temperature to adjust to in °C, by default None,
            in which case temperature is not adjusted.
        pressure : array-like, optional
            The pressure to adjust to in °C, by default None,
            in which case pressure is not adjusted.

        Parameters when one core MCS parameter is known
        -----------------------------------------------
        temperature : array-like
            The temperature to adjust to in °C.
        method_fCO2 : int
            How to do the temperature conversion:
                1: parameterised υh equation of H24 (default).
                2: constant υh fitted to the TOG93 dataset by H24.
                3: constant theoretical υx of H24.
                4: H24 approach but using a user-provided bh.
                5: linear fit of TOG93.
                6: quadratic fit of TOG93.

        Additional parameter when method_fCO2 is 1
        ------------------------------------------
        * which_fCO2_insitu: whether the input- (1, default) or output-
        (2) condition pCO2, fCO2, [CO2(aq)] and/or xCO2 values are at
        in situ conditions, for determining bh with the parameterisation
        of H24.

        Additional parameter when method_fCO2 is 4
        ------------------------------------------
        bh : array-like
            bh of H24 in J/mol.

        Returns
        -------
        CO2System
            A separate CO2System adjusted to the requested temperature
            and/or pressure.
        """
        self_requested = self._requested.copy()  # needs to stay here
        kwargs = {shortcuts[k.lower()]: v for k, v in kwargs.items()}
        if self._icase == 102:
            self_adjusted = self._adjust_alkalinity_dic(**kwargs)
            return self_adjusted
        elif self._icase > 102:
            self_adjusted = self._adjust_2p(**kwargs)
        elif self._icase in [4, 5, 8, 9]:
            self_adjusted = self._adjust_1p(**kwargs)
        else:
            raise PyCO2SYSError("This CO2System cannot be adjusted.")
        self._requested = self_requested
        self_adjusted._adjusted = True
        state_zero = {}
        for p in self_adjusted._nodes_user.copy():
            if p in self._nodes_defaults:
                self_adjusted._nodes_user.remove(p)
                self_adjusted._nodes_defaults |= {p}
                state_zero[p] = 0
        nx.set_node_attributes(self_adjusted.graph, state_zero, "state")
        return self_adjusted

    def get_u_coeffs_from_single(self, **u_single) -> dict[str, float]:
        """Convert a set of single uncertainty values for pKs (e.g., from
        OEDG18) into the vectors needed for propagation in PyCO2SYS.

        The lengths of these vectors might be different depending on
        which parameterisation has been chosen for each pK.

        Parameters
        ----------
        u_single
            The single uncertainty values in the pKs, e.g.:
            `**dict(pk_H2O=0.01)`.

        Returns
        -------
        dict[str, float]
            The uncertainty values in the pK coefficients.
        """
        u_coeffs = {}
        for k, v in u_single.items():
            try:
                u_coeffs["coeffs_" + k] = np.zeros_like(self["coeffs_" + k])
                u_coeffs["coeffs_" + k] = u_coeffs["coeffs_" + k].at[-1].set(v)
            except nx.NetworkXError:
                warn(f'No coeffs available for "{k}"', stacklevel=3)
        return u_coeffs

    def set_u_coeffs_from_single(self, **u_single):
        """Convert a set of single uncertainty values for pKs (e.g., from
        OEDG18) into the vectors needed for propagation in PyCO2SYS and assign
        them as the uncertainty values in this CO2System.

        The lengths of these vectors might be different depending on which
        parameterisation has been chosen for each pK.

        Parameters
        ----------
        u_single
            The single uncertainty values in the pKs, e.g.:
            `**dict(pk_H2O=0.01)`.

        Returns
        -------
        CO2System
            The CO2System with the uncertainties assigned.
        """
        u_coeffs = self.get_u_coeffs_from_single(**u_single)
        self.set_u(**u_coeffs)
        return self


def sys(data=None, **kwargs):
    """Initialise a `CO2System`.

    Once initialised, various methods are available including `solve`, `adjust`
    and `propagate`.

    PARAMETERS PROVIDED AS KWARGS
    =============================
    There are many possible kwargs, so they are grouped into sections below.

    Data as separate variables
    --------------------------
    If the input data are all stored as separate variables, they can be
    provided with the appropriate kwargs from below.  In this case, each
    variable must be one of a scalar, a `list` or a NumPy `array`, and the
    shapes of these arrays must be mutually broadcastable.  For example:

      >>> import PyCO2SYS as pyco2, numpy as np
      >>> dic = [2100, 2150]
      >>> temperature = np.array([15, 13.5])
      >>> co2s = pyco2.sys(dic=dic, pH=8.1, temperature=temperature)

    Data variables in a container
    -----------------------------
    If the input data are gathered in a `dict`, pandas `DataFrame` or xarray
    `Dataset`, then this can be provided with the `data` kwarg.  The keys in
    the container dataset must be strings and they should correspond to the
    kwargs below.  If they do not correspond, then the kwarg can be passed with
    the corresponding key instead.  For example:

      >>> import PyCO2SYS as pyco2
      >>> data = {"dic": [2100, 2150], "pH_lab": 8.1, "t_lab": 25}
      >>> co2s = pyco2.sys(data=data, pH="pH_lab", temperature="t_lab")

    Core marine carbonate system parameters
    ---------------------------------------
    A maximum of two core parameters may be provided, and not all combinations
    are valid.  In some cases, a subset of calculations can still be carried
    out with only one parameter.  It is also possible to pass none of these
    parameters and still calculate e.g. equilibrium constants.

               Parameter | Description (unit)
    -------------------: | :---------------------------------------------------
              alkalinity | Total alkalinity (µmol/kg)
                     dic | Dissolved inorganic carbon (µmol/kg)
                      pH | Seawater pH on the scale given by `opt_pH_scale`
                    pCO2 | Seawater partial pressure of CO2 (µatm)
                    fCO2 | Seawater fugacity of CO2 (µatm)
                     CO2 | Aqueous CO2 content (µmol/kg)
                    HCO3 | Bicarbonate ion content (µmol/kg)
                     CO3 | Carbonate ion content (µmol/kg)
                    xCO2 | Seawater dry air mole fraction of CO2 (ppm)
      saturation_calcite | Saturation state with respect to calcite
    saturation_aragonite | Saturation state with respect to aragonite

    Hydrographic conditions
    -----------------------
    Middle column gives default values if not provided.

              Parameter | Df. | Description (unit)
    ------------------: | --: | :----------------------------------------------
               salinity |  35 | Practical salinity
            temperature |  25 | Temperature (°C)
               pressure |   0 | Hydrostatic pressure (dbar)
    pressure_atmosphere |   1 | Atmospheric pressure (atm)

    Nutrients and other solutes
    ---------------------------
    Middle column gives default values; S = calculated from salinity.

          Parameter | Df. | Description (unit)
    --------------: | :-: | :----------------------------------------------
     total_silicate |  0  | Total dissolved silicate (µmol/kg)
    total_phosphate |  0  | Total dissolved phosphate (µmol/kg)
      total_ammonia |  0  | Total dissolved ammonia (µmol/kg)
      total_sulfide |  0  | Total dissolved sulfide (µmol/kg)
      total_nitrite |  0  | Total dissolved nitrite (µmol/kg)
       total_borate |  S  | Total dissolved borate (µmol/kg)
     total_fluoride |  S  | Total dissolved fluoride (µmol/kg)
      total_sulfate |  S  | Total dissolved sulfate (µmol/kg)
                 Ca |  S  | Dissolved calcium (µmol/kg)

    SETTINGS PROVIDED AS KWARGS
    ===========================
    Options such as which pH scale to use and the choice of parameterisation
    for various e.g. equilibrium constants can also be altered with kwargs.
    These kwargs must all be scalar integers.

    Citations use a code with the initials of the first few authors' surnames
    plus the final two digits of the year of publication.  Refer to the online
    documentation for full references.

    pH scale
    --------
    opt_pH_scale: pH scale of `pH` (if provided); also used for calculating
    equilibrium constants.
        1: total [DEFAULT].
        2: seawater.
        3: free.
        4: NBS.

    Carbonic acid dissociation
    --------------------------
    opt_k_carbonic: parameterisation for carbonic acid dissociation (K1 and
    K2).
         1: RRV93.
         2: GP89.
         3: H73a and H73b refit by DM87.
         4: MCHP73 refit by DM87.
         5: H73a, H73b and MCHP73 refit by DM87.
         6: MCHP73 ("GEOSECS").
         7: MCHP73 ("GEOSECS-Peng").
         8: M79 (freshwater).
         9: CW98.
        10: LDK00 [DEFAULT].
        11: MM02.
        12: MPL02.
        13: MGH06.
        14: M10.
        15: WMW14.
        16: SLH20.
        17: SB21.
        18: PLR18.
        19: MMB25.
    opt_factor_k_H2CO3: pressure correction for the first carbonic acid
    dissociation constant (K1).
        1: M95 [DEFAULT].
        2: EG70 (GEOSECS).
        3: M83 (freshwater).
    opt_factor_k_HCO3: pressure correction for the second carbonic acid
    dissociation constant (K2).
        1: M95 [DEFAULT].
        2: EG70 (GEOSECS).
        3: M83 (freshwater).

    Other equilibrium constants
    ---------------------------
    opt_k_BOH3: parameterisation for boric acid equilibrium.
        1: D90b [DEFAULT].
        2: LTB69 (GEOSECS).
    opt_k_H2O: parameterisation for water dissociation.
        1: M95 [DEFAULT].
        2: M79 (GEOSECS).
        3: HO58 refit by M79 (freshwater).
    opt_k_HF: parameterisation for hydrogen fluoride dissociation.
        1: DR79 [DEFAULT].
        2: PF87.
    opt_k_HNO2: parameterisation for nitrous acid dissociation.
        1: BBWB24 for seawater [DEFAULT].
        2: BBWB24 for freshwater.
    opt_k_HSO4: parameterisation for bisulfate dissociation.
        1: D90a [DEFAULT].
        2: KRCB77.
        3: WM13/WMW14.
    opt_k_NH3: parameterisation for ammonium dissociation.
        1: CW95 [DEFAULT].
        2: YM95.
    opt_k_phosphate: parameterisation for the phosphoric acid dissocation
    constants.
        1: YM95 [DEFAULT].
        2: KP67 (GEOSECS).
    opt_k_Si: parameterisation for bisulfate dissociation.
        1: YM95 [DEFAULT].
        2: SMB64 (GEOSECS).
    opt_k_aragonite: parameterisation for aragonite solubility product.
        1: M83 [DEFAULT].
        2: ICHP73 (GEOSECS).
    opt_k_calcite: parameterisation for calcite solubility product.
        1: M83 [DEFAULT].
        2: I75 (GEOSECS).

    Other dissociation constant pressure corrections
    ------------------------------------------------
    opt_factor_k_BOH3: pressure correction for boric acid equilibrium.
        1: M79 [DEFAULT].
        2: EG70 (GEOSECS).
    opt_factor_k_H2O: pressure correction for water dissociation.
        1: M95 [DEFAULT].
        2: M83 (freshwater).

    Total salt contents
    -------------------
    These settings are ignored if their values are provided directly as kwargs.

    opt_total_borate: for calculating `total_borate` from `salinity`.
        1: U74 [DEFAULT].
        2: LKB10.
        3: KSK18 (Baltic Sea).
    opt_Ca: for calculating `Ca` from `salinity`.
        1: RT67 [DEFAULT].
        2: C65 (GEOSECS).

    Other settings
    --------------
    opt_gas_constant: the universal gas constant (R)
        1: DOEv2 (consistent with pre-July 2020 CO2SYS software).
        2: DOEv3.
        3: 2018 CODATA [DEFAULT].
    opt_fugacity_factor: how to convert between partial pressure and fugacity.
        1: with a fugacity factor [DEFAULT].
        2: assuming they are equal (GEOSECS).
    opt_HCO3_root: with known `dic` and `HCO3`, which root to solve for.
        1: the lower pH root.
        2: the higher pH root [DEFAULT].

    Equilibrium constants
    ---------------------
    Any equilibrium constant calculated within PyCO2SYS can be provided
    directly as an input instead.  The kwarg should use the same key as would
    be used to solve for this parameter.  See the docs for `CO2System.solve`
    for more detail.
    """
    # Check for double precision
    if np.array(1.0).dtype == np.dtype("float32"):
        warn(
            "JAX does not appear to be using double precision - "
            + "set the environment variable `JAX_ENABLE_X64=True`",
            stacklevel=2,
        )
    # Merge data with kwargs
    pd_index = None
    xr_dims = None
    xr_shape = None
    data_is_dict = isinstance(data, dict)
    kwargs_data = {}
    if data is not None:
        # First, check for string kwargs, which indicate keys in data that need
        # renaming
        renamer_user = {}
        for k, v in kwargs.items():
            if isinstance(v, str):
                if v in renamer_user:
                    # Can't repeat keys e.g. `data=df, dic="var", pH="var"`
                    raise PyCO2SYSError(
                        f'"{v}" cannot be used for {k} because'
                        + f" it is already being used for {renamer_user[v]}"
                    )
                else:
                    renamer_user[v] = shortcuts[k]
        # Next, go through keys of data and get shortcuts or renames for them
        renamer_data = {}
        for k in data:
            if k in renamer_user:
                renamer_data[k] = renamer_user[k]
            else:
                renamer_data[k] = shortcuts[k]
        # Check for duplicates
        renamer_values = []
        for v in renamer_data.values():
            if v in renamer_values:
                raise PyCO2SYSError(
                    f"data contains multiple keys corresponding to {v}, "
                    + "possibly under different shortcuts"
                )
            else:
                renamer_values.append(v)
        # Rename keys if data is a dict
        if data_is_dict:
            for k, v in renamer_data.items():
                kwargs_data[v] = data[k]
        # If `data` isn't a dict, it might be a pandas df
        else:
            data_is_pandas = False
            try:
                import pandas as pd

                data_is_pandas = isinstance(data, pd.DataFrame)
                # If `data` is a pandas df, we need to rename string keys,
                # convert Series to numpy arrays, and save the df index
                if data_is_pandas:
                    pd_index = data.index.copy()
                    for c in data.columns:
                        if c in renamer_data:
                            kwargs_data[renamer_data[c]] = data[c].to_numpy()
            except ImportError:
                warn(
                    "pandas could not be imported - ignoring data.",
                    stacklevel=2,
                )
            data_is_xarray = False
            if not data_is_pandas:
                try:
                    import xarray as xr

                    data_is_xarray = isinstance(data, xr.Dataset)
                    # If `data` is an xarray ds, we need to rename string keys,
                    # convert DataArrays to numpy arrays, and store all the
                    # dimensions
                    if data_is_xarray:
                        xr_dims = list(data.sizes.keys())
                        xr_shape = list(data.sizes.values())
                        for k, v in data.items():
                            if k in renamer_data:
                                kwargs_data[renamer_data[k]] = da_to_array(
                                    v, xr_dims
                                )
                except ImportError:
                    warn(
                        "xarray could not be imported - ignoring data.",
                        stacklevel=2,
                    )
                if not data_is_xarray:
                    # If we reach this point, `data` is neither dict nor
                    # pandas df nor xarray ds, so it's ignored
                    warn(
                        "Type of data not recognised - it will be ignored.",
                        stacklevel=2,
                    )
    else:
        for k, v in kwargs.items():
            if isinstance(v, str):
                raise PyCO2SYSError(
                    "Arguments cannot be provided as strings"
                    + f" when data=None ({k})."
                )
    # Check there aren't any duplicate kwargs with different aliases, and drop
    # any kwargs that are strings (used to identify `data` columns)
    kwargs_nodups = {}
    for k, v in kwargs.items():
        if shortcuts[k] in kwargs_nodups:
            raise PyCO2SYSError(
                f"Repeated kwarg, possibly under a different shortcut: {k}"
            )
        elif not isinstance(v, str):
            kwargs_nodups[shortcuts[k]] = v
            if shortcuts[k] in kwargs_data:
                warn(
                    f"{shortcuts[k]} found in both data and kwargs, "
                    + "possibly under different shortcuts - using the "
                    + "kwargs value",
                    stacklevel=2,
                )
    # Merge data and user kwargs
    kwargs_data.update(kwargs_nodups)
    # Parse kwargs
    for k, v in kwargs_data.copy().items():
        # Convert lists to numpy arrays
        if isinstance(kwargs_data[k], list):
            kwargs_data[k] = np.array(kwargs_data[k])
        # Convert None to np.nan
        if kwargs_data[k] is None:
            kwargs_data[k] = np.nan
        # If an opts is not scalar, take only the first value
        if k in opts_default:
            if np.isscalar(kwargs_data[k]):
                try:
                    kwargs_data[k] = kwargs_data[k].item()
                except (AttributeError, ValueError):
                    pass
            else:
                raise PyCO2SYSError(
                    f"{k} (and all other opts) must be scalar."
                )
            if isinstance(kwargs_data[k], float):
                if kwargs_data[k] == int(kwargs_data[k]):
                    kwargs_data[k] = int(kwargs_data[k])
                else:
                    raise PyCO2SYSError(
                        f"{k} (and all other opts) must be an integer."
                    )
        # For non-opts
        else:
            # Downgrade pd.Series and xr.DataArray to numpy arrays without
            # importing pandas or xarray---but the user should avoid doing this
            # because it doesn't take care of indices properly
            try:
                _ = kwargs_data[k].values
                raise PyCO2SYSError(
                    f"{k} provided as a pd.Series or xr.DataArray, "
                    + "which is not allowed."
                )
            except AttributeError:
                pass
            # Convert ints to floats
            if isinstance(kwargs_data[k], int):
                kwargs_data[k] = float(kwargs_data[k])
            elif hasattr(kwargs_data[k], "dtype"):
                try:
                    kwargs_data[k] = kwargs_data[k].astype(float)
                except ValueError:
                    pass
            # Turn not-allowed negatives to NaN
            if (
                k not in ["alkalinity", "pH", "temperature"]
                and not k.startswith("pH_")
                and not k.startswith("pk_")
                and not k.startswith("pkt_")
            ):
                kwargs_data[k] = np.where(
                    kwargs_data[k] < 0, np.nan, kwargs_data[k]
                )
    opts = {k: v for k, v in kwargs_data.items() if k in opts_default}
    opts = opts_default | opts
    data = {
        shortcuts[k]: v
        for k, v in kwargs_data.items()
        if k not in opts_default
    }
    # Get icase
    core_known = np.array([v in data for v in parameters_core])
    icase_all = np.arange(1, len(parameters_core) + 1)
    icase = icase_all[core_known]
    if len(icase) > 2:
        icase_params = [
            parameters_core[i] for i, k in enumerate(core_known) if k
        ]
        raise PyCO2SYSError(
            "A maximum of 2 known core parameters can be provided"
            + f" (you provided: {icase_params[0]}"
            + (", {}" * (len(icase_params) - 1)).format(*icase_params[1:])
            + ")."
        )
    if len(icase) == 0:
        icase = np.array(0)
    elif len(icase) == 2:
        icase = icase[0] * 100 + icase[1]
    icase = icase.item()
    # Assemble relevant functions
    if icase not in get_funcs_core:
        icase_params = [
            parameters_core[i] for i, k in enumerate(core_known) if k
        ]
        if len(icase_params) == 1:
            raise PyCO2SYSError(
                "A second known core parameter must be provided "
                + f"together with {icase_params[0]}."
            )
        else:
            raise PyCO2SYSError(
                "{} and {}".format(*icase_params)
                + " is not a valid pair of core parameters."
            )
    funcs = get_funcs | get_funcs_core[icase]
    for opt, v in opts.items():
        # opt_HCO3_root is available only for icase == 207 (known DIC & HCO3)
        if not (opt == "opt_HCO3_root" and icase != 207):
            try:
                funcs.update(get_funcs_opts[opt][v])
            except KeyError:
                raise PyCO2SYSError(f"{v} is not a valid option for {opt}.")
    # Add defaults that depend on opts (i.e., coeffs)
    defaults = values_default.copy()
    for opt, v in get_coeffs_opts.items():
        defaults.update(v[opts[opt]])
    # If pH is not accessible, we can't calculate it on different scales
    if icase < 100 and icase not in [3]:
        pH_vars = ["pH", "pH_total", "pH_sws", "pH_free", "pH_nbs"]
        for v in pH_vars:
            if v in funcs:
                funcs.pop(v)
    co2s = CO2System(
        funcs=funcs,
        shortcuts=shortcuts,
        defaults=defaults,
        no_store=exclude_on_store_steps_1,
        icase=icase,
        opts=opts,
        pd_index=pd_index,
        xr_dims=xr_dims,
        xr_shape=xr_shape,
    ).set_data(**data)
    return co2s
