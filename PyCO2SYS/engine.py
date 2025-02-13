# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2025  Matthew P. Humphreys et al.  (GNU GPLv3)
import itertools
import warnings
from collections import UserDict
from inspect import signature

import networkx as nx
from jax import numpy as np
from jaxlib.xla_extension import ArrayImpl

from . import (
    bio,
    buffers,
    constants,
    convert,
    equilibria,
    gas,
    meta,
    salts,
    solubility,
    solve,
    upsilon,
)

# Define functions for calculations that depend neither on icase nor opts:
get_funcs = {
    # Total salt contents
    "ionic_strength": salts.ionic_strength_DOE94,
    "total_fluoride": salts.total_fluoride_R65,
    "total_sulfate": salts.total_sulfate_MR66,
    # Equilibrium constants at 1 atm and on reported pH scale
    "k_CO2_1atm": equilibria.p1atm.k_CO2_W74,
    "k_H2S_total_1atm": equilibria.p1atm.k_H2S_total_YM95,
    # pH scale conversion factors at 1 atm
    "free_to_sws_1atm": lambda total_fluoride,
    total_sulfate,
    k_HF_free_1atm,
    k_HSO4_free_1atm: convert.pH_free_to_sws(
        total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm
    ),
    "nbs_to_sws": convert.pH_nbs_to_sws,  # because fH doesn't get pressure-corrected
    "tot_to_sws_1atm": lambda total_fluoride,
    total_sulfate,
    k_HF_free_1atm,
    k_HSO4_free_1atm: convert.pH_tot_to_sws(
        total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm
    ),
    # Equilibrium constants at 1 atm and on the seawater pH scale
    "k_H2S_sws_1atm": lambda k_H2S_total_1atm, tot_to_sws_1atm: (
        k_H2S_total_1atm * tot_to_sws_1atm
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
    "factor_k_HNO2": lambda: 1.0,  # unknown!
    # Equilibrium constants at pressure and on the free pH scale
    "k_HF_free": lambda k_HF_free_1atm, factor_k_HF: k_HF_free_1atm * factor_k_HF,
    "k_HSO4_free": lambda k_HSO4_free_1atm, factor_k_HSO4: (
        k_HSO4_free_1atm * factor_k_HSO4
    ),
    # Equilibrium constants at pressure and on the seawater pH scale
    "k_BOH3_sws": lambda k_BOH3_sws_1atm, factor_k_BOH3: (
        k_BOH3_sws_1atm * factor_k_BOH3
    ),
    "k_H2O_sws": lambda k_H2O_sws_1atm, factor_k_H2O: k_H2O_sws_1atm * factor_k_H2O,
    "k_H2S_sws": lambda k_H2S_sws_1atm, factor_k_H2S: k_H2S_sws_1atm * factor_k_H2S,
    "k_H3PO4_sws": lambda k_H3PO4_sws_1atm, factor_k_H3PO4: (
        k_H3PO4_sws_1atm * factor_k_H3PO4
    ),
    "k_H2PO4_sws": lambda k_H2PO4_sws_1atm, factor_k_H2PO4: (
        k_H2PO4_sws_1atm * factor_k_H2PO4
    ),
    "k_HPO4_sws": lambda k_HPO4_sws_1atm, factor_k_HPO4: (
        k_HPO4_sws_1atm * factor_k_HPO4
    ),
    "k_Si_sws": lambda k_Si_sws_1atm, factor_k_Si: k_Si_sws_1atm * factor_k_Si,
    "k_NH3_sws": lambda k_NH3_sws_1atm, factor_k_NH3: k_NH3_sws_1atm * factor_k_NH3,
    "k_H2CO3_sws": lambda k_H2CO3_sws_1atm, factor_k_H2CO3: (
        k_H2CO3_sws_1atm * factor_k_H2CO3
    ),
    "k_HCO3_sws": lambda k_HCO3_sws_1atm, factor_k_HCO3: (
        k_HCO3_sws_1atm * factor_k_HCO3
    ),
    "k_HNO2_sws": lambda k_HNO2_sws_1atm, factor_k_HNO2: (
        k_HNO2_sws_1atm * factor_k_HNO2
    ),
    # Equilibrium constants at pressure and on the requested pH scale
    "k_CO2": lambda k_CO2_1atm, factor_k_CO2: k_CO2_1atm * factor_k_CO2,
    "k_BOH3": lambda sws_to_opt, k_BOH3_sws: sws_to_opt * k_BOH3_sws,
    "k_H2O": lambda sws_to_opt, k_H2O_sws: sws_to_opt * k_H2O_sws,
    "k_H2S": lambda sws_to_opt, k_H2S_sws: sws_to_opt * k_H2S_sws,
    "k_H3PO4": lambda sws_to_opt, k_H3PO4_sws: sws_to_opt * k_H3PO4_sws,
    "k_H2PO4": lambda sws_to_opt, k_H2PO4_sws: sws_to_opt * k_H2PO4_sws,
    "k_HPO4": lambda sws_to_opt, k_HPO4_sws: sws_to_opt * k_HPO4_sws,
    "k_Si": lambda sws_to_opt, k_Si_sws: sws_to_opt * k_Si_sws,
    "k_NH3": lambda sws_to_opt, k_NH3_sws: sws_to_opt * k_NH3_sws,
    "k_H2CO3": lambda sws_to_opt, k_H2CO3_sws: sws_to_opt * k_H2CO3_sws,
    "k_HCO3": lambda sws_to_opt, k_HCO3_sws: sws_to_opt * k_HCO3_sws,
    "k_HNO2": lambda sws_to_opt, k_HNO2_sws: sws_to_opt * k_HNO2_sws,
    # Gasses
    "vp_factor": gas.vpfactor,
    # Mg-calcite solubility
    "acf_Ca": solubility.get_activity_coefficient_Ca,
    "acf_Mg": solubility.get_activity_coefficient_Mg,
    "acf_CO3": solubility.get_activity_coefficient_CO3,
    "k_Mg_calcite_1atm": solubility.get_k_Mg_calcite_1atm,
    "k_Mg_calcite": solubility.get_k_Mg_calcite,
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
    "H": lambda pH: 10.0**-pH,
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
get_funcs_opts["opt_gas_constant"] = {
    1: dict(gas_constant=lambda: constants.RGasConstant_DOEv2),
    2: dict(gas_constant=lambda: constants.RGasConstant_DOEv3),
    3: dict(gas_constant=lambda: constants.RGasConstant_CODATA2018),
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
        k_H2CO3_total_1atm=equilibria.p1atm.k_H2CO3_total_RRV93,
        k_HCO3_total_1atm=equilibria.p1atm.k_HCO3_total_RRV93,
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, tot_to_sws_1atm: (
            k_H2CO3_total_1atm * tot_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, tot_to_sws_1atm: (
            k_HCO3_total_1atm * tot_to_sws_1atm
        ),
    ),
    2: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_GP89,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_GP89,
    ),
    3: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_H73_DM87,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_H73_DM87,
    ),
    4: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_MCHP73_DM87,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_MCHP73_DM87,
    ),
    5: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_HM_DM87,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_HM_DM87,
    ),
    6: dict(
        k_H2CO3_nbs_1atm=equilibria.p1atm.k_H2CO3_nbs_MCHP73,
        k_HCO3_nbs_1atm=equilibria.p1atm.k_HCO3_nbs_MCHP73,
        k_H2CO3_sws_1atm=lambda k_H2CO3_nbs_1atm, nbs_to_sws: (
            k_H2CO3_nbs_1atm * nbs_to_sws
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_nbs_1atm, nbs_to_sws: (
            k_HCO3_nbs_1atm * nbs_to_sws
        ),
    ),
    # 7: same as 6; see note at end
    8: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_M79,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_M79,
    ),
    9: dict(
        k_H2CO3_nbs_1atm=equilibria.p1atm.k_H2CO3_nbs_CW98,
        k_HCO3_nbs_1atm=equilibria.p1atm.k_HCO3_nbs_CW98,
        k_H2CO3_sws_1atm=lambda k_H2CO3_nbs_1atm, nbs_to_sws: (
            k_H2CO3_nbs_1atm * nbs_to_sws
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_nbs_1atm, nbs_to_sws: (
            k_HCO3_nbs_1atm * nbs_to_sws
        ),
    ),
    10: dict(
        k_H2CO3_total_1atm=equilibria.p1atm.k_H2CO3_total_LDK00,
        k_HCO3_total_1atm=equilibria.p1atm.k_HCO3_total_LDK00,
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, tot_to_sws_1atm: (
            k_H2CO3_total_1atm * tot_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, tot_to_sws_1atm: (
            k_HCO3_total_1atm * tot_to_sws_1atm
        ),
    ),
    11: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_MM02,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_MM02,
    ),
    12: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_MPL02,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_MPL02,
    ),
    13: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_MGH06,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_MGH06,
    ),
    14: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_M10,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_M10,
    ),
    15: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_WMW14,
        k_HCO3_sws_1atm=equilibria.p1atm.k_HCO3_sws_WMW14,
    ),
    16: dict(
        k_H2CO3_total_1atm=equilibria.p1atm.k_H2CO3_total_SLH20,
        k_HCO3_total_1atm=equilibria.p1atm.k_HCO3_total_SLH20,
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, tot_to_sws_1atm: (
            k_H2CO3_total_1atm * tot_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, tot_to_sws_1atm: (
            k_HCO3_total_1atm * tot_to_sws_1atm
        ),
    ),
    17: dict(
        # k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_WMW14,
        # ^ although the above should work, it gives slightly different answers
        #   than he conversion below, and below is consistent with the MATLAB
        #   implementation
        k_H2CO3_total_1atm=equilibria.p1atm.k_H2CO3_total_WMW14,
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, tot_to_sws_1atm: (
            k_H2CO3_total_1atm * tot_to_sws_1atm
        ),
        k_HCO3_total_1atm=equilibria.p1atm.k_HCO3_total_SB21,
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, tot_to_sws_1atm: (
            k_HCO3_total_1atm * tot_to_sws_1atm
        ),
    ),
    18: dict(
        k_H2CO3_total_1atm=equilibria.p1atm.k_H2CO3_total_PLR18,
        k_HCO3_total_1atm=equilibria.p1atm.k_HCO3_total_PLR18,
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, tot_to_sws_1atm: (
            k_H2CO3_total_1atm * tot_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, tot_to_sws_1atm: (
            k_HCO3_total_1atm * tot_to_sws_1atm
        ),
    ),
}
# For historical reasons, these are the same as each other (one also gets the Peng
# "correction", but that's handled elsewhere):
get_funcs_opts["opt_k_carbonic"][7] = get_funcs_opts["opt_k_carbonic"][6].copy()
get_funcs_opts["opt_k_phosphate"] = {
    1: dict(
        k_H3PO4_sws_1atm=equilibria.p1atm.k_H3PO4_sws_YM95,
        k_H2PO4_sws_1atm=equilibria.p1atm.k_H2PO4_sws_YM95,
        k_HPO4_sws_1atm=equilibria.p1atm.k_HPO4_sws_YM95,
    ),
    2: dict(
        k_H3PO4_sws_1atm=equilibria.p1atm.k_H3PO4_sws_KP67,
        k_H2PO4_nbs_1atm=equilibria.p1atm.k_H2PO4_nbs_KP67,
        k_H2PO4_sws_1atm=lambda k_H2PO4_nbs_1atm, nbs_to_sws: (
            k_H2PO4_nbs_1atm * nbs_to_sws
        ),
        k_HPO4_nbs_1atm=equilibria.p1atm.k_HPO4_nbs_KP67,
        k_HPO4_sws_1atm=lambda k_HPO4_nbs_1atm, nbs_to_sws: (
            k_HPO4_nbs_1atm * nbs_to_sws
        ),
    ),
}
get_funcs_opts["opt_k_BOH3"] = {
    1: dict(
        k_BOH3_total_1atm=equilibria.p1atm.k_BOH3_total_D90b,
        k_BOH3_sws_1atm=lambda k_BOH3_total_1atm, tot_to_sws_1atm: (
            k_BOH3_total_1atm * tot_to_sws_1atm
        ),
    ),
    2: dict(
        k_BOH3_nbs_1atm=equilibria.p1atm.k_BOH3_nbs_LTB69,
        k_BOH3_sws_1atm=lambda k_BOH3_nbs_1atm, nbs_to_sws: (
            k_BOH3_nbs_1atm * nbs_to_sws
        ),
    ),
}
get_funcs_opts["opt_k_H2O"] = {
    1: dict(k_H2O_sws_1atm=equilibria.p1atm.k_H2O_sws_M95),
    2: dict(k_H2O_sws_1atm=equilibria.p1atm.k_H2O_sws_M79),
    3: dict(k_H2O_sws_1atm=equilibria.p1atm.k_H2O_sws_HO58_M79),
}
get_funcs_opts["opt_k_HF"] = {
    1: dict(k_HF_free_1atm=equilibria.p1atm.k_HF_free_DR79),
    2: dict(k_HF_free_1atm=equilibria.p1atm.k_HF_free_PF87),
}
get_funcs_opts["opt_k_HSO4"] = {
    1: dict(k_HSO4_free_1atm=equilibria.p1atm.k_HSO4_free_D90a),
    2: dict(k_HSO4_free_1atm=equilibria.p1atm.k_HSO4_free_KRCB77),
    3: dict(k_HSO4_free_1atm=equilibria.p1atm.k_HSO4_free_WM13),
}
get_funcs_opts["opt_k_NH3"] = {
    1: dict(
        k_NH3_total_1atm=equilibria.p1atm.k_NH3_total_CW95,
        k_NH3_sws_1atm=lambda k_NH3_total_1atm, tot_to_sws_1atm: (
            k_NH3_total_1atm * tot_to_sws_1atm
        ),
    ),
    2: dict(k_NH3_sws_1atm=equilibria.p1atm.k_NH3_sws_YM95),
}
get_funcs_opts["opt_k_Si"] = {
    1: dict(k_Si_sws_1atm=equilibria.p1atm.k_Si_sws_YM95),
    2: dict(
        k_Si_nbs_1atm=equilibria.p1atm.k_Si_nbs_SMB64,
        k_Si_sws_1atm=lambda k_Si_nbs_1atm, nbs_to_sws: (k_Si_nbs_1atm * nbs_to_sws),
    ),
}
get_funcs_opts["opt_k_HNO2"] = {
    1: dict(
        k_HNO2_total_1atm=equilibria.p1atm.k_HNO2_total_BBWB24,
        k_HNO2_sws_1atm=lambda k_HNO2_total_1atm, tot_to_sws_1atm: (
            k_HNO2_total_1atm * tot_to_sws_1atm
        ),
    ),
    2: dict(
        k_HNO2_nbs_1atm=equilibria.p1atm.k_HNO2_nbs_BBWB24_freshwater,
        k_HNO2_sws_1atm=lambda k_HNO2_nbs_1atm, nbs_to_sws: (
            k_HNO2_nbs_1atm * nbs_to_sws
        ),
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
        sws_to_opt=lambda: 1.0,
        opt_to_free=convert.pH_sws_to_free,
        opt_to_tot=convert.pH_sws_to_tot,
        opt_to_nbs=convert.pH_sws_to_nbs,
    ),
    3: dict(  # free
        sws_to_opt=convert.pH_sws_to_free,
        opt_to_free=lambda: 1.0,
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
# TODO these below can be added only if there is a pH accessible!
# While also depending on an opt!  See also below TODO for fCO2
# i.e. icase == 3 or icase > 100
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
        funcs.update(dict(pH_total=lambda pH, opt_to_tot: pH - np.log10(opt_to_tot)))
    if o in [1, 3, 4]:
        funcs.update(dict(pH_sws=lambda pH, opt_to_sws: pH - np.log10(opt_to_sws)))
    if o in [1, 2, 4]:
        funcs.update(dict(pH_free=lambda pH, opt_to_free: pH - np.log10(opt_to_free)))
    if o in [1, 2, 3]:
        funcs.update(dict(pH_nbs=lambda pH, opt_to_nbs: pH - np.log10(opt_to_nbs)))
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
get_funcs_opts["opt_fugacity_factor"] = {
    1: dict(fugacity_factor=gas.fugacity_factor),
    2: dict(fugacity_factor=lambda: 1.0),  # for GEOSECS
}
get_funcs_opts["opt_HCO3_root"] = {  # only added if icase == 207
    1: dict(pH=solve.inorganic.pH_from_dic_HCO3_lo),
    2: dict(pH=solve.inorganic.pH_from_dic_HCO3_hi),  # for typical seawater
}
get_funcs_opts["opt_k_calcite"] = {
    1: dict(k_calcite=solubility.k_calcite_M83),
    2: dict(k_calcite=solubility.k_calcite_I75),  # for GEOSECS
}
get_funcs_opts["opt_k_aragonite"] = {
    1: dict(k_aragonite=solubility.k_aragonite_M83),
    2: dict(k_aragonite=solubility.k_aragonite_GEOSECS),  # for GEOSECS
}
# TODO option 1 below can only be added if there is an fCO2 value accessible
# (see also similar TODO above about pH)
get_funcs_opts["opt_fCO2_temperature"] = {
    1: dict(
        bh=upsilon.get_bh_H24,
        upsilon=upsilon.inverse,
    ),
    2: dict(
        bl=lambda: upsilon.bl_TOG93,
        upsilon=upsilon.linear,
    ),
    3: dict(
        aq=lambda: upsilon.aq_TOG93,
        bq=lambda: upsilon.bq_TOG93,
        upsilon=upsilon.quadratic,
    ),
}
get_funcs_opts["opt_Mg_calcite_type"] = {
    1: dict(kt_Mg_calcite_25C_1atm=solubility.get_kt_Mg_calcite_25C_1atm_minprep),
    2: dict(kt_Mg_calcite_25C_1atm=solubility.get_kt_Mg_calcite_25C_1atm_biogenic),
    3: dict(kt_Mg_calcite_25C_1atm=solubility.get_kt_Mg_calcite_25C_1atm_synthetic),
}
get_funcs_opts["opt_Mg_calcite_kt_Tdep"] = {
    1: dict(kt_Mg_calcite_1atm=solubility.get_kt_Mg_calcite_1atm_vantHoff),
    2: dict(kt_Mg_calcite_1atm=solubility.get_kt_Mg_calcite_1atm_PB82),
}

# Automatically set up graph for calculations that depend neither on icase nor opts
# based on the function names and signatures in get_funcs
graph_fixed = nx.DiGraph()
for k, func in get_funcs.items():
    for f in signature(func).parameters.keys():
        graph_fixed.add_edge(f, k)

# Automatically set up graph for each icase based on the function names and signatures
# in get_funcs_core
graph_core = {}
for icase, funcs in get_funcs_core.items():
    graph_core[icase] = nx.DiGraph()
    for t, func in get_funcs_core[icase].items():
        for f in signature(func).parameters.keys():
            graph_core[icase].add_edge(f, t)


def get_graph_opts(exclude=[]):
    """Automatically set up graph for each opt based on the function names and
    signatures in ``get_funcs_opts``.
    """
    graph_opts = {}
    for o, opts in get_funcs_opts.items():
        if o not in exclude:
            graph_opts[o] = {}
            for opt, funcs in opts.items():
                graph_opts[o][opt] = nx.DiGraph()
                for k, func in funcs.items():
                    for f in signature(func).parameters.keys():
                        graph_opts[o][opt].add_edge(f, k)
    return graph_opts


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
    "Mg_percent": 0.0,  # %
    "pressure_atmosphere": 1.0,  # atm
    "pressure": 0.0,  # dbar
    "salinity": 35.0,
    "temperature": 25.0,  # °C
    "total_ammonia": 0.0,  # µmol/kg-sw
    "total_phosphate": 0.0,  # µmol/kg-sw
    "total_silicate": 0.0,  # µmol/kg-sw
    "total_sulfide": 0.0,  # µmol/kg-sw
    "total_nitrite": 0.0,  # µmol/kg-sw
}

opts_default = {
    "opt_Ca": 1,
    "opt_factor_k_BOH3": 1,
    "opt_factor_k_H2CO3": 1,
    "opt_factor_k_H2O": 1,
    "opt_factor_k_HCO3": 1,
    "opt_fCO2_temperature": 1,
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

# Define labels for parameter plotting
set_node_labels = {
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
    "free_to_sws_1atm": r"$_\mathrm{F}^\mathrm{S}Y^0$",
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
    "HS": r"$[\mathrm{HS}^–]$",
    "HSO4": r"$[\mathrm{HSO}_4^–]$",
    "ionic_strength": "$I$",
    "k_aragonite": r"$K_\mathrm{a}^*$",
    "k_BOH3_sws_1atm": r"$K_\mathrm{B}^\mathrm{S0}$",
    "k_BOH3_sws": r"$K_\mathrm{B}^\mathrm{S}$",
    "k_BOH3_total_1atm": r"$K_\mathrm{B}^\mathrm{T0}$",
    "k_BOH3": r"$K_\mathrm{B}^*$",
    "k_calcite": r"$K_\mathrm{c}^*$",
    "k_CO2_1atm": "$K_0′^0$",
    "k_CO2": "$K_0′$",
    "k_H2CO3_sws_1atm": r"$K_1^\mathrm{S0}$",
    "k_H2CO3_sws": "$K_1^s$",
    "k_H2CO3_total_1atm": r"$K_1^\mathrm{T0}$",
    "k_H2CO3": "$K_1^*$",
    "k_H2O_sws_1atm": r"$K_w^\mathrm{S0}$",
    "k_H2O_sws": r"$K_w^\mathrm{S}$",
    "k_H2O": "$K_w^*$",
    "k_H2PO4_sws_1atm": r"$K_\mathrm{P2}^\mathrm{S0}$",
    "k_H2PO4_sws": r"$K_\mathrm{P2}^\mathrm{S}$",
    "k_H2PO4": r"$K_\mathrm{P2}^*$",
    "k_H2S_sws_1atm": r"$K_\mathrm{H_2S}^\mathrm{S0}$",
    "k_H2S_sws": r"$K_\mathrm{H_2S}^\mathrm{S}$",
    "k_H2S_total_1atm": r"$K_\mathrm{H_2S}^\mathrm{T0}$",
    "k_H2S": r"$K_\mathrm{H_2S}^*$",
    "k_H3PO4_sws_1atm": r"$K_\mathrm{P1}^\mathrm{S0}$",
    "k_H3PO4_sws": r"$K_\mathrm{P1}^\mathrm{S}$",
    "k_H3PO4": r"$K_\mathrm{P1}^*$",
    "k_HCO3_sws_1atm": r"$K_2^\mathrm{S0}$",
    "k_HCO3_sws": "$K_2^s$",
    "k_HCO3_total_1atm": r"$K_2^\mathrm{T0}$",
    "k_HCO3": "$K_2^*$",
    "k_HF_free_1atm": r"$K_\mathrm{HF}^\mathrm{F0}$",
    "k_HF_free": r"$K_\mathrm{HF}^\mathrm{F}$",
    "k_HNO2_sws_1atm": r"$K_\mathrm{HNO_2}^\mathrm{S0}$",
    "k_HNO2_sws": r"$K_\mathrm{HNO_2}^\mathrm{S}$",
    "k_HNO2_total_1atm": r"$K_\mathrm{HNO_2}^\mathrm{T0}$",
    "k_HNO2": r"$K_\mathrm{HNO_2}^*$",
    "k_HPO4_sws_1atm": r"$K_\mathrm{P3}^\mathrm{S0}$",
    "k_HPO4_sws": r"$K_\mathrm{P3}^\mathrm{S}$",
    "k_HPO4": r"$K_\mathrm{P3}^*$",
    "k_HSO4_free_1atm": r"$K_\mathrm{HSO_4}^\mathrm{F0}$",
    "k_HSO4_free": r"$K_\mathrm{HSO_4}^\mathrm{F}$",
    "k_NH3_sws_1atm": r"$K_\mathrm{NH_3}^\mathrm{S0}$",
    "k_NH3_sws": r"$K_\mathrm{NH_3}^\mathrm{S}$",
    "k_NH3_total_1atm": r"$K_\mathrm{NH_3}^\mathrm{T0}$",
    "k_NH3": r"$K_\mathrm{NH_3}^*$",
    "k_Si_sws_1atm": r"$K_\mathrm{Si}^\mathrm{S0}$",
    "k_Si_sws": r"$K_\mathrm{Si}^\mathrm{S}$",
    "k_Si": r"$K_\mathrm{Si}^*$",
    "Mg_percent": "Mg%",
    "Mg": r"$[\mathrm{Mg}^{2+}]$",
    "nbs_to_sws": r"$_\mathrm{N}^\mathrm{S}Y$",
    "NH3": r"$[\mathrm{NH}_3]$",
    "NH4": r"$[\mathrm{NH}_4^+]$",
    "OH": r"$[\mathrm{OH}^–]$",
    "omega_alkalinity": r"$\omega_{A_\mathrm{T}}$",
    "omega_dic": r"$\omega_{C_\mathrm{T}}$",
    "opt_to_free": r"$_*^\mathrm{F}Y$",
    "opt_to_nbs": r"$_*^\mathrm{N}Y$",
    "opt_to_sws": r"$_*^\mathrm{S}Y$",
    "pCO2": r"$p\mathrm{CO}_2$",
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
    "sws_to_opt": r"$_\mathrm{S}^*Y$",
    "temperature": "$t$",
    "tot_to_sws_1atm": r"$_\mathrm{T}^\mathrm{S}Y^0$",
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
}

# Parameters that do not change between input and output conditions
condition_independent = (
    "alkalinity",
    "Ca",
    "dic",
    "ionic_strength",
    "Mg_percent",
    "salinity",
    "total_ammonia",
    "total_borate",
    "total_fluoride",
    "total_phosphate",
    "total_silicate",
    "total_sulfate",
    "total_sulfide",
    "total_nitrite",
)


def _remove_jax_overhead(d):
    for k, v in d.items():
        try:
            d[k] = v.item()
        except (AttributeError, ValueError):
            pass
        try:
            d[k] = v.__array__()
        except AttributeError:
            pass


class CO2System(UserDict):
    def __init__(self, pd_index=None, xr_dims=None, xr_shape=None, **kwargs):
        super().__init__()
        opts = {k: v for k, v in kwargs.items() if k.startswith("opt_")}
        data = {k: v for k, v in kwargs.items() if k not in opts}
        # Get icase
        core_known = np.array([v in data for v in parameters_core])
        icase_all = np.arange(1, len(parameters_core) + 1)
        icase = icase_all[core_known]
        assert len(icase) < 3, "A maximum of 2 known core parameters can be provided."
        if len(icase) == 0:
            icase = np.array(0)
        elif len(icase) == 2:
            icase = icase[0] * 100 + icase[1]
        self.icase = icase.item()
        self.opts = opts_default.copy()
        # Assign opts
        for k, v in opts.items():
            if k in get_funcs_opts:
                assert np.isscalar(v)
                assert v in get_funcs_opts[k].keys(), f"{v} is not allowed for {k}!"
            else:
                warnings.warn(f"'{k}' not recognised - it will be ignored.")
                opts.pop(k)
        self.opts.update(opts)
        # Deal with tricky special cases
        if self.icase != 207:
            self.opts.pop("opt_HCO3_root")
        if self.icase not in [0, 4, 5, 8, 9]:
            self.opts.pop("opt_fCO2_temperature")
        # Assemble graphs and computation functions
        self.graph, self.funcs, self.data = self._assemble(self.icase, data)
        self.grads = {}
        self.uncertainty = {}
        self.requested = set()  # keep track of all parameters that have been requested
        self.pd_index = pd_index
        if xr_dims is not None:
            assert xr_shape is not None
            assert len(xr_dims) == len(xr_shape)
        else:
            assert xr_shape is None
        self.xr_dims = xr_dims
        self.xr_shape = xr_shape
        self.state_colours = {
            0: "xkcd:grey",  # unknown
            1: "xkcd:grass",  # provided by user i.e. known but not calculated
            2: "xkcd:azure",  # calculated en route to a user-requested parameter
            3: "xkcd:tangerine",  # calculated after direct user request
        }
        self.valid_colours = {
            -1: "xkcd:light red",  # invalid
            0: "xkcd:light grey",  # unknown
            1: "xkcd:sky blue",  # valid
        }
        self.checked_valid = False

    def __getitem__(self, key):
        # When the user requests a dict key that hasn't been solved for yet, then
        # solve and provide the requested parameter
        self.solve(parameters=key)
        if isinstance(key, list):
            # If the user provides a list of keys to solve for, return all of them
            # as a dict
            return {k: self.data[k] for k in key}
        else:
            # If a single key is requested, return the corresponding value(s) directly
            return self.data[key]

    def __getattr__(self, attr):
        # This allows solved parameter values to be accessed with dot notation,
        # purely for convenience.
        # So, when the user tries to access something with dot notation...
        try:
            # ... then if it's an attribute, return it (this is the standard behaviour).
            return object.__getattribute__(self, attr)
        except AttributeError:
            # But if it's not an attribute...
            try:
                # ... return the corresponding parameter value, if it's already
                # been solved for...
                return self.data[attr]
            except KeyError:
                # ... but it if hasn't been solved for, throw an error.  The user
                # needs to use the normal dict notation (or solve method) to solve
                # for it.
                raise AttributeError(attr)

    def __setitem__(self, key, value):
        # Don't allow the user to assign new key-value pairs to the dict
        raise RuntimeError("Item assignment is not supported.")

    def _assemble(self, icase, data):
        # Deal with tricky special cases
        if icase == 207:
            graph_opts = get_graph_opts()
        else:
            graph_opts = get_graph_opts(exclude="opt_HCO3_root")
        # Assemble graph and functions
        graph = nx.compose(graph_fixed, graph_core[icase])
        funcs = get_funcs.copy()
        funcs.update(get_funcs_core[icase])
        for opt, v in self.opts.items():
            graph = nx.compose(graph, graph_opts[opt][v])
            funcs.update(get_funcs_opts[opt][v])
        # If fCO2 is not accessible, we can't calculate bh with
        # opt_fCO2_temperature = 1, so use a default constant bh value instead
        if icase < 100 and icase not in [4, 5, 8, 9]:
            graph.remove_nodes_from(["fCO2", "bh"])
            funcs["bh"] = lambda: upsilon.bh_TOG93_H24
            graph.add_edge("bh", "upsilon")
        # If pH is not accessible, we can't calculate it on different scales
        if icase < 100 and icase not in [3]:
            pH_vars = ["pH", "pH_total", "pH_sws", "pH_free", "pH_nbs"]
            for v in pH_vars:
                graph.remove_node(v)
                if v in funcs:
                    funcs.pop(v)
        # Save arguments
        to_remove = []
        for k, v in data.items():
            if v is not None:
                if k in graph.nodes:
                    # state 1 means that the value was provided as an argument
                    nx.set_node_attributes(graph, {k: 1}, name="state")
                else:
                    # TODO need to rethink how it is judged whether a value is
                    # allowed here --- things that are not part of the graph
                    # but that could be added as an isolated element should be
                    # kept?  Or could change the warning below
                    warnings.warn(f"'{k}' is not recognised - it will be ignored.")
                    to_remove.append(k)
        for k in to_remove:
            data.pop(k)
        # Assign default values
        for k, v in values_default.items():
            if k not in data and k in graph.nodes:
                data[k] = v
                nx.set_node_attributes(graph, {k: 1}, name="state")
        self.nodes_original = list(k for k, v in data.items() if v is not None)
        return graph, funcs, data

    def solve(self, parameters=None, store_steps=1):
        """Calculate parameter(s) and store them internally.

        Parameters
        ----------
        parameters : str or list of str, optional
            Which parameter(s) to calculate and store, by default None, in which case
            all possible parameters are calculated and stored internally.
        store_steps : int, optional
            Whether/which non-requested parameters calculated during intermediate
            calculation steps should be stored, by default 1.  The options are
                0 - store only the specifically requested parameters,
                1 - store the most used set of intermediate parameters, or
                2 - store the complete set of parameters.
        """
        # Parse user-provided parameters (if there are any)
        if parameters is None:
            # If no parameters are provided, then we solve for everything possible
            parameters = list(self.graph.nodes)
        elif isinstance(parameters, str):
            # Allow user to provide a string if only one parameter is desired
            parameters = [parameters]
        parameters = set(parameters)  # get rid of duplicates
        self.requested |= parameters
        self_data = self.data.copy()  # what was already known before this solve
        # Remove known nodes from a copy of self.graph, so that ancestors of known
        # nodes are not unnecessarily recomputed
        graph_unknown = self.graph.copy()
        graph_unknown.remove_nodes_from([k for k in self_data if k not in parameters])
        # Add intermediate parameters that we need to know in order to calculate
        # the requested parameters
        parameters_all = parameters.copy()
        for p in parameters:
            parameters_all = parameters_all | nx.ancestors(graph_unknown, p)
        # Convert the set of parameters into a list, exclude already-known ones,
        # and organise the list into the order required for calculations
        parameters_all = [
            p
            for p in nx.topological_sort(self.graph)
            if p in parameters_all and p not in self_data
        ]
        store_parameters = []
        for p in parameters_all:
            priors = self.graph.pred[p]
            if len(priors) == 0 or all([r in self_data for r in priors]):
                self_data[p] = self.funcs[p](
                    *[self_data[r] for r in signature(self.funcs[p]).parameters.keys()]
                )
                store_here = (
                    #  If store_steps is 0, store only requested parameters
                    (store_steps == 0 and p in parameters)
                    | (
                        # If store_steps is 1, store all but the equilibrium constants
                        # on the seawater scale, at 1 atm and their pressure-correction
                        # factors, and a few selected others
                        store_steps == 1
                        and not p.startswith("factor_k_")
                        and not (p.startswith("k_") and p.endswith("_sws"))
                        and not p.endswith("_1atm")
                        and p not in ["sws_to_opt", "opt_to_free", "ionic_strength"]
                    )
                    |  # If store_steps is 2, store everything
                    (store_steps == 2)
                )
                if store_here:
                    store_parameters.append(p)
                    if p in parameters:
                        # state = 3 means that the value was calculated internally
                        # due to direct request
                        nx.set_node_attributes(self.graph, {p: 3}, name="state")
                    else:
                        # state = 2 means that the value was calculated internally
                        # as an intermediate to a requested parameter
                        nx.set_node_attributes(self.graph, {p: 2}, name="state")
                    for f in signature(self.funcs[p]).parameters.keys():
                        nx.set_edge_attributes(self.graph, {(f, p): 2}, name="state")
        # Get rid of jax overhead on results
        self_data = {k: v for k, v in self_data.items() if k in store_parameters}
        _remove_jax_overhead(self_data)
        self.data.update(self_data)

    def to_pandas(self, parameters=None, store_steps=1):
        """Return parameters as a pandas `Series` or `DataFrame`.  All parameters should
        be scalar or one-dimensional vectors of the same size.

        Parameters
        ----------
        parameters : str or list of str, optional
            The parameter(s) to return.  These are solved for if not already available.
            If `None`, then all parameters that have already been solved for are
            returned.
        store_steps : int, optional
            See `solve`.

        Returns
        -------
        pd.Series or pd.DataFrame
            The parameter(s) as a `pd.Series` (if `parameters` is a `str`) or as a
            `pd.DataFrame` (if `parameters` is a `list`) with the original pandas index
            passed into the `CO2System` as `data`.  If `data` was not a `pd.DataFrame`
            then the default index will be used.
        """
        try:
            import pandas as pd

            if parameters is None:
                parameters = self.keys()
            self.solve(parameters=parameters, store_steps=store_steps)
            if isinstance(parameters, str):
                return pd.Series(data=self[parameters], index=self.pd_index)
            else:
                return pd.DataFrame(
                    {
                        p: pd.Series(
                            data=self[p] * np.ones(self.pd_index.shape),
                            index=self.pd_index,
                        )
                        for p in parameters
                    }
                )
        except ImportError:
            warnings.warn("pandas could not be imported.")

    def _get_xr_ndims(self, parameter):
        ndims = []
        if not np.isscalar(self[parameter]):
            for i, vs in enumerate(self[parameter].shape):
                if vs == self.xr_shape[i]:
                    ndims.append(self.xr_dims[i])
        return ndims

    def to_xarray(self, parameters=None, store_steps=1):
        """Return parameters as an xarray `DataArray` or `Dataset`.

        Parameters
        ----------
        parameters : str or list of str, optional
            The parameter(s) to return.  These are solved for if not already available.
            If `None`, then all parameters that have already been solved for are
            returned.
        store_steps : int, optional
            See `solve`.

        Returns
        -------
        xr.DataArray or xr.Dataset
            The parameter(s) as a `xr.DataArray` (if `parameters` is a `str`) or as a
            `xr.Dataset` (if `parameters` is a `list`) with the original xarray
            dimensions passed into the `CO2System` as `data`.  If `data` was not an
            `xr.Dataset` then this function will not work.
        """
        assert self.xr_dims is not None and self.xr_shape is not None, (
            "`data` was not provided as an `xr.Dataset` "
            + "when creating this `CO2System`."
        )
        try:
            import xarray as xr

            if parameters is None:
                parameters = self.keys()
            self.solve(parameters=parameters, store_steps=store_steps)
            if isinstance(parameters, str):
                ndims = self._get_xr_ndims(parameters)
                return xr.DataArray(np.squeeze(self[parameters]), dims=ndims)
            else:
                return xr.Dataset(
                    {
                        p: xr.DataArray(np.squeeze(self[p]), dims=self._get_xr_ndims(p))
                        for p in parameters
                    }
                )
        except ImportError:
            warnings.warn("xarray could not be imported.")

    def _get_expUps(
        self,
        method_fCO2,
        temperature,
        bh_upsilon=None,
        opt_which_fCO2_insitu=1,
    ):
        if method_fCO2 in [1, 2, 3, 4]:
            self.solve("gas_constant")
        match method_fCO2:
            case 1:
                self.solve("fCO2", store_steps=0)
                fCO2 = self.fCO2
                assert opt_which_fCO2_insitu in [1, 2]
                if opt_which_fCO2_insitu == 2:
                    # If the output conditions are the environmental ones, then
                    # we need to provide an estimate of output fCO2 in order to
                    # use the bh parameterisation; we get this using the method_fCO2=2
                    # approach:
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
                    opt_which_fCO2_insitu=opt_which_fCO2_insitu,
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

    def adjust(
        self,
        temperature=None,
        pressure=None,
        store_steps=1,
        method_fCO2=1,
        opt_which_fCO2_insitu=1,
        bh_upsilon=None,
    ):
        """Adjust the system to a different temperature and/or pressure.

        Parameters
        ----------
        temperature : float, optional
            Temperature in °C to adjust to.  If `None`, temperature is not adjusted.
        pressure : float, optional
            Hydrostatic pressure in dbar to adjust to.  If `None`, pressure is
            not adjusted.
        store_steps : int, optional
            Whether/which non-requested parameters calculated during intermediate
            calculation steps should be stored.  The options are:

              - `0`: Store only the requested parameters.
              - `1`: Store the requested and most commonly used set of intermediate
              parameters (default).
              - `2`: Store the requested and complete set of intermediate parameters.
        method_fCO2 : int, optional
            If this is a single-parameter system, which method to use for the
            adjustment.  The options are:

              - `1`: parameterisation of H24 (default).
              - `2`: constant bh fitted to TOG93 dataset by H24.
              - `3`: constant theoretical bh of H24.
              - `4`: user-specified bh with the equations of H24.
              - `5`: linear fit of TOG93.
              - `6`: quadratic fit of TOG93.
        opt_which_fCO2_insitu : int, optional
            If this is a single-parameter system and `method_fCO2` is `1`, whether:
              - `1` the input condition (starting; default) or
              - `2` output condition (adjusted) temperature
            should be used to calculate $b_h$.
        bh_upsilon : float, optional
            If this is a single-parameter system and `method_fCO2` is `4`, then the
            value of $b_h$ in J/mol must be specified here.

        Returns
        -------
        CO2System
            A new `CO2System` with all values adjusted to the requested temperature(s)
            and/or pressure(s).
        """
        if self.icase > 100:
            # If we know (any) two MCS parameters, solve for alkalinity and DIC under
            # the "input" conditions
            self.solve(parameters=["alkalinity", "dic"], store_steps=store_steps)
            core = {k: self[k] for k in ["alkalinity", "dic"]}
            data = {}
        elif self.icase in [4, 5, 8, 9]:
            assert pressure is None, (
                "Cannot adjust pressure for a single-parameter system!"
            )
            # If we know only one of pCO2, fCO2, xCO2 or CO2(aq), first get fCO2 under
            # the "input" conditions
            self.solve(parameters="fCO2", store_steps=store_steps)
            core = {"fCO2": self.fCO2}
            # Then, convert this to the value at the new temperature using the requested
            # method
            assert method_fCO2 in range(1, 7), (
                "`method_fCO2` must be an integer from 1-6."
            )
            expUps = self._get_expUps(
                method_fCO2,
                temperature,
                bh_upsilon=bh_upsilon,
                opt_which_fCO2_insitu=opt_which_fCO2_insitu,
            )
            data = {"fCO2": core["fCO2"] * expUps}
        else:
            warnings.warn(
                "Single-parameter temperature adjustments are possible only "
                + "if the known parameter is one of pCO2, fCO2, xCO2 and CO2."
            )
        # Copy all parameters that are T/P independent into new data dict
        for k in condition_independent:
            if k in self.nodes_original:
                data[k] = self.data[k]
            elif k in core:
                data[k] = core[k]
        # Set temperature and/or pressure to adjusted value(s), unless they are None, in
        # which case, don't adjust
        if temperature is not None:
            data["temperature"] = temperature
        else:
            data["temperature"] = self.data["temperature"]
        if pressure is not None:
            data["pressure"] = pressure
        else:
            data["pressure"] = self.data["pressure"]
        sys = CO2System(
            **data,
            **self.opts,
            pd_index=self.pd_index,
            xr_dims=self.xr_dims,
            xr_shape=self.xr_shape,
        )
        sys.solve(parameters=self.data)
        return sys

    def _get_func_of(self, var_of):
        """Create a function to compute ``var_of`` directly from an input set of
        values.

        The created function has the signature

            value_of = get_value_of(**value)

        where the ``values`` are the originally user-defined values, obtained with
        either of the following:

            values_original = {k: sys.data[k] for k in sys.nodes_original}
            values_original = sys.get_values_original()
        """
        # We get a sub-graph of the node of interest and all its ancestors, excluding
        # originally fixed / user-defined values
        nodes_vo_all = nx.ancestors(self.graph, var_of)
        nodes_vo_all.add(var_of)
        # nodes_vo = onp.array([n for n in nodes_vo if n not in self.nodes_original])
        nodes_vo = [n for n in nodes_vo_all if n not in self.nodes_original]
        graph_vo = self.graph.subgraph(nodes_vo)

        def get_value_of(**data):
            data = data.copy()
            # This loops through the functions in the correct order determined above so
            # we end up calculating the value of interest, which is returned
            for n in nx.topological_sort(graph_vo):
                data.update(
                    {
                        n: self.funcs[n](
                            *[
                                data[v]
                                for v in signature(self.funcs[n]).parameters.keys()
                            ]
                        )
                    }
                )
            return data[var_of]

        # Generate docstring
        get_value_of.__doc__ = (
            f"Calculate ``{var_of}``."
            + "\n\nParameters\n----------"
            + "\nvalues : dict"
            + "\n    Key-value pairs for the following parameters:"
        )
        for p in nodes_vo_all:
            if p in self.nodes_original:
                get_value_of.__doc__ += f"\n        {p}"
        get_value_of.__doc__ += "\n\nReturns\n-------"
        get_value_of.__doc__ += f"\n{var_of}"
        return get_value_of

    def _get_func_of_from_wrt(self, get_value_of, var_wrt):
        """Reorganise a function created with ``_get_func_of`` so that one of its
        kwargs is instead a positional arg (and which can thus be gradded).

        Parameters
        ----------
        get_value_of : func
            Function created with ``_get_func_of``.
        var_wrt : str
            Name of the value to use as a positional arg instead.

        Returns
        -------
        A function with the signature
            value_of = get_of_from_wrt(value_wrt, **other_values_original)
        """

        def get_value_of_from_wrt(value_wrt, **other_values_original):
            other_values_original = other_values_original.copy()
            other_values_original.update({var_wrt: value_wrt})
            return get_value_of(**other_values_original)

        return get_value_of_from_wrt

    def get_grad_func(self, var_of, var_wrt):
        get_value_of = self._get_func_of(var_of)
        get_value_of_from_wrt = self._get_func_of_from_wrt(get_value_of, var_wrt)
        return meta.egrad(get_value_of_from_wrt)

    def get_grad(self, var_of, var_wrt):
        """Compute the derivative of `var_of` with respect to `var_wrt` and store
        it in `sys.grads[var_of][var_wrt]`.  If there is already a value there,
        then that value is returned instead of recalculating.

        Parameters
        ----------
        var_of : str
            The name of the variable to get the derivative of.
        var_wrt : str
            The name of the variable to get the derivative with respect to.  This
            must be one of the fixed values provided when creating the `CO2System`,
            i.e., listed in its `nodes_original` attribute.
        """
        assert var_wrt in self.nodes_original, (
            "`var_wrt` must be one of `sys.nodes_original!`"
        )
        try:  # see if we've already calculated this value
            d_of__d_wrt = self.grads[var_of][var_wrt]
        except KeyError:  # only do the calculations if there isn't already a value
            # We need to know the shape of the variable that we want the grad of,
            # the easy way to get this is just to solve for it (if that hasn't
            # already been done)
            if var_of not in self.data:
                self.solve(var_of)
            # Next, we extract the originally set values, which are fixed during the
            # differentiation
            values_original = self.get_values_original()
            other_values_original = values_original.copy()
            # We have to make sure the value we are differentiating with respect
            # to has the same shape as the value we want the differential of
            value_wrt = other_values_original.pop(var_wrt) * np.ones_like(
                self.data[var_of]
            )
            # Here we compute the gradient
            grad_func = self.get_grad_func(var_of, var_wrt)
            d_of__d_wrt = grad_func(value_wrt, **other_values_original)
            # Put the final value into self.grads, first creating a new sub-dict
            # if necessary
            if var_of not in self.grads:
                self.grads[var_of] = {}
            self.grads[var_of][var_wrt] = d_of__d_wrt
        return d_of__d_wrt

    def get_grads(self, vars_of, vars_wrt):
        """Compute the derivatives of `vars_of` with respect to `vars_wrt` and
        store them in `sys.grads[var_of][var_wrt]`.  If there are already values
        there, then those values are returned instead of recalculating.

        Parameters
        ----------
        vars_of : list
            The names of the variables to get the derivatives of.
        vars_wrt : list
            The names of the variables to get the derivatives with respect to.
            These must all be one of the fixed values provided when creating the
            `CO2System`, i.e., listed in its `nodes_original` attribute.
        """
        if isinstance(vars_of, str):
            vars_of = [vars_of]
        if isinstance(vars_wrt, str):
            vars_wrt = [vars_wrt]
        for var_of, var_wrt in itertools.product(vars_of, vars_wrt):
            self.get_grad(var_of, var_wrt)

    def get_values_original(self):
        return {k: self.data[k] for k in self.nodes_original}

    def propagate(self, uncertainty_in, uncertainty_from):
        """Propagate independent uncertainties through the calculations.  Covariances
        are not accounted for.

        New entries are added in the `uncertainty` attribute, for example:

            co2s = pyco2.sys(dic=2100, alkalinity=2300)
            co2s.propagate("pH", {"dic": 2, "alkalinity": 2})
            co2s.uncertainty["pH"]["total"]  # total uncertainty in pH
            co2s.uncertainty["pH"]["dic"]  # component of ^ due to DIC uncertainty

        Parameters
        ----------
        uncertainty_in : list
            The parameters to calculate the uncertainty in.
        uncertainty_from : dict
            The parameters to propagate the uncertainty from (keys) and their
            uncertainties (values).
        """
        self.solve(uncertainty_in)
        if isinstance(uncertainty_in, str):
            uncertainty_in = [uncertainty_in]
        for var_in in uncertainty_in:
            # This should always be reset to zero and all values wiped, even if
            # it already exists (so you don't end up with old uncertainty_from
            # components from a previous calculation which are no longer part of
            # the total)
            self.uncertainty[var_in] = {"total": np.zeros_like(self.data[var_in])}
            u_total = self.uncertainty[var_in]["total"]
            for var_from, u_from in uncertainty_from.items():
                is_pk = var_from.startswith("pk_")
                if is_pk:
                    # If the uncertainty is given in terms of a pK value, we do
                    # the calculations as if it were a K value, and convert at
                    # the end
                    var_from = var_from[1:]
                is_fractional = var_from.endswith("__f")
                if is_fractional:
                    # If the uncertainty is fractional, multiply through by this
                    var_from = var_from[:-3]
                    u_from = self.data[var_from] * u_from
                if var_from in self.nodes_original:
                    self.get_grad(var_in, var_from)
                    u_part = np.abs(self.grads[var_in][var_from] * u_from)
                else:
                    # If the uncertainty is from some internally calculated value,
                    # then we need to make a second CO2System where that value
                    # is one of the known inputs, and get the grad from that
                    self.solve(var_from)
                    data = self.get_values_original()
                    data.update({var_from: self.data[var_from]})
                    sys = CO2System(**data, **self.opts)
                    sys.get_grad(var_in, var_from)
                    u_part = np.abs(sys.grads[var_in][var_from] * u_from)
                # Add the p back and convert value, if necessary
                if is_pk:
                    u_part = u_part * np.log(10) * np.abs(sys.data[var_from])
                    var_from = "p" + var_from
                if is_fractional:
                    var_from += "__f"
                self.uncertainty[var_in][var_from] = u_part
                u_total = u_total + u_part**2
            self.uncertainty[var_in]["total"] = np.sqrt(u_total)

    def get_graph_to_plot(
        self,
        show_tsp=True,
        show_unknown=True,
        keep_unknown=None,
        exclude_nodes=None,
        show_isolated=True,
        skip_nodes=None,
    ):
        graph_to_plot = self.graph.copy()
        # Remove nodes as requested by user
        if not show_tsp:
            graph_to_plot.remove_nodes_from(["pressure", "salinity", "temperature"])
        if not show_unknown:
            if keep_unknown is None:
                keep_unknown = []
            elif isinstance(keep_unknown, str):
                keep_unknown = [keep_unknown]
            node_states = nx.get_node_attributes(graph_to_plot, "state", default=0)
            to_remove = [
                n for n, s in node_states.items() if s == 0 and n not in keep_unknown
            ]
            graph_to_plot.remove_nodes_from(to_remove)
        # Connect nodes that are missing due to store_steps=1 mode
        _graph_to_plot = graph_to_plot.copy()
        for n, properties in _graph_to_plot.nodes.items():
            if (
                "state" in properties
                and properties["state"] in [2, 3]
                and len(_graph_to_plot.pred[n]) == 0
                and len(nx.ancestors(self.graph, n)) > 0
            ):
                for a in nx.ancestors(self.graph, n):
                    if a in _graph_to_plot.nodes:
                        graph_to_plot.add_edge(a, n, state=2)
        if exclude_nodes:
            # Excluding nodes just makes them disappear from the graph without
            # caring about what they were connected to
            if isinstance(exclude_nodes, str):
                exclude_nodes = [exclude_nodes]
            graph_to_plot.remove_nodes_from(exclude_nodes)
        if not show_isolated:
            graph_to_plot.remove_nodes_from(
                [n for n, d in dict(graph_to_plot.degree).items() if d == 0]
            )
        if skip_nodes:
            # Skipping nodes removes them but then shows their predecessors as
            # being directly connected to their children
            edge_states = nx.get_edge_attributes(graph_to_plot, "state", default=0)
            if isinstance(skip_nodes, str):
                skip_nodes = [skip_nodes]
            for n in skip_nodes:
                for p, s in itertools.product(
                    graph_to_plot.predecessors(n), graph_to_plot.successors(n)
                ):
                    graph_to_plot.add_edge(p, s)
                    if edge_states[(p, n)] + edge_states[(n, s)] == 4:
                        new_state = {(p, s): 2}
                    else:
                        new_state = {(p, s): 0}
                    nx.set_edge_attributes(graph_to_plot, new_state, name="state")
                    edge_states.update(new_state)
                graph_to_plot.remove_node(n)
        return graph_to_plot

    def get_graph_pos(
        self,
        graph_to_plot=None,
        prog_graphviz=None,
        root_graphviz=None,
        args_graphviz="",
        nx_layout=nx.spring_layout,
        nx_args=None,
        nx_kwargs=None,
    ):
        if graph_to_plot is None:
            graph_to_plot = self.graph
        if prog_graphviz is not None:
            pos = nx.nx_agraph.graphviz_layout(
                graph_to_plot,
                prog=prog_graphviz,
                root=root_graphviz,
                args=args_graphviz,
            )
        else:
            if nx_args is None:
                nx_args = ()
            if nx_kwargs is None:
                nx_kwargs = {}
            pos = nx_layout(graph_to_plot, *nx_args, **nx_kwargs)
        return pos

    def plot_graph(
        self,
        ax=None,
        exclude_nodes=None,
        show_tsp=True,
        show_unknown=True,
        keep_unknown=None,
        show_isolated=True,
        skip_nodes=None,
        prog_graphviz=None,
        root_graphviz=None,
        args_graphviz="",
        nx_layout=nx.spring_layout,
        nx_args=None,
        nx_kwargs=None,
        node_kwargs=None,
        edge_kwargs=None,
        label_kwargs=None,
        mode="state",
    ):
        """Draw a graph showing the relationships between the different parameters.

        Parameters
        ----------
        ax : matplotlib axes, optional
            The axes on which to plot.  If `None`, a new figure and axes are created.
        exclude_nodes : list of str, optional
            List of nodes to exclude from the plot, by default `None`.  Nodes in
            this list are not shown, nor are connections to them or through them.
        prog_graphviz : str, optional
            Name of Graphviz layout program, by default "neato".
        show_tsp : bool, optional
            Whether to show temperature, salinity and pressure nodes, by default
            `True`.
        show_unknown : bool, optional
            Whether to show nodes for parameters that have not (yet) been calculated,
            by default `True`.
        show_isolated : bool, optional
            Whether to show nodes for parameters that are not connected to the
            graph, by default `True`.
        skip_nodes : bool, optional
            List of nodes to skip from the plot, by default `None`.  Nodes in this
            list are not shown, but the connections between their predecessors
            and children are still drawn.

        Returns
        -------
        matplotlib axes
            The axes on which the graph is plotted.
        """
        from matplotlib import pyplot as plt

        # NODE STATES
        # -----------
        # no state (grey) = unknwown
        # 1 (grass) = provided by user (or default) i.e. known but not calculated
        # 2 (azure) = calculated en route to a user-requested parameter
        # 3 (tangerine) = calculated after direct user request
        #
        # EDGE STATES
        # -----------
        # no state (grey) = calculation not performed
        # 2 = (azure) calculation performed
        #
        if ax is None:
            ax = plt.subplots(dpi=300, figsize=(8, 7))[1]
        if mode == "valid" and not self.checked_valid:
            self.check_valid()
        graph_to_plot = self.get_graph_to_plot(
            exclude_nodes=exclude_nodes,
            show_tsp=show_tsp,
            show_unknown=show_unknown,
            keep_unknown=keep_unknown,
            show_isolated=show_isolated,
            skip_nodes=skip_nodes,
        )
        pos = self.get_graph_pos(
            graph_to_plot=graph_to_plot,
            prog_graphviz=prog_graphviz,
            root_graphviz=root_graphviz,
            args_graphviz=args_graphviz,
            nx_layout=nx_layout,
            nx_args=nx_args,
            nx_kwargs=nx_kwargs,
        )
        if mode == "state":
            node_states = nx.get_node_attributes(graph_to_plot, "state", default=0)
            edge_states = nx.get_edge_attributes(graph_to_plot, "state", default=0)
            node_colour = [
                self.state_colours[node_states[n]] for n in nx.nodes(graph_to_plot)
            ]
            edge_colour = [
                self.state_colours[edge_states[e]] for e in nx.edges(graph_to_plot)
            ]
        elif mode == "valid":
            node_valid = nx.get_node_attributes(graph_to_plot, "valid", default=0)
            edge_valid = nx.get_edge_attributes(graph_to_plot, "valid", default=0)
            node_valid_p = nx.get_node_attributes(graph_to_plot, "valid_p", default=0)
            node_colour = [
                self.valid_colours[node_valid[n]] for n in nx.nodes(graph_to_plot)
            ]
            edge_colour = [
                self.valid_colours[edge_valid[e]] for e in nx.edges(graph_to_plot)
            ]
            node_edgecolors = [
                self.valid_colours[node_valid_p[n]] for n in nx.nodes(graph_to_plot)
            ]
            node_linewidths = [[0, 2][node_valid_p[n]] for n in nx.nodes(graph_to_plot)]
        else:
            warnings.warn(
                f'mode "{mode}" not recognised, options are "state", "valid".'
            )
            node_colour = "xkcd:grey"
            edge_colour = "xkcd:grey"
        node_labels = {k: k for k in graph_to_plot.nodes}
        for k, v in set_node_labels.items():
            if k in node_labels:
                node_labels[k] = v
        if node_kwargs is None:
            node_kwargs = {}
        if edge_kwargs is None:
            edge_kwargs = {}
        if label_kwargs is None:
            label_kwargs = {}
        if mode == "valid":
            node_kwargs["edgecolors"] = node_edgecolors
            node_kwargs["linewidths"] = node_linewidths
        nx.draw_networkx_nodes(
            graph_to_plot,
            ax=ax,
            node_color=node_colour,
            pos=pos,
            **node_kwargs,
        )
        nx.draw_networkx_edges(
            graph_to_plot,
            ax=ax,
            edge_color=edge_colour,
            pos=pos,
            **edge_kwargs,
        )
        nx.draw_networkx_labels(
            graph_to_plot,
            ax=ax,
            labels=node_labels,
            pos=pos,
            **label_kwargs,
        )
        return ax

    def keys_all(self):
        """Return a tuple of all possible results keys, including those that have
        not yet been solved for.
        """
        return tuple(self.graph.nodes)

    def check_valid(self, ignore=None):
        """Check which parameters are valid."""
        if ignore is None:
            ignore = []
        if isinstance(ignore, str):
            ignore = [ignore]
        for n in nx.topological_sort(self.graph):
            # First, assign validity for functions that do have valid ranges
            # (shown by node fill colour on the graph plot)
            if n in self.funcs and n not in ignore and hasattr(self.funcs[n], "valid"):
                n_valid = []
                for p, p_range in self.funcs[n].valid.items():
                    # If all predecessor parameters fall within valid ranges, it's valid
                    if np.all(
                        (self.data[p] >= p_range[0]) & (self.data[p] <= p_range[1])
                    ):
                        n_valid.append(1)
                        nx.set_edge_attributes(
                            self.graph,
                            {(p, n): 1},
                            name="valid",
                        )
                    # If any predecessor parameter is outside any range, it's invalid
                    else:
                        n_valid.append(-1)
                        nx.set_edge_attributes(
                            self.graph,
                            {(p, n): -1},
                            name="valid",
                        )
                nx.set_node_attributes(
                    self.graph,
                    {n: min(n_valid)},
                    name="valid",
                )
            # Next, assign inherited validity
            # (shown by node edge colour on the graph plot)
            n_valid_p = []
            for p in self.graph.predecessors(n):
                p_attrs = self.graph.nodes[p]
                for v in ["valid", "valid_p"]:
                    if v in p_attrs:
                        n_valid_p.append(p_attrs[v])
                        if p_attrs[v] == -1:
                            nx.set_edge_attributes(
                                self.graph,
                                {(p, n): -1},
                                name="valid",
                            )
            if -1 in n_valid_p:
                nx.set_node_attributes(
                    self.graph,
                    {n: -1},
                    name="valid_p",
                )
        self.checked_valid = True


def sys(data=None, **kwargs):
    """Create a `CO2System`."""
    # Check for double precision
    if np.array(1.0).dtype is np.dtype("float32"):
        warnings.warn(
            "JAX does not appear to be using double precision - "
            + "set the environment variable `JAX_ENABLE_X64=True`."
        )
    # Merge data with kwargs
    pd_index = None
    xr_dims = None
    xr_shape = None
    data_is_dict = False
    if data is not None:
        for k in kwargs:
            if k in data:
                warnings.warn(
                    f'"{k}" found in both `data` and `kwargs` - the value in '
                    + "`data` will be used."
                )
        data_is_dict = isinstance(data, dict)
        # Any kwargs other than `data` provided as strings will be interpreted
        # as being the keys for the corresponding values
        renamer = {}
        for k, v in kwargs.items():
            if isinstance(v, str):
                if v in renamer:
                    raise Exception(
                        f'"{v}" cannot be used for {k}'
                        + f" because it is already being used for {renamer[v]}!"
                    )
                else:
                    renamer[v] = k
        if data_is_dict:
            for k, v in data.items():
                if k in renamer:
                    kwargs[renamer[k]] = v
                else:
                    kwargs[k] = v
        else:
            data_is_pandas = False
            try:
                import pandas as pd

                data_is_pandas = isinstance(data, pd.DataFrame)
                if data_is_pandas:
                    pd_index = data.index.copy()
                    for c in data.columns:
                        if c in renamer:
                            kwargs[renamer[c]] = data[c].to_numpy()
                        else:
                            kwargs[c] = data[c].to_numpy()
            except ImportError:
                warnings.warn("pandas could not be imported - ignoring `data`.")
            data_is_xarray = False
            if not data_is_pandas:
                try:
                    import xarray as xr

                    data_is_xarray = isinstance(data, xr.Dataset)
                    if data_is_xarray:
                        xr_dims = list(data.sizes.keys())
                        xr_shape = list(data.sizes.values())
                        for k, v in data.items():
                            ndims = []
                            for d in xr_dims:
                                if d in v.sizes:
                                    ndims.append(v.sizes[d])
                                else:
                                    ndims.append(1)
                            if k in renamer:
                                kwargs[renamer[k]] = np.reshape(v.data, ndims)
                            else:
                                kwargs[k] = np.reshape(v.data, ndims)
                except ImportError:
                    warnings.warn("xarray could not be imported - ignoring `data`.")
                if not data_is_xarray:
                    warnings.warn("Type of `data` not recognised - it will be ignored.")
    # Parse kwargs
    for k in kwargs:
        # Convert lists to numpy arrays
        if isinstance(kwargs[k], list):
            kwargs[k] = np.array(kwargs[k])
        # If opts are scalar, only take first value
        if k.startswith("opt_"):
            if np.isscalar(kwargs[k]):
                if isinstance(kwargs[k], ArrayImpl):
                    kwargs[k] = kwargs[k].item()
            else:
                kwargs[k] = np.ravel(np.array(kwargs[k]))[0].item()
                warnings.warn(
                    f"`{k}` is not scalar, so only the first value will be used."
                )
            if isinstance(kwargs[k], float):
                kwargs[k] = int(kwargs[k])
        # Convert ints to floats
        else:
            if isinstance(kwargs[k], int):
                kwargs[k] = float(kwargs[k])
            elif hasattr(kwargs[k], "dtype"):
                if kwargs[k].dtype == int:
                    kwargs[k] = kwargs[k].astype(float)
    return CO2System(pd_index=pd_index, xr_dims=xr_dims, xr_shape=xr_shape, **kwargs)
