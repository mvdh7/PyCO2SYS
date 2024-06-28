# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
import itertools
import networkx as nx
from jax import numpy as np
from matplotlib import pyplot as plt
from .. import (
    bio,
    buffers,
    constants,
    convert,
    equilibria,
    gas,
    salts,
    solubility,
    solve,
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
    "free_to_sws_1atm": lambda total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm: convert.pH_free_to_sws(
        total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm
    ),
    "nbs_to_sws": convert.pH_nbs_to_sws,  # because fH doesn't get pressure-corrected
    "tot_to_sws_1atm": lambda total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm: convert.pH_tot_to_sws(
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
    # Chemical speciation
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
    # Gasses
    "vp_factor": gas.vpfactor,
}

# Define functions for calculations that depend on icase:
get_funcs_core = {}
for i in [0, 3, 4, 5, 6, 8, 9, 10, 11]:
    get_funcs_core[i] = {}
# alkalinity and DIC
get_funcs_core[102] = {
    "pH": solve.get.inorganic.pH_from_alkalinity_dic,
    "fCO2": solve.get.inorganic.fCO2_from_dic_pH,
    "CO3": solve.get.inorganic.CO3_from_dic_pH,
    "HCO3": solve.get.inorganic.HCO3_from_dic_pH,
}
# alkalinity and pH
get_funcs_core[103] = {
    "dic": solve.get.inorganic.dic_from_alkalinity_pH_speciated,
    "fCO2": solve.get.inorganic.fCO2_from_dic_pH,
    "CO3": solve.get.inorganic.CO3_from_dic_pH,
    "HCO3": solve.get.inorganic.HCO3_from_dic_pH,
}
# alkalinity and pCO2, fCO2, CO2, xCO2
for i in [104, 105, 108, 109]:
    get_funcs_core[i] = {
        "pH": solve.get.inorganic.pH_from_alkalinity_fCO2,
        "dic": solve.get.inorganic.dic_from_pH_fCO2,
        "HCO3": solve.get.inorganic.HCO3_from_pH_fCO2,
        "CO3": solve.get.inorganic.CO3_from_dic_pH,
    }
# alkalinity and CO3, omega
for i in [106, 110, 111]:
    get_funcs_core[i] = {
        "pH": solve.get.inorganic.pH_from_alkalinity_CO3,
        "dic": solve.get.inorganic.dic_from_pH_CO3,
        "HCO3": solve.get.inorganic.HCO3_from_pH_CO3,
        "fCO2": solve.get.inorganic.fCO2_from_pH_CO3,
    }
# alkalinity and HCO3
get_funcs_core[107] = {
    "pH": solve.get.inorganic.pH_from_alkalinity_HCO3,
    "dic": solve.get.inorganic.dic_from_pH_HCO3,
    "CO3": solve.get.inorganic.CO3_from_pH_HCO3,
    "fCO2": solve.get.inorganic.fCO2_from_pH_HCO3,
}
# DIC and pH
get_funcs_core[203] = {
    "fCO2": solve.get.inorganic.fCO2_from_dic_pH,
    "CO3": solve.get.inorganic.CO3_from_dic_pH,
    "HCO3": solve.get.inorganic.HCO3_from_dic_pH,
    "alkalinity": solve.speciate.sum_alkalinity,
}
# DIC and pCO2, fCO2, CO2, xCO2
for i in [204, 205, 208, 209]:
    get_funcs_core[i] = {
        "pH": solve.get.inorganic.pH_from_dic_fCO2,
        "HCO3": solve.get.inorganic.HCO3_from_pH_fCO2,
        "CO3": solve.get.inorganic.CO3_from_dic_pH,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# DIC and CO3, omega
for i in [206, 210, 211]:
    get_funcs_core[i] = {
        "pH": solve.get.inorganic.pH_from_dic_CO3,
        "HCO3": solve.get.inorganic.HCO3_from_pH_CO3,
        "fCO2": solve.get.inorganic.fCO2_from_pH_CO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# DIC and HCO3
get_funcs_core[207] = {
    # pH is taken care of by opt_HCO3_root
    "CO3": solve.get.inorganic.CO3_from_pH_HCO3,
    "fCO2": solve.get.inorganic.fCO2_from_pH_HCO3,
    "alkalinity": solve.speciate.sum_alkalinity,
}
# pH and pCO2, fCO2, CO2, xCO2
for i in [304, 305, 308, 309]:
    get_funcs_core[i] = {
        "dic": solve.get.inorganic.dic_from_pH_fCO2,
        "HCO3": solve.get.inorganic.HCO3_from_pH_fCO2,
        "CO3": solve.get.inorganic.CO3_from_dic_pH,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# pH and CO3, omega
for i in [306, 310, 311]:
    get_funcs_core[i] = {
        "dic": solve.get.inorganic.dic_from_pH_CO3,
        "HCO3": solve.get.inorganic.HCO3_from_pH_CO3,
        "fCO2": solve.get.inorganic.fCO2_from_pH_CO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# pH and HCO3
get_funcs_core[307] = {
    "dic": solve.get.inorganic.dic_from_pH_HCO3,
    "CO3": solve.get.inorganic.CO3_from_pH_HCO3,
    "fCO2": solve.get.inorganic.fCO2_from_pH_HCO3,
    "alkalinity": solve.speciate.sum_alkalinity,
}
# CO3, omega and pCO2, fCO2, CO2, xCO2
for i in [406, 506, 608, 609, 410, 510, 810, 910, 411, 511, 811, 911]:
    get_funcs_core[i] = {
        "pH": solve.get.inorganic.pH_from_fCO2_CO3,
        "dic": solve.get.inorganic.dic_from_pH_CO3,
        "HCO3": solve.get.inorganic.HCO3_from_pH_CO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# HCO3 and pCO2, fCO2, CO2, xCO2
for i in [407, 507, 708, 709]:
    get_funcs_core[i] = {
        "pH": solve.get.inorganic.pH_from_fCO2_HCO3,
        "dic": solve.get.inorganic.dic_from_pH_HCO3,
        "CO3": solve.get.inorganic.CO3_from_pH_HCO3,
        "alkalinity": solve.speciate.sum_alkalinity,
    }
# CO3, omega and HCO3
for i in [607, 710, 711]:
    get_funcs_core[i] = {
        "pH": solve.get.inorganic.pH_from_CO3_HCO3,
        "fCO2": solve.get.inorganic.fCO2_from_CO3_HCO3,
        "dic": solve.get.inorganic.dic_from_pH_CO3,
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
            }
        )
    elif k in [10, 110, 210, 310, 410, 510, 710, 810, 910]:
        fc.update(
            {
                "CO3": solubility.CO3_from_OC,
                "saturation_aragonite": solubility.OA_from_CO3,
            }
        )
    elif k in [11, 111, 211, 311, 411, 511, 711, 811, 911]:
        fc.update(
            {
                "CO3": solubility.CO3_from_OA,
                "saturation_calcite": solubility.OC_from_CO3,
            }
        )

# TODO Add parameters that require a solved carbonate system
for k, fc in get_funcs_core.items():
    if k > 100:
        fc.update(
            {
                "substrate_inhibitor_ratio": bio.substrate_inhibitor_ratio,
                "d_lnOmega__d_CO3": buffers.d_lnOmega__d_CO3,
                "d_CO3__d_pH__alkalinity": buffers.d_CO3__d_pH__alkalinity,
                "d_CO3__d_pH__dic": buffers.d_CO3__d_pH__dic,
                "d_dic__d_pH__alkalinity": buffers.d_dic__d_pH__alkalinity,
                "d_alkalinity__d_pH__dic": buffers.d_alkalinity__d_pH__dic,
                "d_lnCO2__d_pH__alkalinity": buffers.d_lnCO2__d_pH__alkalinity,
                "d_lnCO2__d_pH__dic": buffers.d_lnCO2__d_pH__dic,
                "gamma_dic": buffers.gamma_dic,
                "gamma_alkalinity": buffers.gamma_alkalinity,
                "beta_dic": buffers.beta_dic,
                "beta_alkalinity": buffers.beta_alkalinity,
                "omega_dic": buffers.omega_dic,
                "omega_alkalinity": buffers.omega_alkalinity,
                "d_alkalinity__d_pH__fCO2": buffers.d_alkalinity__d_pH__fCO2,
                "d_dic__d_pH__fCO2": buffers.d_dic__d_pH__fCO2,
                "Q_isocap": buffers.Q_isocap,
                "Q_isocap_approx": buffers.Q_isocap_approx,
                "psi": buffers.psi,
                "d_fCO2__d_pH__alkalinity": buffers.d_fCO2__d_pH__alkalinity,
                "revelle_factor": buffers.revelle_factor,
            }
        )

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
    2: dict(factor_k_H2CO3=equilibria.pcx.factor_k_H2CO3_fw),
    3: dict(factor_k_H2CO3=equilibria.pcx.factor_k_H2CO3_GEOSECS),
}
get_funcs_opts["opt_factor_k_HCO3"] = {
    1: dict(factor_k_HCO3=equilibria.pcx.factor_k_HCO3),
    2: dict(factor_k_HCO3=equilibria.pcx.factor_k_HCO3_fw),
    3: dict(factor_k_HCO3=equilibria.pcx.factor_k_HCO3_GEOSECS),
}
get_funcs_opts["opt_factor_k_H2O"] = {
    1: dict(factor_k_H2O=equilibria.pcx.factor_k_H2O),
    2: dict(factor_k_H2O=equilibria.pcx.factor_k_H2O_fw),
}
get_funcs_opts["opt_fH"] = {
    1: dict(fH=convert.fH_TWB82),
    2: dict(fH=convert.fH_PTBO87),
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
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_WMW14,
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
        k_BOH3_sws_1atm=lambda k_BOH3_total_1atm, nbs_to_sws: (
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
    1: dict(k_NH3_sws_1atm=equilibria.p1atm.k_NH3_sws_YM95),
    2: dict(
        k_NH3_total_1atm=equilibria.p1atm.k_NH3_total_CW95,
        k_NH3_sws_1atm=lambda k_NH3_total_1atm, tot_to_sws_1atm: (
            k_NH3_total_1atm * tot_to_sws_1atm
        ),
    ),
}
get_funcs_opts["opt_k_Si"] = {
    1: dict(k_Si_sws_1atm=equilibria.p1atm.k_Si_sws_YM95),
    2: dict(
        k_Si_nbs_1atm=equilibria.p1atm.k_Si_nbs_SMB64,
        k_Si_sws_1atm=lambda k_Si_nbs_1atm, nbs_to_sws: (k_Si_nbs_1atm * nbs_to_sws),
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
for o, funcs in get_funcs_opts["opt_pH_scale"].items():
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
    1: dict(pH=solve.get.inorganic.pH_from_dic_HCO3_lo),
    2: dict(pH=solve.get.inorganic.pH_from_dic_HCO3_hi),  # for typical seawater
}
get_funcs_opts["opt_k_calcite"] = {
    1: dict(k_calcite=solubility.k_calcite_M83),
    2: dict(k_calcite=solubility.k_calcite_I75),  # for GEOSECS
}
get_funcs_opts["opt_k_aragonite"] = {
    1: dict(k_aragonite=solubility.k_aragonite_M83),
    2: dict(k_aragonite=solubility.k_aragonite_GEOSECS),  # for GEOSECS
}

# Automatically set up graph for calculations that depend neither on icase nor opts
# based on the function names and signatures in get_funcs
graph_fixed = nx.DiGraph()
for k, func in get_funcs.items():
    fcode = func.__code__
    func_args = fcode.co_varnames[: fcode.co_argcount]
    for f in func_args:
        graph_fixed.add_edge(f, k)

# Automatically set up graph for each icase based on the function names and signatures
# in get_funcs_core
graph_core = {}
for icase, funcs in get_funcs_core.items():
    graph_core[icase] = nx.DiGraph()
    for t, func in get_funcs_core[icase].items():
        fcode = func.__code__
        func_args = fcode.co_varnames[: fcode.co_argcount]
        for f in func_args:
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
                    fcode = func.__code__
                    func_args = fcode.co_varnames[: fcode.co_argcount]
                    for f in func_args:
                        graph_opts[o][opt].add_edge(f, k)
    return graph_opts


parameters_core = [
    "alkalinity",
    "dic",
    "pH",
    "pCO2",
    "fCO2",
    "CO3",
    "HCO3",
    "CO2",
    "xCO2",
    "saturation_calcite",
    "saturation_aragonite",
]

default_values = {
    "temperature": 25.0,  # °C
    "total_ammonia": 0.0,  # µmol/kg-sw
    "total_phosphate": 0.0,  # µmol/kg-sw
    "total_silicate": 0.0,  # µmol/kg-sw
    "total_sulfide": 0.0,  # µmol/kg-sw
    "salinity": 35.0,
    "pressure": 0.0,  # dbar
    "pressure_atmosphere": 1.0,  # atm
}

default_opts = {
    "opt_gas_constant": 3,
    "opt_factor_k_BOH3": 1,
    "opt_factor_k_H2CO3": 1,
    "opt_factor_k_HCO3": 1,
    "opt_factor_k_H2O": 1,
    "opt_fH": 1,
    "opt_k_carbonic": 1,
    "opt_k_phosphate": 1,
    "opt_k_BOH3": 1,
    "opt_k_H2O": 1,
    "opt_k_HF": 1,
    "opt_k_HSO4": 1,
    "opt_k_NH3": 1,
    "opt_k_Si": 1,
    "opt_pH_scale": 1,
    "opt_total_borate": 1,
    "opt_Ca": 1,
    "opt_fugacity_factor": 1,
    "opt_HCO3_root": 2,
    "opt_k_calcite": 1,
    "opt_k_aragonite": 1,
}

# Define labels for parameter plotting
thinspace = " "
f = "$ƒ$" + thinspace
set_node_labels = {
    "dic": r"$T_\mathrm{C}$",
    "k_CO2": "$K_0′$",
    "k_CO2_1atm": "$K_0′^0$",
    "k_H2CO3": "$K_1^*$",
    "k_HCO3": "$K_2^*$",
    "k_H2CO3_sws": "$K_1^s$",
    "k_HCO3_sws": "$K_2^s$",
    "k_H2CO3_sws_1atm": r"$K_1^\mathrm{S0}$",
    "k_HCO3_sws_1atm": r"$K_2^\mathrm{S0}$",
    "k_H2CO3_total_1atm": r"$K_1^\mathrm{T0}$",
    "k_HCO3_total_1atm": r"$K_2^\mathrm{T0}$",
    "k_HF_free": r"$K_\mathrm{HF}^\mathrm{F}$",
    "k_HSO4_free": r"$K_\mathrm{HSO_4}^\mathrm{F}$",
    "k_HF_free_1atm": r"$K_\mathrm{HF}^\mathrm{F0}$",
    "k_HSO4_free_1atm": r"$K_\mathrm{HSO_4}^\mathrm{F0}$",
    "pressure_atmosphere": r"$p_\mathrm{atm}$",
    "ionic_strength": "$I$",
    "temperature": "$t$",
    "salinity": "$S$",
    "pressure": "$p$",
    "gas_constant": "$R$",
    "CO3": "[CO$_3^{2–}$]",
    "HCO3": "[HCO$_3^–$]",
    "total_sulfate": r"$T_\mathrm{SO_4}$",
    "total_fluoride": r"$T_\mathrm{F}$",
    "total_ammonia": r"$T_\mathrm{NH_3}$",
    "total_phosphate": r"$T_\mathrm{P}$",
    "total_sulfide": r"$T_\mathrm{H_2S}$",
    "total_silicate": r"$T_\mathrm{Si}$",
    "total_borate": r"$T_\mathrm{B}$",
    "Ca": r"$[\mathrm{Ca}^{2+}]$",
    "tot_to_sws_1atm": r"$_\mathrm{T}^\mathrm{S}Y^0$",
    "sws_to_opt": r"$_\mathrm{S}^*Y$",
    "opt_to_free": r"$_*^\mathrm{F}Y$",
    "fCO2": f + "CO$_2$",
    "factor_k_CO2": "$P_0$",
    "factor_k_H2CO3": "$P_1$",
    "factor_k_HCO3": "$P_2$",
    "factor_k_HSO4": r"$P_\mathrm{SO_4}$",
    "factor_k_HF": r"$P_\mathrm{HF}$",
    "factor_k_BOH3": r"$P_\mathrm{B}$",
    "factor_k_H2O": r"$P_w$",
    "factor_k_Si": r"$P_\mathrm{Si}$",
    "factor_k_NH3": r"$P_\mathrm{NH_3}$",
    "factor_k_H2S": r"$P_\mathrm{H_2S}$",
    "CO2": r"$[\mathrm{CO}_2(\mathrm{aq})]$",
    "H3PO4": r"$[\mathrm{H}_3\mathrm{PO}_4]$",
    "H2PO4": r"$[\mathrm{H}_2\mathrm{PO}_4^–]$",
    "HPO4": r"$[\mathrm{HPO}_4^{2–}]$",
    "PO4": r"$[\mathrm{PO}_4^{3–}]$",
    "k_H3PO4": r"$K_\mathrm{P1}^*$",
    "k_H2PO4": r"$K_\mathrm{P2}^*$",
    "k_HPO4": r"$K_\mathrm{P3}^*$",
    "k_BOH3": r"$K_\mathrm{B}^*$",
    "k_Si": r"$K_\mathrm{Si}^*$",
    "k_NH3": r"$K_\mathrm{NH_3}^*$",
    "k_H2S": r"$K_\mathrm{H_2S}^*$",
    "k_H2O": "$K_w^*$",
    "k_H3PO4_sws": r"$K_\mathrm{P1}^\mathrm{S}$",
    "k_H2PO4_sws": r"$K_\mathrm{P2}^\mathrm{S}$",
    "k_HPO4_sws": r"$K_\mathrm{P3}^\mathrm{S}$",
    "k_BOH3_sws": r"$K_\mathrm{B}^\mathrm{S}$",
    "k_Si_sws": r"$K_\mathrm{Si}^\mathrm{S}$",
    "k_NH3_sws": r"$K_\mathrm{NH_3}^\mathrm{S}$",
    "k_H2S_sws": r"$K_\mathrm{H_2S}^\mathrm{S}$",
    "k_H2O_sws": r"$K_w^\mathrm{S}$",
    "k_H3PO4_sws_1atm": r"$K_\mathrm{P1}^\mathrm{S0}$",
    "k_H2PO4_sws_1atm": r"$K_\mathrm{P2}^\mathrm{S0}$",
    "k_HPO4_sws_1atm": r"$K_\mathrm{P3}^\mathrm{S0}$",
    "k_BOH3_sws_1atm": r"$K_\mathrm{B}^\mathrm{S0}$",
    "k_BOH3_total_1atm": r"$K_\mathrm{B}^\mathrm{T0}$",
    "k_Si_sws_1atm": r"$K_\mathrm{Si}^\mathrm{S0}$",
    "k_NH3_sws_1atm": r"$K_\mathrm{NH_3}^\mathrm{S0}$",
    "k_H2S_sws_1atm": r"$K_\mathrm{H_2S}^\mathrm{S0}$",
    "k_H2S_total_1atm": r"$K_\mathrm{H_2S}^\mathrm{T0}$",
    "k_H2O_sws_1atm": r"$K_w^\mathrm{S0}$",
    "factor_k_H3PO4": r"$P_\mathrm{P1}$",
    "factor_k_H2PO4": r"$P_\mathrm{P2}$",
    "factor_k_HPO4": r"$P_\mathrm{P3}$",
    "BOH4": r"$[\mathrm{B(OH)}_4^–]$",
    "BOH3": r"$[\mathrm{B(OH)}_3]$",
    "OH": r"$[\mathrm{OH}^–]$",
    "H": r"$[\mathrm{H}^+]^*$",
    "H_free": r"$[\mathrm{H}^+]^\mathrm{F}$",
    "H3SiO4": r"$[\mathrm{H}_3\mathrm{SiO}_4^–]$",
    "H4SiO4": r"$[\mathrm{H}_4\mathrm{SiO}_4]$",
    "HSO4": r"$[\mathrm{HSO}_4^–]$",
    "SO4": r"$[\mathrm{SO}_4^{2–}]$",
    "HF": "[HF]",
    "F": r"$[\mathrm{F}^-]$",
    "NH3": r"$[\mathrm{NH}_3]$",
    "NH4": r"$[\mathrm{NH}_4^+]$",
    "H2S": r"$[\mathrm{H_2S}]$",
    "HS": r"$[\mathrm{HS}^–]$",
    "alkalinity": r"$A_\mathrm{T}$",
    "fugacity_factor": "$ƒ$",
    "vp_factor": "$v$",
    "pCO2": r"$p\mathrm{CO}_2$",
    "xCO2": r"$x\mathrm{CO}_2$",
    "k_aragonite": r"$K_\mathrm{a}^*$",
    "k_calcite": r"$K_\mathrm{c}^*$",
    "saturation_aragonite": r"$Ω_\mathrm{a}$",
    "saturation_calcite": r"$Ω_\mathrm{c}$",
    "pH_total": r"pH$_\mathrm{T}$",
    "pH_sws": r"pH$_\mathrm{S}$",
    "pH_free": r"pH$_\mathrm{F}$",
    "pH_nbs": r"pH$_\mathrm{N}$",
    "substrate_inhibitor_ratio": "SIR",
    "gamma_alkalinity": r"$\gamma_{A_\mathrm{T}}$",
    "gamma_dic": r"$\gamma_{C_\mathrm{T}}$",
    "beta_alkalinity": r"$\beta_{A_\mathrm{T}}$",
    "beta_dic": r"$\beta_{C_\mathrm{T}}$",
    "omega_alkalinity": r"$\omega_{A_\mathrm{T}}$",
    "omega_dic": r"$\omega_{C_\mathrm{T}}$",
    "Q_isocap": "$Q$",
    "Q_isocap_approx": "$Q_x$",
    "psi": r"$\psi$",
    "revelle_factor": r"$R_\mathrm{F}$",
}

# Parameters that do not change between input and output conditions
condition_independent = [
    "alkalinity",
    "dic",
    "salinity",
    "ionic_strength",
    "Ca",
    "total_sulfate",
    "total_fluoride",
    "total_ammonia",
    "total_phosphate",
    "total_sulfide",
    "total_silicate",
    "total_borate",
]


class CO2System:
    def __init__(self, values=None, opts=None):
        if values is None:
            values = {}
        values = values.copy()
        # Get icase
        core_known = np.array([v in values for v in parameters_core])
        icase_all = np.arange(1, len(parameters_core) + 1)
        icase = icase_all[core_known]
        assert len(icase) < 3, "You may not provide more than 2 known core parameters."
        if len(icase) == 0:
            icase = np.array(0)
        elif len(icase) == 2:
            icase = icase[0] * 100 + icase[1]
        self.icase = icase.item()
        self.opts = default_opts.copy()
        # Assign opts
        if opts is not None:
            for k, v in opts.items():
                assert (
                    v in get_funcs_opts[k].keys()
                ), "{} is not allowed for {}!".format(v, k)
            self.opts.update(opts)
        # Deal with tricky special cases
        if self.icase != 207:
            self.opts.pop("opt_HCO3_root")
        # Assemble graphs and computation functions
        self.graph, self.funcs, self.values = self._assemble(self.icase, values)

    def _assemble(self, icase, values):
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
        # Assign default values
        values = values.copy()
        for k, v in default_values.items():
            if k not in values:
                values[k] = v
                graph.add_node(k)
        # Save arguments
        for k, v in values.items():
            if v is not None:
                # state 1 means that the value was provided as an argument
                nx.set_node_attributes(graph, {k: 1}, name="state")
        return graph, funcs, values

    def _get(self, parameters, graph, funcs, values, save_steps, verbose):

        def printv(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # needs: which intermediate parameters we need to get the requested parameters
        graph_unknown = graph.copy()
        graph_unknown.remove_nodes_from([v for v in values if v not in parameters])
        needs = parameters.copy()
        for p in parameters:
            needs = needs | nx.ancestors(graph_unknown, p)
        # The got counter increments each time we successfully get a value, either from
        # the arguments, already-calculated values, or by calculating it.
        # The loop stops once got reaches the number of parameters in `needs`, because
        # then we're done.
        got = 0
        # We will cycle through the set of needed parameters
        needs_cycle = itertools.cycle(needs)
        self_values = values.copy()  # what is already known
        results = {}  # values for the requested parameters will go in here
        while got < len(needs):
            p = next(needs_cycle)
            printv("")
            printv(p)
            if p in self_values:
                if p not in results:
                    results[p] = self_values[p]
                    got += 1
                    printv("{} is available!".format(p))
            else:
                priors = graph.pred[p]
                if len(priors) == 0 or all([r in self_values for r in priors]):
                    printv("Calculating {}".format(p))
                    self_values[p] = funcs[p](
                        *[
                            self_values[r]
                            for r in funcs[p].__code__.co_varnames[
                                : funcs[p].__code__.co_argcount
                            ]
                        ]
                    )
                    # state 2 means that the value was calculated internally
                    if save_steps:
                        nx.set_node_attributes(graph, {p: 2}, name="state")
                        for f in funcs[p].__code__.co_varnames[
                            : funcs[p].__code__.co_argcount
                        ]:
                            nx.set_edge_attributes(graph, {(f, p): 2}, name="state")
                    results[p] = self_values[p]
                    got += 1
            printv("Got", got, "of", len(set(needs)))
        # Get rid of jax overhead on results
        for k, v in results.items():
            try:
                results[k] = v.item()
            except (AttributeError, ValueError):
                pass
            try:
                results[k] = v.__array__()
            except AttributeError:
                pass
        if save_steps:
            for k, v in self_values.items():
                try:
                    self_values[k] = v.item()
                except (AttributeError, ValueError):
                    pass
                try:
                    self_values[k] = v.__array__()
                except AttributeError:
                    pass
            values.update(self_values)
        return results, graph, values

    def solve(self, parameters=None, save_steps=True, verbose=False):
        """Calculate and return parameter(s) and (optionally) save them internally.

        Parameters
        ----------
        parameters : str or list of str, optional
            Which parameter(s) to calculate and save, by default None, in which case all
            possible parameters are calculated and returned.
        save_steps : bool, optional
            Whether to save non-requested parameters calculated during intermediate
            calculation steps in CO2System.values, by default True.
        verbose : bool, optional
            Whether to print calculation status messages, by default False.

        Returns
        -------
        results : dict
            The value(s) of the requested parameter(s).
        """
        if parameters is None:
            parameters = list(self.graph.nodes)
        elif isinstance(parameters, str):
            parameters = [parameters]
        parameters = set(parameters)  # get rid of duplicates
        # Solve the system
        results, self.graph, self.values = self._get(
            parameters, self.graph, self.funcs, self.values, save_steps, verbose
        )
        results = {k: v for k, v in results.items() if k in parameters}
        return results

    def adjust(self, temperature=None, pressure=None, values=None, save_steps=False):
        if self.icase > 100:
            core = self.solve(parameters=["alkalinity", "dic"], save_steps=save_steps)
        elif self.icase in [4, 5, 8, 9]:
            core = self.solve(parameters="fCO2", save_steps=save_steps)
            # TODO convert core["fCO2"] to new temperature here!
        if values is None:
            values = {}
        else:
            values = values.copy()
        for k in condition_independent:
            if k in self.values:
                values[k] = self.values[k]
            elif k in core:
                values[k] = core[k]
        for k, v in core.items():
            if k not in condition_independent:
                values[k] = core[k]
        if temperature is not None:
            values["temperature"] = temperature
        else:
            values["temperature"] = self.values["temperature"]
        if pressure is not None:
            values["pressure"] = pressure
        else:
            values["pressure"] = self.values["pressure"]
        print(values)
        return CO2System(values=values, opts=self.opts)

    def plot_graph(
        self,
        ax=None,
        exclude_nodes=None,
        prog_graphviz="neato",
        show_tsp=True,
        show_unknown=True,
        show_isolated=True,
        skip_nodes=None,
    ):
        """Draw a graph showing the relationships between the different parameters.

        Parameters
        ----------
        ax : matplotlib axes, optional
            The axes, by default None, in which case new axes are generated.
        conditions : str, optional
            Whether to show the graph for the "input" or "output" condition
            calculations, by default "input".
        exclude_nodes : list of str, optional
            List of nodes to exclude from the plot, by default None.
        prog_graphviz : str, optional
            Name of Graphviz layout program, by default "neato".
        show_tsp : bool, optional
            Whether to show temperature, salinity and pressure nodes, by default False.
        show_unknown : bool, optional
            Whether to show nodes for parameters that have not (yet) been calculated,
            by default True.
        show_isolated : bool, optional
            Whether to show nodes for parameters that are not connected to the graph,
            by default True.
        skip_nodes


        Returns
        -------
        matplotlib axes
            The axes on which the graph is plotted.
        """
        if ax is None:
            ax = plt.subplots(dpi=300, figsize=(8, 7))[1]
        self_graph = self.graph.copy()
        node_states = nx.get_node_attributes(self_graph, "state", default=0)
        edge_states = nx.get_edge_attributes(self_graph, "state", default=0)
        if not show_tsp:
            self_graph.remove_nodes_from(["pressure", "salinity", "temperature"])
        if not show_unknown:
            self_graph.remove_nodes_from([n for n, s in node_states.items() if s == 0])
        if not show_isolated:
            self_graph.remove_nodes_from(
                [n for n, d in dict(self_graph.degree).items() if d == 0]
            )
        if exclude_nodes:
            if isinstance(exclude_nodes, str):
                exclude_nodes = [exclude_nodes]
            self_graph.remove_nodes_from(exclude_nodes)
        if skip_nodes:
            if isinstance(skip_nodes, str):
                skip_nodes = [skip_nodes]
            for n in skip_nodes:
                for p, s in itertools.product(
                    self_graph.predecessors(n), self_graph.successors(n)
                ):
                    self_graph.add_edge(p, s)
                    if edge_states[(p, n)] + edge_states[(n, s)] == 4:
                        new_state = {(p, s): 2}
                    else:
                        new_state = {(p, s): 0}
                    nx.set_edge_attributes(self_graph, new_state, name="state")
                    edge_states.update(new_state)
                self_graph.remove_node(n)
        state_colours = {0: "xkcd:grey", 1: "xkcd:grass", 2: "xkcd:azure"}
        node_colour = [state_colours[node_states[n]] for n in nx.nodes(self_graph)]
        edge_colour = [state_colours[edge_states[e]] for e in nx.edges(self_graph)]
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog=prog_graphviz)
        node_labels = {k: k for k in self_graph.nodes}
        for k, v in set_node_labels.items():
            if k in node_labels:
                node_labels[k] = v
        nx.draw_networkx(
            self_graph,
            ax=ax,
            clip_on=False,
            with_labels=True,
            node_color=node_colour,
            edge_color=edge_colour,
            pos=pos,
            labels=node_labels,
        )
        return ax
