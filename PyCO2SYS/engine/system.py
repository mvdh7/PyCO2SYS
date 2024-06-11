# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
import itertools
import networkx as nx
from jax import numpy as np
from matplotlib import pyplot as plt
from .. import constants, convert, equilibria, salts

# Define functions for calculations that depend neither on icase nor opts:
get_funcs = {
    # Total salt contents
    "ionic_strength": salts.ionic_strength_DOE94,
    "total_fluoride": salts.total_fluoride_R65,
    "total_sulfate": salts.total_sulfate_MR66,
    # Equilibrium constants at 1 atm and on reported pH scale
    "k_H2S_total_1atm": equilibria.p1atm.k_H2S_total_YM95,
    # pH scale conversion factors at 1 atm
    "free_to_sws_1atm": lambda total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm: convert.pH_free_to_sws(
        total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm
    ),
    "nbs_to_sws": convert.pH_nbs_to_sws,  # because fH doesn't get pressure-corrected
    "total_to_sws_1atm": lambda total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm: convert.pH_total_to_sws(
        total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm
    ),
    # Equilibrium constants at 1 atm and on the seawater pH scale
    "k_H2S_sws_1atm": lambda k_H2S_total_1atm, total_to_sws_1atm: (
        k_H2S_total_1atm * total_to_sws_1atm
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
    "k_H2CO3_sws": lambda k_H2CO3_sws_1atm, factor_k_H2CO3: k_H2CO3_sws_1atm
    * factor_k_H2CO3,
    "k_HCO3_sws": lambda k_HCO3_sws_1atm, factor_k_HCO3: k_HCO3_sws_1atm
    * factor_k_HCO3,
    # Equilibrium constants at pressure and on the requested pH scale
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
}

# Define functions for calculations that depend on icase:
get_funcs_core = {}
get_funcs_core[0] = {}
# get_funcs_core[5] = {
#     "pCO2": pCO2_from_fCO2,
# }
# get_funcs_core[102] = {
#     "pH": pH_from_alkalinity_dic,
#     "pCO2": pCO2_from_fCO2,
#     "fCO2": fCO2_from_dic_pH,
#     "HCO3": HCO3_from_dic_pH,
#     "CO3": CO3_from_dic_pH,
# }

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
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, total_to_sws_1atm: (
            k_H2CO3_total_1atm * total_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, total_to_sws_1atm: (
            k_HCO3_total_1atm * total_to_sws_1atm
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
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, total_to_sws_1atm: (
            k_H2CO3_total_1atm * total_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, total_to_sws_1atm: (
            k_HCO3_total_1atm * total_to_sws_1atm
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
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, total_to_sws_1atm: (
            k_H2CO3_total_1atm * total_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, total_to_sws_1atm: (
            k_HCO3_total_1atm * total_to_sws_1atm
        ),
    ),
    17: dict(
        k_H2CO3_sws_1atm=equilibria.p1atm.k_H2CO3_sws_WMW14,
        k_HCO3_total_1atm=equilibria.p1atm.k_HCO3_total_SB21,
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, total_to_sws_1atm: (
            k_HCO3_total_1atm * total_to_sws_1atm
        ),
    ),
    18: dict(
        k_H2CO3_total_1atm=equilibria.p1atm.k_H2CO3_total_PLR18,
        k_HCO3_total_1atm=equilibria.p1atm.k_HCO3_total_PLR18,
        k_H2CO3_sws_1atm=lambda k_H2CO3_total_1atm, total_to_sws_1atm: (
            k_H2CO3_total_1atm * total_to_sws_1atm
        ),
        k_HCO3_sws_1atm=lambda k_HCO3_total_1atm, total_to_sws_1atm: (
            k_HCO3_total_1atm * total_to_sws_1atm
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
        k_BOH3_sws_1atm=lambda k_BOH3_total_1atm, total_to_sws_1atm: (
            k_BOH3_total_1atm * total_to_sws_1atm
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
        k_NH3_sws_1atm=lambda k_NH3_total_1atm, total_to_sws_1atm: (
            k_NH3_total_1atm * total_to_sws_1atm
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
    1: dict(sws_to_opt=convert.pH_sws_to_total),
    2: dict(sws_to_opt=lambda: 1.0),
    3: dict(sws_to_opt=convert.pH_sws_to_free),
    4: dict(sws_to_opt=convert.pH_sws_to_nbs),
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

# Automatically set up graph for calculations that depend neither on icase nor opts
# based on the function names and signatures in get_funcs
graph = nx.DiGraph()
for k, func in get_funcs.items():
    fcode = func.__code__
    func_args = fcode.co_varnames[: fcode.co_argcount]
    for f in func_args:
        graph.add_edge(f, k)

# Automatically set up graph for each icase based on the function names and signatures
# in get_funcs_core
graph_core = {}
for icase, funcs in get_funcs_core.items():
    graph_core[icase] = nx.DiGraph()
    for t, func in get_funcs_core[icase].items():
        for f in func.__name__.split("_")[2:]:
            graph_core[icase].add_edge(f, t)

# Automatically set up graph for each opt based on the function names and signatures in
# get_funcs_opts
graph_opts = {}
for o, opts in get_funcs_opts.items():
    graph_opts[o] = {}
    for opt, funcs in opts.items():
        graph_opts[o][opt] = nx.DiGraph()
        for k, func in funcs.items():
            fcode = func.__code__
            func_args = fcode.co_varnames[: fcode.co_argcount]
            for f in func_args:
                graph_opts[o][opt].add_edge(f, k)

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
    "temperature": 25.0,
    "total_ammonia": 0.0,
    "total_phosphate": 0.0,
    "total_silicate": 0.0,
    "total_sulfide": 0.0,
    "salinity": 35.0,
    "pressure": 0.0,
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
}


class CO2System:
    def __init__(self, values=None, opts=None, use_default_values=True):
        if values is None:
            values = {}
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
        # Assign opts
        self.opts = default_opts.copy()
        if opts is not None:
            for k, v in opts.items():
                assert (
                    v in get_funcs_opts[k].keys()
                ), "{} is not allowed for {}!".format(v, k)
            self.opts.update(opts)
        # Assemble graph and functions
        self.graph = nx.compose(graph, graph_core[self.icase])
        self.get_funcs = get_funcs.copy()
        self.get_funcs.update(get_funcs_core[self.icase])
        for opt, v in self.opts.items():
            self.graph = nx.compose(self.graph, graph_opts[opt][v])
            self.get_funcs.update(get_funcs_opts[opt][v])
        # Assign default values, if requested
        if use_default_values:
            values = values.copy()
            for k, v in default_values.items():
                if k not in values:
                    values[k] = v
                    self.graph.add_node(k)
        # Save arguments
        self.values = {}
        for k, v in values.items():
            if k != "self" and v is not None:
                self.values[k] = v
                # state 1 means that the value was provided as an argument
                nx.set_node_attributes(self.graph, {k: 1}, name="state")

    def get(self, parameters=None, save_steps=True):
        """Calculate and return parameter(s) and (optionally) save them internally.

        Parameters
        ----------
        parameters : str or list of str, optional
            Which parameter(s) to calculate and save, by default None, in which case
            all possible parameters are calculated and returned.
        save_steps : bool, optional
            Whether to save non-requested parameters calculated during intermediate
            calculation steps in CO2System.values, by default True.

        Returns
        -------
        results : dict
            The value(s) of the requested parameter(s).
            Also saved in CO2System.values if save_steps is True.
        """
        if parameters is None:
            parameters = list(self.graph.nodes)
        elif isinstance(parameters, str):
            parameters = [parameters]
        parameters = set(parameters)  # get rid of duplicates
        # needs: which intermediate parameters we need to get the requested parameters
        graph_unknown = self.graph.copy()
        graph_unknown.remove_nodes_from([v for v in self.values if v not in parameters])
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
        self_values = self.values.copy()  # what is already known
        results = {}  # values for the requested parameters will go in here
        while got < len(needs):
            p = next(needs_cycle)
            print("")
            print(p)
            if p in self_values:
                if p not in results:
                    results[p] = self_values[p]
                    got += 1
                    print("{} is available!".format(p))
            else:
                priors = self.graph.pred[p]
                if len(priors) == 0 or all([r in self_values for r in priors]):
                    print("Calculating {}".format(p))
                    self_values[p] = self.get_funcs[p](
                        *[
                            self_values[r]
                            for r in self.get_funcs[p].__code__.co_varnames[
                                : self.get_funcs[p].__code__.co_argcount
                            ]
                        ]
                    )
                    # state 2 means that the value was calculated internally
                    if save_steps:
                        nx.set_node_attributes(self.graph, {p: 2}, name="state")
                        for f in self.get_funcs[p].__code__.co_varnames[
                            : self.get_funcs[p].__code__.co_argcount
                        ]:
                            nx.set_edge_attributes(
                                self.graph, {(f, p): 2}, name="state"
                            )
                    results[p] = self_values[p]
                    got += 1
            print("Got", got, "of", len(set(needs)))
        # Get rid of jax overhead on results
        for k, v in results.items():
            try:
                results[k] = v.item()
            except:
                pass
        if save_steps:
            for k, v in self_values.items():
                try:
                    self_values[k] = v.item()
                except:
                    pass
            self.values.update(self_values)
        return results

    def plot_graph(
        self,
        ax=None,
        prog_graphviz="neato",
        show_tsp=True,
        show_unknown=True,
        show_isolated=True,
    ):
        """Draw a graph showing the relationships between the different parameters.

        Parameters
        ----------
        ax : matplotlib axes, optional
            The axes, by default None, in which case new axes are generated.
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
        state_colours = {0: "xkcd:grey", 1: "xkcd:grass", 2: "xkcd:azure"}
        node_colour = [state_colours[node_states[n]] for n in nx.nodes(self_graph)]
        edge_colour = [state_colours[edge_states[e]] for e in nx.edges(self_graph)]
        pos = nx.nx_agraph.graphviz_layout(self.graph, prog=prog_graphviz)
        nx.draw_networkx(
            self_graph,
            ax=ax,
            clip_on=False,
            with_labels=True,
            node_color=node_colour,
            edge_color=edge_colour,
            pos=pos,
        )
        return ax
