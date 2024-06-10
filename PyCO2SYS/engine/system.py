# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
import itertools
import networkx as nx
from jax import numpy as np
from matplotlib import pyplot as plt
from . import convert, equilibria, salts

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
    "nbs_to_sws": convert.pH_nbs_to_sws,
    "total_to_sws_1atm": lambda total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm: convert.pH_total_to_sws(
        total_fluoride, total_sulfate, k_HF_free_1atm, k_HSO4_free_1atm
    ),
    # Equilibrium constants at 1 atm and on the seawater pH scale
    "k_H2S_sws_1atm": lambda k_H2S_total_1atm, total_to_sws_1atm: (
        k_H2S_total_1atm * total_to_sws_1atm
    ),
    # Pressure correction factors for equilibrium constants
    # Equilibrium constants at pressure and on the seawater pH scale
    # Equilibrium constants at pressure and on the requested pH scale (YES, HERE!)
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
get_funcs_opts["opt_fH"] = {
    1: dict(fH=convert.fH_TWB82),
    2: dict(fH=convert.fH_PTBO87),
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
    "opt_fH": 1,
    "opt_k_HF": 1,
    "opt_k_HSO4": 1,
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

    def get(self, parameters, save_steps=True):
        """Calculate and return parameter(s) and (optionally) save them internally.

        Parameters
        ----------
        parameters : str or list of str
            Which parameter(s) to calculate and save.
        save_steps : bool, optional
            Whether to save non-requested parameters calculated during intermediate
            calculation steps in CO2System.values, by default True.

        Returns
        -------
        results : dict
            The value(s) of the requested parameter(s).
            Also saved in CO2System.values if save_steps is True.
        """
        if isinstance(parameters, str):
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
