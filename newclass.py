import itertools
import networkx as nx
from jax import numpy as np
from matplotlib import pyplot as plt


def get_ts(salinity, temperature):
    return salinity * temperature


def get_tsp(pressure, ts):
    return pressure * ts


def get_tsp_fCO2(fCO2, tsp):
    return fCO2 * tsp


def get_tsp_HCO3(HCO3, tsp):
    return HCO3 * tsp


def pH_from_alkalinity_dic(alkalinity, dic):
    return alkalinity - dic


def fCO2_from_dic_pH(dic, pH):
    return dic + pH


def HCO3_from_dic_pH(dic, pH):
    return dic + pH


def CO3_from_dic_pH(dic, pH):
    return dic + pH


def pCO2_from_fCO2(fCO2):
    return 1.1 * fCO2


get_funcs = {
    "ts": get_ts,
    "tsp": get_tsp,
    "tsp_fCO2": get_tsp_fCO2,
    "tsp_HCO3": get_tsp_HCO3,
}


links = nx.DiGraph()
for k, func in get_funcs.items():
    func_args = func.__code__.co_varnames
    for f in func_args:
        links.add_edge(f, k)

nx.draw_planar(links, with_labels=True)

# %%
get_funcs_core = {}
get_funcs_core[0] = {}
get_funcs_core[5] = {
    "pCO2": pCO2_from_fCO2,
}
get_funcs_core[102] = {
    "pH": pH_from_alkalinity_dic,
    "pCO2": pCO2_from_fCO2,
    "fCO2": fCO2_from_dic_pH,
    "HCO3": HCO3_from_dic_pH,
    "CO3": CO3_from_dic_pH,
}
# Automatically set up graph for each icase based on the corresponding function names
# in get_funcs_core
links_core = {}
for icase, funcs in get_funcs_core.items():
    links_core[icase] = nx.DiGraph()
    for t, func in get_funcs_core[icase].items():
        for f in func.__name__.split("_")[2:]:
            links_core[icase].add_edge(f, t)

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
    "salinity": 35.0,
    "pressure": 0.0,
}

default_opts = {
    "opt_k_carbonic": 10,
    "opt_total_borate": 1,
}
allowed_opts = {
    "opt_k_carbonic": range(1, 19),
    "opt_total_borate": range(1, 4),
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
        self.links = nx.compose(links, links_core[self.icase])
        self.get_funcs = get_funcs.copy()
        self.get_funcs.update(get_funcs_core[self.icase])
        # Assign opts
        self.opts = default_opts.copy()
        if opts is not None:
            for k, v in opts.items():
                assert v in allowed_opts[k], "{} is not allowed for {}!".format(v, k)
            self.opts.update(opts)
        # Assign default values, if requested
        if use_default_values:
            values = values.copy()
            for k, v in default_values.items():
                if k not in values:
                    values[k] = v
        # Save arguments
        self.values = {}
        for k, v in values.items():
            if k != "self" and v is not None:
                self.values[k] = v
                # state 1 means that the value was provided as an argument
                nx.set_node_attributes(self.links, {k: 1}, name="state")

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
        links_unknown = self.links.copy()
        links_unknown.remove_nodes_from(self.values)
        needs = parameters.copy()
        for p in parameters:
            needs = needs | nx.ancestors(links_unknown, p)
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
                priors = self.links.pred[p]
                if len(priors) == 0 or all([r in self_values for r in priors]):
                    print("Calculating {}".format(p))
                    self_values[p] = self.get_funcs[p](
                        *[
                            self_values[r]
                            for r in self.get_funcs[p].__code__.co_varnames
                        ]
                    )
                    # state 2 means that the value was calculated internally
                    if save_steps:
                        nx.set_node_attributes(self.links, {p: 2}, name="state")
                        for f in self.get_funcs[p].__code__.co_varnames:
                            nx.set_edge_attributes(
                                self.links, {(f, p): 2}, name="state"
                            )
                    results[p] = self_values[p]
                    got += 1
            print("Got", got, "of", len(set(needs)))
        if save_steps:
            self.values.update(self_values)
        return results

    def plot_links(self, ax=None, show_tsp=True, show_missing=True):
        """Draw a graph showing the relationships between the different parameters.

        Parameters
        ----------
        ax : matplotlib axes, optional
            The axes, by default None, in which case new axes are generated.
        show_tsp : bool, optional
            Whether to show temperature, salinity and pressure nodes, by default False.
        show_missing : bool, optional
            Whether to show nodes for parameters that have not (yet) been calculated,
            by default True.

        Returns
        -------
        matplotlib axes
            The axes on which the graph is plotted.
        """
        if ax is None:
            ax = plt.subplots(dpi=300, figsize=(8, 7))[1]
        self_links = self.links.copy()
        node_states = nx.get_node_attributes(self_links, "state", default=0)
        edge_states = nx.get_edge_attributes(self_links, "state", default=0)
        if not show_tsp:
            self_links.remove_nodes_from(["pressure", "salinity", "temperature"])
        if not show_missing:
            self_links.remove_nodes_from([n for n, s in node_states.items() if s == 0])
        state_colours = {0: "xkcd:grey", 1: "xkcd:grass", 2: "xkcd:azure"}
        node_colour = [state_colours[node_states[n]] for n in nx.nodes(self_links)]
        edge_colour = [state_colours[edge_states[e]] for e in nx.edges(self_links)]
        nx.draw_planar(
            self_links,
            ax=ax,
            clip_on=False,
            with_labels=True,
            node_color=node_colour,
            edge_color=edge_colour,
        )
        return ax


sys = CO2System(opts=dict(opt_total_borate=1))
test = sys.get(
    [
        "tsp",
    ],
    save_steps=True,
)
sys.plot_links(show_tsp=True, show_missing=True)
