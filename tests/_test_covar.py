# %%
from inspect import signature

import jax
import networkx as nx
from jax import grad
from jax import numpy as np

import PyCO2SYS as pyco2
from PyCO2SYS.engine import (
    CO2System,
    assemble_graph,
    condition_independent,
    make_positional,
)

jax.config.update("jax_enable_x64", True)


def dic_from_ph_alkalinity(ph, alkalinity, temperature):
    return alkalinity * ph / temperature


def pco2_from_dic_alkalinity(dic, alkalinity, temperature):
    return temperature * (dic + alkalinity) / 200


def pco2_from_ph_alkalinity(ph, alkalinity, temperature_0, temperature_1):
    # Can't use this function in final solution - just for testing!
    dic = dic_from_ph_alkalinity(ph, alkalinity, temperature_0)
    return pco2_from_dic_alkalinity(dic, alkalinity, temperature_1)


# Set values
ph = 8.1
alkalinity = 2300.5
temperature_0 = 10.5
temperature_1 = 25.0

# Set uncertainties
u_ph = 0.01
u_alkalinity = 2.0
u_temperature_0 = 0.005
u_temperature_1 = 0.005

# Initial calculations
dic = dic_from_ph_alkalinity(ph, alkalinity, temperature_0)
pco2 = pco2_from_dic_alkalinity(dic, alkalinity, temperature_1)
pco2_direct = pco2_from_ph_alkalinity(ph, alkalinity, temperature_0, temperature_1)
print(dic, pco2, pco2_direct)

# Get gradients
d_dic = grad(dic_from_ph_alkalinity, argnums=(0, 1, 2))(ph, alkalinity, temperature_0)
print([d.item() for d in d_dic])

#
co2s = (
    pyco2.sys(pH=ph, alkalinity=alkalinity, temperature=temperature_0)
    .set_uncertainty(pH=0.01, alkalinity=2)
    .solve(["dic", "alkalinity"])
    .propagate()
)
testfunc = co2s._get_func_of("k_H2CO3")
tf = make_positional(testfunc)

co2a = co2s.adjust(temperature=12)

# New adjust approach, retaining old graph
# TODO incorporate into engine
# TODO make version for the fCO2-pCO2-xCO2-CO2aq-only icases
kwargs_adjust = {
    "temperature": 12,
    # "pressure": 1000,
}
# TODO ^ this needs to come automatically from kwargs to adjust

# To adjust to a different temperature/pressure, we need to know alkalinity
# and DIC for the original system.  First, we get the subgraph from the
# original system that contains just alkalinity, DIC, all their ancestors.
graph_pre = co2s.graph.subgraph(
    nx.ancestors(co2s.graph, "alkalinity")
    | nx.ancestors(co2s.graph, "dic")
    | {"alkalinity", "dic"}
)
# All of the nodes in graph_pre that are not condition-independent are now
# renamed with "__pre" appended, to keep the distinct from the same nodes under
# the adjusted conditions.  Temperature and pressure are also considered to be
# condition-independent if they were not adjusted.
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
            for k in signature(attrs["func"]).parameters.keys()
        ]
nx.set_node_attributes(graph_pre, args, name="args")
# graph_pre can now be merged with a new graph to compute everything from
# alkalinity and DIC.  The original system's `opts` are retained.
graph_adj = nx.compose(graph_pre, assemble_graph(102, co2s.opts))
# The new system will have the same set of user-provided parameter values as
# the original, but the ones that are condition-dependent get renamed with
# "__pre" appended.
data_pre = co2s[co2s.nodes_original]
for k, v in data_pre.copy().items():
    if k not in no_pre:
        data_pre[k + "__pre"] = data_pre.pop(k)
co2t = CO2System(graph=graph_adj, **data_pre, **kwargs_adjust)
# Parameters that have already been solved for in the original system will be
# copied across, so that they don't need solving for again.
for k, v in co2s.data.items():
    if k not in co2t:
        if k in no_pre:
            co2t.data[k] = v
        else:
            co2t.data[k + "__pre"] = v
# Uncertainties that were assigned in the original system are copied across.
uncertainty_pre = {}
for k, v in co2s.uncertainty.items():
    if not isinstance(v, dict):
        if k in no_pre:
            uncertainty_pre[k] = v
        else:
            uncertainty_pre[k + "__pre"] = v
co2t.set_uncertainty(**uncertainty_pre)
# Final housekeeping: the new CO2System will usually get its icase wrong,
# because it doesn't recognise parameters with keys ending "__pre".  Adjusted
# systems will get assigned whichever icase the original system had.  This
# doesn't affect any calculations, but it does affect __str__ and __repr__.
# TODO make it actually affect __str__ and __repr__
co2t.icase = co2s.icase
co2t.adjusted = True

co2t.solve("pH").propagate()
co2t.plot_graph(prog_graphviz="dot", mode="state")
