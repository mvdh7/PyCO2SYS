# %%
import networkx as nx
from matplotlib import pyplot as plt

from tests.function_graph import FunctionGraph

funcs = {
    "beta": lambda: 1,
    "gamma": lambda alpha, beta: alpha + beta,
    "delta": lambda beta, gamma: beta + gamma,
    "epsilon": lambda alpha, delta: alpha + delta,
    "phi": lambda delta, gamma: 2 * delta + gamma,
}
defaults = dict(
    alpha=0.0,
    gamma=0.0,
)
shortcuts = dict(
    a="alpha",
    b="beta",
    c="gamma",
    d="delta",
    e="epsilon",
    g="gamma",
    f="phi",
)

fg = FunctionGraph(
    defaults=defaults,
    funcs=funcs,
    shortcuts=shortcuts,
)
print(fg.data)

data = dict(
    # ALPHA=1.0,
    # d=3,
    # b=2,
    # f=3,
    g=4.0,
    # h=2,
)
fg.set_data(**data)
print(fg.data)
fg.solve("e")
print(fg.data)
result_f = fg.f
print(fg.data)
results = fg[["f", "g"]]

get_d = fg.get_func_of("d")
kwargs = {k: fg[k] for k in fg.nodes_original}
d = get_d(**kwargs)

get_d_from_a = fg.get_func_of_from_wrt(get_d, "a")
get_dd_dg = fg.get_grad_func("d", "g")
# TODO make a more convenient way to get the args and kwargs for get_func_of_from_wrt
dd_dg = get_dd_dg(kwargs["gamma"], **{k: v for k, v in kwargs.items() if k != "gamma"})
get_dd_da = fg.get_grad_func("d", "a")
dd_da = get_dd_da(kwargs["alpha"], **{k: v for k, v in kwargs.items() if k != "alpha"})

# %%
pos = nx.nx_agraph.graphviz_layout(fg.graph, prog="dot")
fig, ax = plt.subplots()
nx.draw_networkx(
    fg.graph,
    pos=pos,
    nodelist=fg.graph.nodes,
    node_color=[
        nx.get_node_attributes(fg.graph, "state", default=-1)[n] for n in fg.graph.nodes
    ],
    vmin=-1,
    vmax=3,
)
