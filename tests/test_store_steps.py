# %%
from jax import numpy as np

from PyCO2SYS.classes.function_graph import FunctionGraph


# alpha: standard input with default
# beta: computed by function with no inputs
# gamma, epsilon: computed from part of coeffs and other inputs/intermediates
# delta: standard intermediate
# phi: standard end product
# coeffs: set of coefficients used in parts by multiple other steps
funcs = {
    "beta": lambda: 1.5,
    "gamma": lambda coeffs, alpha, beta: (
        coeffs[0] + np.log(alpha) * coeffs[1] + beta * coeffs[2]
    ),
    "Delta": lambda beta, gamma: beta + np.sqrt(gamma),
    "epsilon": lambda coeffs, alpha, Delta: (
        coeffs[3] * alpha**2 + coeffs[4] * Delta * alpha
    ),
    "phi": lambda Delta, epsilon: 2 * Delta + epsilon,
    # Below is to get covariances between two (or more) parameters
    "combi": lambda gamma, phi: np.array([gamma, phi]),
}
defaults = dict(
    alpha=12.0,
    coeffs=np.array([0.0, 1, 1, 1, 1]),
)
shortcuts = dict(
    a="alpha",
    b="beta",
    c="gamma",
    delta="Delta",
    d="Delta",
    e="epsilon",
    g="gamma",
    f="phi",
)
data = dict(
    alpha=np.vstack([1.5, 2.5, 3.5]),
    b=np.vstack([1.5, 2.5, 3.5]),
)


def test_store_steps_0():
    fg = (
        FunctionGraph(
            defaults=defaults,
            funcs=funcs,
            shortcuts=shortcuts,
            no_store="gamma",
        )
        .set_data(**data)
        .solve("e", store_steps=0)
    )
    for v in ["gamma", "Delta"]:
        assert v not in fg.data
        assert "state" not in fg.graph.nodes[v]
    assert "epsilon" in fg.data
    assert fg.graph.nodes["epsilon"]["state"] == 3


def test_store_steps_1():
    fg = (
        FunctionGraph(
            defaults=defaults,
            funcs=funcs,
            shortcuts=shortcuts,
            no_store="gamma",
        )
        .set_data(**data)
        .solve("e", store_steps=1)
    )
    assert "gamma" not in fg.data
    assert "state" not in fg.graph.nodes["gamma"]
    assert "Delta" in fg.data
    assert fg.graph.nodes["Delta"]["state"] == 2
    assert "epsilon" in fg.data
    assert fg.graph.nodes["epsilon"]["state"] == 3


def test_store_steps_2():
    fg = (
        FunctionGraph(
            defaults=defaults,
            funcs=funcs,
            shortcuts=shortcuts,
            no_store="gamma",
        )
        .set_data(**data)
        .solve("e", store_steps=2)
    )
    for v in ["gamma", "Delta"]:
        assert v in fg.data
        assert fg.graph.nodes[v]["state"] == 2
    assert "epsilon" in fg.data
    assert fg.graph.nodes["epsilon"]["state"] == 3


# # For testing the tests
# import networkx as nx
# from matplotlib import pyplot as plt


# fg = (
#     FunctionGraph(
#         defaults=defaults,
#         funcs=funcs,
#         shortcuts=shortcuts,
#         no_store="gamma",
#     )
#     .set_data(**data)
#     .solve("e", store_steps=1)
# )
# pos = nx.nx_agraph.graphviz_layout(fg.graph, prog="dot")
# fig, ax = plt.subplots()
# nx.draw_networkx(
#     fg.graph,
#     pos=pos,
#     nodelist=fg.graph.nodes,
#     node_color=[
#         nx.get_node_attributes(fg.graph, "state", default=-1)[n]
#         for n in fg.graph.nodes
#     ],
#     vmin=-1,
#     vmax=3,
# )

# test_store_steps_0()
# test_store_steps_1()
# test_store_steps_2()
