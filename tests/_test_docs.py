# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2
from PyCO2SYS.engine import node_labels


# co2s = pyco2.sys(t=[10, 20, 30], s=np.vstack([0, 35]))
co2s = pyco2.sys(t=[25, 25, 25], s=np.vstack([35, 35]))  # TODO P1 weird
co2s.check_valid("pk1")  # TODO nan_invalid switch
# TODO only invalid if a property actually affects result
# e.g. pcx when pressure = 0
# e.g. pk_salt when total_salt = 0

# NOTE just do full demo code for the graph in the docs, don't build it in
graph = co2s.v.get_graph()
c_valid = "xkcd:turquoise blue"
c_invalid = "xkcd:light red"
fig, ax = plt.subplots(figsize=(5, 5))
pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
node_color = []
for n, attrs in graph.nodes.items():
    if "pct" in attrs:
        if attrs["pct"] < 100:
            node_color.append(c_invalid)
        else:
            node_color.append(c_valid)
    else:
        node_color.append("xkcd:light grey")
nx.draw_networkx_nodes(
    graph,
    ax=ax,
    pos=pos,
    node_color=node_color,
)
nx.draw_networkx_edges(
    graph,
    ax=ax,
    pos=pos,
    edge_color=[
        c_invalid if graph.edges[e]["pct"] < 100 else c_valid
        for e in graph.edges
    ],
    style=[
        ":" if graph.edges[e]["type"] == "indirect" else "-"
        for e in graph.edges
    ],
)
nx.draw_networkx_edge_labels(
    graph,
    ax=ax,
    pos=pos,
    edge_labels={
        e: str(np.round(graph.edges[e]["pct"]).astype(int))
        for e in graph.edges
    },
    bbox={"alpha": 0},
)
nx.draw_networkx_labels(
    graph, ax=ax, pos=pos, labels={n: node_labels[n] for n in graph.nodes}
)
fig.tight_layout()
