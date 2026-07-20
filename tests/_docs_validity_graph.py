# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2


co2s = pyco2.sys(t=[10, 20, 30], s=np.vstack([15, 35]))
co2s.check_valid("pk1")
# TODO invalid only if a property actually affects result
# e.g. pcx when pressure = 0
# e.g. pk_salt when total_salt = 0

graph = co2s.v.get_graph()
c_valid = "xkcd:turquoise blue"
c_invalid = "xkcd:light red"
fig, ax = plt.subplots(figsize=(5, 5))
pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
# Assign node colour based on valid or invalid
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
# Edge colour matches valid/invalid node colour,
# edge style shows direct (solid) vs indirect (dashed) (in)validity
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
# Edge labels indicate what percentage of the parameter values are valid
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
    graph, ax=ax, pos=pos, labels={n: pyco2.labels[n] for n in graph.nodes}
)
ax.axis("off")
fig.tight_layout()
fig.savefig("docs/img/fig_valid.png")
