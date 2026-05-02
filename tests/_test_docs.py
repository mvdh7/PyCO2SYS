# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2
from PyCO2SYS.engine import node_labels


co2s = pyco2.sys(t=[10, 20, 30], s=np.vstack([0, 35]))
co2s.check_valid(
    ["pk1"]
    # ignore=None,
    # nan_invalid=False,
)

# print(co2s.valid)
# print(co2s.valid.direct)
# print(co2s.valid.indirect)
# co2s.valid.pk_H2CO3
# co2s.valid.parts.total_borate

# co2s = pyco2.sys()
# co2s.solve(store_steps=2)  # Solve for all parameters
# co2s.plot_graph(mode="valid")

graph = co2s.v.get_graph()

fig, ax = plt.subplots(figsize=(5, 5))
pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
nx.draw_networkx_nodes(
    graph,
    ax=ax,
    pos=pos,
    node_color="xkcd:light grey",
)
edge_colors = {"direct": "xkcd:light red", "indirect": "xkcd:sea blue"}
nx.draw_networkx_edges(
    graph,
    ax=ax,
    pos=pos,
    edge_color=[edge_colors[graph.edges[e]["type"]] for e in graph.edges],
)
nx.draw_networkx_edge_labels(
    graph,
    ax=ax,
    pos=pos,
    edge_labels={
        e: str(np.round(graph.edges[e]["pct"]).astype(int))
        for e in graph.edges
    },
    # rotate=False,
    bbox={
        "boxstyle": "round",
        "ec": (1.0, 1.0, 1.0, 0.0),
        "fc": (1.0, 1.0, 1.0, 0.0),
    },
)
nx.draw_networkx_labels(
    graph, ax=ax, pos=pos, labels={n: node_labels[n] for n in graph.nodes}
)
fig.tight_layout()
