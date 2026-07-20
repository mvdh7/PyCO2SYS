# %%
import networkx as nx
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2


co2s = pyco2.sys(dic=2150, pH=8.1).solve("pk1")
nodes = (
    nx.ancestors(co2s.graph, "pk_H2CO3")
    | nx.ancestors(co2s.graph, "pk_HCO3")
    | {"pk_H2CO3", "pk_HCO3"}
)
nodes = {n for n in nodes if not n.startswith("coeffs_")}
graph = co2s.graph.subgraph(nodes)
fig, ax = plt.subplots(figsize=(7, 5))
pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
node_color = []
for n in graph.nodes:
    if "state" in graph.nodes[n]:
        print(n, graph.nodes[n]["state"])
        if graph.nodes[n]["state"] in [0, 1]:
            node_color.append("xkcd:turtle green")
        elif graph.nodes[n]["state"] == 2:
            node_color.append("xkcd:cool blue")
        elif graph.nodes[n]["state"] == 3:
            node_color.append("xkcd:pumpkin orange")
    else:
        if n in nx.ancestors(co2s.graph, "pk_H2CO3"):
            node_color.append("xkcd:light blue")
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
    edge_color="xkcd:dark",
    alpha=0.8,
)
nx.draw_networkx_labels(
    graph,
    ax=ax,
    pos=pos,
    labels={n: pyco2.labels[n] for n in graph.nodes},
    font_size=8,
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("tests/_fig_graph_example.png")
