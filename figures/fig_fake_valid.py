# %%
import networkx as nx
from matplotlib import pyplot as plt

valid_colours = {
    -1: "xkcd:light red",  # invalid
    0: "xkcd:light grey",  # unknown
    1: "xkcd:sky blue",  # valid
}
graph = nx.DiGraph()
graph.add_edges_from(
    [
        ["$A_1$", "$B_1$"],
        ["$A_3$", "$B_1$"],
        ["$A_2$", "$B_1$"],
        ["$A_3$", "$B_2$"],
        ["$A_2$", "$B_2$"],
        ["$A_1$", "$C_1$"],
        ["$B_1$", "$C_1$"],
        ["$B_1$", "$C_2$"],
        ["$B_2$", "$C_2$"],
    ]
)
nx.set_node_attributes(
    graph,
    {
        "$B_1$": 1,
        "$B_2$": -1,
    },
    name="valid",
)
nx.set_node_attributes(
    graph,
    {
        "$C_2$": -1,
    },
    name="valid_p",
)
nx.set_edge_attributes(
    graph,
    {
        ("$A_2$", "$B_1$"): 1,
        ("$A_3$", "$B_1$"): 1,
        ("$A_2$", "$B_2$"): 1,
        ("$A_3$", "$B_2$"): -1,
    },
    name="valid",
)
node_valid = nx.get_node_attributes(graph, "valid", default=0)
node_valid_p = nx.get_node_attributes(graph, "valid_p", default=0)
edge_valid = nx.get_edge_attributes(graph, "valid", default=0)
node_color = [valid_colours[node_valid[n]] for n in nx.nodes(graph)]
edge_color = [valid_colours[edge_valid[e]] for e in nx.edges(graph)]
node_edgecolors = [valid_colours[node_valid_p[n]] for n in nx.nodes(graph)]
node_linewidths = [[0, 2][node_valid_p[n]] for n in nx.nodes(graph)]
node_size = 500
pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
fig, ax = plt.subplots(dpi=300)
nx.draw_networkx_nodes(
    graph,
    pos=pos,
    node_color=node_color,
    edgecolors=node_edgecolors,
    node_size=node_size,
    linewidths=node_linewidths,
)
nx.draw_networkx_edges(graph, pos=pos, edge_color=edge_color, node_size=node_size)
nx.draw_networkx_labels(graph, pos=pos)
