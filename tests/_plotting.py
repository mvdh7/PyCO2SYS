import networkx as nx
from matplotlib import pyplot as plt


def plot_graph(fu):
    pos = nx.nx_agraph.graphviz_layout(fu.graph, prog="dot")
    fig, ax = plt.subplots()
    nx.draw_networkx(
        fu.graph,
        pos=pos,
        nodelist=fu.graph.nodes,
        node_color=[
            nx.get_node_attributes(fu.graph, "state", default=-1)[n]
            for n in fu.graph.nodes
        ],
        vmin=-1,
        vmax=3,
    )
