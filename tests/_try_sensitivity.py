# %%
import networkx as nx
import numpy as np
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2
from PyCO2SYS.engine import set_node_labels

svar = "pCO2"
co2s = pyco2.sys(dic=2100, pH=8.1, pressure=1000)
co2s.solve(svar, store_steps=2)

uncertainties = {
    "salinity": 0.01,
    "temperature": 0.1,
    "pressure": 1,
    "dic": 2,
    "pH": 0.001,
}
co2s.propagate([svar, *nx.ancestors(co2s.graph, svar)], uncertainties)


unorm = {
    u_into: {u_from: u / into_dict["total"] for u_from, u in into_dict.items()}
    for u_into, into_dict in co2s.uncertainty.items()
}

fig, ax = plt.subplots(dpi=300)
graph_to_plot = co2s.get_graph_to_plot(
    show_isolated=False,
    show_unknown=False,
)
pos = co2s.get_graph_pos(
    graph_to_plot=graph_to_plot,
    prog_graphviz="dot",
)
pos = {k: np.array(v) / np.array([200, 100]) for k, v in pos.items()}
# for e in graph_to_plot.edges:
#     ax.plot(*np.array([pos[e[0]], pos[e[1]]]).T)
nx.draw_networkx_edges(
    graph_to_plot,
    pos,
    ax=ax,
    edge_color="xkcd:grey",
    node_size=200,
)
for p, xy in pos.items():
    if co2s.uncertainty[p]["total"] > 0:
        x = np.array([co2s.uncertainty[p][f] for f in uncertainties])
        center = xy
        startangle = 0
        counterclock = True
        # expl = 0
        radius = 0.25
        colors = [
            "xkcd:light green",
            "xkcd:light red",
            "xkcd:light blue",
            "xkcd:light purple",
            "xkcd:light orange",
        ]

        sx = x.sum()
        x = x / sx

        theta1 = startangle / 360

        for frac, color in zip(x, colors):
            x, y = center
            theta2 = (theta1 + frac) if counterclock else (theta1 - frac)
            thetam = 2 * np.pi * 0.5 * (theta1 + theta2)
            # x += expl * np.cos(thetam)
            # y += expl * np.sin(thetam)
            w = mpatches.Wedge(
                (x, y),
                radius,
                360.0 * min(theta1, theta2),
                360.0 * max(theta1, theta2),
                facecolor=color,
                # hatch=next(hatch_cycle),
                clip_on=False,
            )
            ax.add_patch(w)

            theta1 = theta2

        # ax.pie(
        #     [co2s.uncertainty[p][f] for f in uncertainties],
        #     center=xy,
        #     colors=["xkcd:green", "xkcd:red"],
        #     radius=0.2,
        # )
        # ax.text(
        #     *xy,
        #     set_node_labels[p],
        #     ha="center",
        #     va="center",
        # )

# nx.draw_networkx_nodes(graph_to_plot, pos, node_size=500)
nx.draw_networkx_labels(
    graph_to_plot,
    pos,
    ax=ax,
    labels={k: v for k, v in set_node_labels.items() if k in graph_to_plot.nodes},
)
ax.set_aspect(1)
# xl = ax.get_xlim()
# yl = ax.get_ylim()

# ax.set_xlim(xl)
# ax.set_ylim(-0.1, 1.9)
# fig.tight_layout()
