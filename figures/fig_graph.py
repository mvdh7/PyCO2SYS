# %%
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

co2s = pyco2.sys(
    dic=2100,
    pH=8.1,
)
co2s.solve(["CO3"], store_steps=1)

fig, ax = plt.subplots(dpi=300, figsize=(6, 5))
node_size = 2100
co2s.plot_graph(
    ax=ax,
    show_unknown=False,
    keep_unknown="HCO3",
    show_isolated=False,
    show_tsp=True,
    prog_graphviz="dot",
    skip_nodes=["gas_constant", "pressure"],
    node_kwargs={"alpha": 0.6, "node_size": node_size},
    edge_kwargs={"arrowstyle": "-|>", "alpha": 1, "node_size": node_size},
)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig("figures/files/fig_graph.png")

# %%
co2s = pyco2.sys(
    dic=2100,
    pH=8.1,
)
co2s.solve(["CO3"], store_steps=2)

fig, ax = plt.subplots(dpi=300, figsize=(10, 6))
node_size = 700
co2s.plot_graph(
    ax=ax,
    show_unknown=False,
    keep_unknown="HCO3",
    show_isolated=False,
    show_tsp=True,
    prog_graphviz="dot",
    node_kwargs={"alpha": 0.6, "node_size": node_size},
    edge_kwargs={"arrowstyle": "-|>", "alpha": 1, "node_size": node_size},
    label_kwargs={"font_size": 7},
)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig("figures/files/fig_graph_complete.png")

# %%
co2s = pyco2.sys(
    alkalinity=2100,
    pH=8.1,
)
co2s.solve(["CO3"], store_steps=2)

fig, ax = plt.subplots(dpi=300, figsize=(20, 6))
node_size = 400
co2s.plot_graph(
    ax=ax,
    show_unknown=False,
    keep_unknown="HCO3",
    show_isolated=False,
    show_tsp=True,
    prog_graphviz="dot",
    node_kwargs={"alpha": 0.6, "node_size": node_size},
    edge_kwargs={"arrowstyle": "-|>", "alpha": 0.5, "node_size": node_size},
    label_kwargs={"font_size": 7},
)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig("figures/files/fig_graph_complete_alk.png")
