# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

co2s = pyco2.sys(salinity=35, temperature=3)
co2s.solve(["k_CO2", "k_H2CO3", "k_HCO3"], store_steps=2)

# Visualise
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
co2s.plot_graph(
    ax=ax,
    prog_graphviz="dot",
    show_unknown=False,
    show_isolated=False,
    # skip_nodes=["gas_constant", "ionic_strength"],
    mode="valid",
    node_kwargs={"node_size": 650, "alpha": 0.7},
    edge_kwargs={"node_size": 650, "alpha": 0.8},
    label_kwargs={"font_size": 8},
)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
# fig.savefig("figures/files/fig_valid.png")
