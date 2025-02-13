# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

co2s = pyco2.sys(dic=2200, pCO2=400, salinity=35, temperature=30)
co2s.solve(["k_CO2", "k_H2CO3", "k_HCO3"], store_steps=2)
# co2s.solve(["pH"], store_steps=2)
co2s.check_valid(
    # ignore="total_borate",
)

# Visualise
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
co2s.plot_graph(
    ax=ax,
    # prog_graphviz="sfdp",
    prog_graphviz="dot",
    show_unknown=False,
    show_isolated=False,
    skip_nodes=["gas_constant", "ionic_strength"],
    mode="valid",
    node_kwargs={"node_size": 650, "alpha": 0.8},
    edge_kwargs={"node_size": 650},
    label_kwargs={"font_size": 8},
)
