# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

co2s = pyco2.sys(dic=2200, alkalinity=2300)
co2s.solve(["pH"], store_steps=2)

# Visualise
fig, ax = plt.subplots(figsize=(12, 12))
co2s.plot_graph(
    ax=ax,
    prog_graphviz="sfdp",
    show_unknown=True,
    show_isolated=True,
    # skip_nodes=["gas_constant", "pressure_atmosphere"],
    mode="valid",
    node_kwargs={"node_size": 600},
    edge_kwargs={"node_size": 600},
    label_kwargs={"font_size": 9},
)
