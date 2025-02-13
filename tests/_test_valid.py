# %%
import networkx as nx
import numpy as np

import PyCO2SYS as pyco2

co2s = pyco2.sys(salinity=42)
co2s.solve(["k_HSO4_free", "k_CO2", "k_H2CO3"], store_steps=2)

# Visualise
co2s.plot_graph(
    prog_graphviz="dot",
    show_unknown=False,
    show_isolated=False,
    mode="valid",
    node_kwargs={"node_size": 700},
    edge_kwargs={"node_size": 700},
    label_kwargs={"font_size": 9},
)
