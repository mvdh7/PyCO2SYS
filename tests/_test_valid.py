# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

co2s = pyco2.sys(dic=2200, alkalinity=2300, salinity=30, temperature=25)
co2s.solve(["k_HCO3_sws_1atm"], store_steps=2)
# co2s.solve(["pH"], store_steps=2)
co2s.check_valid(
    # ignore="total_borate",
)

# Visualise
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
co2s.plot_graph(
    ax=ax,
    # prog_graphviz="sfdp",
    prog_graphviz="dot",
    show_unknown=False,
    show_isolated=False,
    # skip_nodes=["gas_constant", "pressure_atmosphere"],
    mode="valid",
    node_kwargs={"node_size": 1500},
    edge_kwargs={"node_size": 1500},
    label_kwargs={"font_size": 12},
)
