# %%
import networkx as nx
import numpy as np
import pandas as pd

import PyCO2SYS as pyco2
from PyCO2SYS import convert

co2s = pyco2.sys(
    data=pd.DataFrame(
        {
            "dich": [2100, 2200],
            "pHx": 8.1,
        }
    ),
    dic="dich",
    pH="pHx",
)
co2s.solve(["CO3"])
print(co2s.CO3)

# %%
co2s = pyco2.sys(
    dic=2100,
    pH=8.1,
)
co2s.solve(["CO3"], store_steps=1)
node_size = 2100
co2s.plot_graph(
    show_unknown=False,
    keep_unknown="HCO3",
    show_isolated=False,
    show_tsp=True,
    prog_graphviz="dot",
    # root_graphviz="dic",
    # args_graphviz="-Granksep=1.0 -Gnodesep=1.0",
    # nx_layout=nx.arf_layout,
    # nx_args=("salinity",),
    skip_nodes=["gas_constant", "pressure"],
    node_kwargs={"alpha": 0.7, "node_size": node_size},
    edge_kwargs={"arrowstyle": "-|>", "alpha": 1, "node_size": node_size},
)
# (
#     osage,
#     gc,
#     sccmap,
#     gvpr,
#     ccomps,
#     sfdp,
#     patchwork,
#     twopi,
#     nop,
#     circo,
#     dot,
#     gvcolor,
#     acyclic,
#     neato,
#     unflatten,
#     tred,
#     fdp,
# )

# %%

# %% Draw the graph
nx.draw(
    co2s.graph,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=500,
    font_size=10,
)
# plt.show()

# %%

temperature = np.arange(10, 20)
salinity = np.vstack(np.arange(31, 35))

co2s = pyco2.sys(
    alkalinity=2250,
    dic=2100,
    temperature=temperature,
    salinity=salinity,
)
co2s.solve("pH")

# %%
co2s = pyco2.sys(
    alkalinity=np.arange(2250, 2260),
    pCO2=400,
    temperature=np.arange(10, 20),
    salinity=np.arange(30, 40),
    pressure=10,
    total_silicate=100,
)
co2s.solve("pH", store_steps=0)

# TODO what if pH and pCO2 were measured discretely but at different temperatures
# from each other?!?!?!??!! --- you need to first convert pCO2 to the pH temperature?

co2adj = co2s.adjust(temperature=30, pressure=1000)

# %% One parameter
co2s = pyco2.sys(pH=8.1)
co2s.solve()

# %%
co2s = pyco2.sys(pH=8.1)
co2s["pH_free"]
co2s.solve(["pH_total", "pH_free"])

# %%
co2s = pyco2.sys(
    fCO2=400,
    temperature=10,
)
co2sa = co2s.adjust(temperature=25)

# Uncertainties
co2s = pyco2.sys(
    alkalinity=np.arange(2250, 2260),
    pCO2=400,
    temperature=np.arange(10, 20),
    salinity=np.arange(30, 40),
    pressure=10,
    total_silicate=100,
    opt_k_carbonic=10,
)
co2s.propagate("pH", {"alkalinity": 2, "pCO2": 1})

# %%
co2s.plot_graph(show_unknown=False)
