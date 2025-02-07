# %%
import numpy as np

import PyCO2SYS as pyco2
from PyCO2SYS import convert

co2s = pyco2.sys(
    dic=2100,
    pH=8.1,
)
co2s.solve(["fCO2", "alkalinity"])
co2s.plot_graph(
    show_unknown=False,
    show_isolated=False,
    show_tsp=False,
    prog_graphviz="dot",
)

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
