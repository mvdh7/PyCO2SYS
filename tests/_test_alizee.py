# %%
import glodap
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

gatl = glodap.atlantic().rename(columns={"fco2": "fco2_20"})

# %%
co2s = pyco2.sys(data=gatl, nitrite=0, t=20, p=0).solve("fco2")
gatl["fco2_co2s"] = co2s.fco2
gatl["fco2_diff"] = gatl.fco2_co2s - gatl.fco2_20

# %%
fig, ax = plt.subplots(dpi=300)
ax.scatter("fco2_20", "fco2_diff", data=gatl, s=5, alpha=0.3, edgecolor="none")
ax.set_ylim(-150, 150)
