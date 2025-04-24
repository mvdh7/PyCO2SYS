# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

import PyCO2SYS as pyco2

glodap_raw = {}
glodap_regions = {}
regions = ["Atlantic", "Pacific", "Indian"]
for region in regions:
    glodap_raw[region] = loadmat(
        f"/Users/matthew/Documents/data/GLODAP/GLODAPv2.2023_{region}_Ocean.mat"
    )
    glodap_regions[region] = {
        k[2:]: np.array(v).ravel()
        for k, v in glodap_raw[region].items()
        if k.startswith("G2")
    }
    L = glodap_regions[region]["depth"] <= 20
    glodap_regions[region] = {k: list(v[L]) for k, v in glodap_regions[region].items()}
# %%
glodap = {}
for k in glodap_regions[regions[0]]:
    glodap[k] = []
    for region in regions:
        glodap[k] += glodap_regions[region][k]
    try:
        glodap[k] = np.array(glodap[k])
    except ValueError:
        glodap.pop(k)

# %%
omega = pyco2.sys(
    data=glodap,
    dic="tco2",
    alkalinity="talk",
).solve("saturation_calcite")

kws = dict(s=2, c="xkcd:dark", alpha=0.1)
fig, axs = plt.subplots(dpi=300, ncols=3, nrows=2, figsize=(8, 5))
ax = axs[0, 0]
ax.scatter("temperature", "saturation_calcite", data=omega, **kws)
ax.set_xlabel("SST / °C")
ax = axs[0, 1]
ax.scatter("salinity", "saturation_calcite", data=omega, **kws)
ax.set_xlabel("SSS")
ax.set_xlim(32, 37)
ax = axs[0, 2]
ax.scatter(glodap["chla"], omega.saturation_calcite, **kws)
ax.set_xscale("log")
ax.set_xlabel("Chlorophyll $a$")
ax = axs[1, 0]
ax.scatter(glodap["temperature"] / glodap["chla"], omega.saturation_calcite, **kws)
ax.set_xlabel("R1 = temperature / chl-a")
ax.set_xlim(0, 250)
ax = axs[1, 1]
ax.scatter(glodap["salinity"] / glodap["chla"], omega.saturation_calcite, **kws)
ax.set_xlabel("R2 = salinity / chl-a")
ax.set_xlim(0, 400)
for ax in axs.ravel():
    ax.set_ylabel("Ω(calcite)")
    ax.grid(alpha=0.2)
ax = axs[1, 2]
ax.scatter(
    omega.saturation_calcite,
    0.01 * glodap["temperature"] / glodap["chla"]
    - 0.006 * glodap["salinity"] / glodap["chla"]
    + 4.676,
    **kws,
)
ax.set_xlabel("Ω GLODAP")
ax.set_ylabel("Ω eq. 13")
ax.axline((0, 0), slope=1, c="k")
ax.set_xlim(1, 8)
ax.set_ylim(1, 8)
fig.tight_layout()
fig.savefig("figures/files/_fig_pr.png")
