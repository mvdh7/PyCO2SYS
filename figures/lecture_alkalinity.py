# %%
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

# Set initial conditions
pure = pyco2.sys(
    dic=0,
    pH=8.05,
    total_sulfate=0,
    total_borate=0,
    total_fluoride=0,
    opt_k_carbonic=8,
)
seawater = pyco2.sys(
    fCO2=400,
    pH=8.05,
    opt_k_carbonic=10,
)

# Do titrations
acid = np.linspace(0, 3500, num=1000)
pure_t = pyco2.sys(
    dic=0,
    alkalinity=pure["alkalinity"] - acid,
    total_sulfate=0,
    total_borate=0,
    total_fluoride=0,
    opt_k_carbonic=8,
)
seawater_t = pyco2.sys(
    dic=seawater["dic"],
    alkalinity=seawater["alkalinity"] - acid,
    opt_k_carbonic=10,
)

# Visualise
fig, axs = plt.subplots(dpi=300, ncols=2, figsize=(10, 4))
ax = axs[1]
ax.plot(
    acid / 1000,
    pure_t["pH"],
    label="Pure water",
    c="xkcd:light blue",
    lw=2,
    clip_on=False,
)
ax.plot(
    acid / 1000,
    seawater_t["pH"],
    label="Seawater",
    c="xkcd:sea blue",
    lw=2,
    clip_on=False,
)
# ax.axhline(4.5, c="xkcd:dark", ls=":")
ax.set_ylabel("pH = –log$_{10}$ [H$^+$]")
ax = axs[0]
ax.plot(
    acid / 1000,
    1e3 * pure_t["H"],
    label="Pure water",
    c="xkcd:light blue",
    lw=2,
    clip_on=False,
)
ax.plot(
    acid / 1000,
    1e3 * seawater_t["H"],
    label="Seawater",
    c="xkcd:sea blue",
    lw=2,
    clip_on=False,
)
# ax.axhline(1e3 * 10**-4.5, c="xkcd:dark", ls=":")
ax.set_ylabel("[H$^+$]")
for ax in axs:
    ax.axvline(
        seawater.alkalinity * 1e-3,
        c="xkcd:sea blue",
        label="Alkalinity",
        ls="--",
    )
    ax.set_xlabel("Acid (HCl) added / mmol kg$^{-1}$")
    ax.grid(alpha=0.2)
axs[0].legend()
fig.tight_layout()
fig.savefig("figures/files/lecture_alkalinity.png")
