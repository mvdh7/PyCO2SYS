# %%
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

# Fig 2a
field = pyco2.sys(
    ph=np.linspace(6, 9, num=200),
    pco2=np.vstack(np.linspace(1, 5000, num=200)),
    s=34.8,
    t=18.3,
).solve("revelle")
line = pyco2.sys(
    ta=2250,
    pco2=field.pco2.ravel(),
    s=34.8,
    t=18.3,
).solve("ph")
fig, ax = plt.subplots(dpi=300)
cf = ax.contourf(field.ph, field.pco2.ravel(), field.revelle)
plt.colorbar(cf)
ax.plot(line.ph, line.pco2, c="w")
ax.set_xlim(6, 9)

# %% Fig 2d
fig2d = pyco2.sys(
    ph=np.linspace(4, 9, num=100),
    ta=2250,
    s=34.8,
    t=18.3,
).solve(["revelle", "gamma_dic"])

fig2d_dic = pyco2.sys(
    dic=fig2d.dic,
    ta=fig2d.ta,
    s=fig2d.s,
    t=fig2d.t,
)
fig2d_dic.get_grads("CO2", "dic")

fig, ax = plt.subplots(dpi=300)
ax.plot(fig2d.ph, fig2d.revelle, c="xkcd:blue/green", label="RF")
ax.plot(
    fig2d.ph, 1 / fig2d_dic.grads["CO2"]["dic"], label="gCO2", c="xkcd:carolina blue"
)
ax.plot(4, 4, c="xkcd:goldenrod", label="gDIC")
ax.set_ylim(0, 25)
ax2 = ax.twinx()
ax2.plot(fig2d.ph, 1e3 * fig2d.gamma_dic, c="xkcd:goldenrod", label="gDIC")
ax2.set_ylim(0, 0.5)
ax.legend()
ax.axvline(7.5, c="xkcd:forest green", ls="--")

# %%
co2s = fig2d_dic
fig, ax = plt.subplots(dpi=300)
ax.plot(fig2d.ph, np.log10(fig2d_dic.grads["CO2"]["dic"]), label="1/gCO2")
ax.plot(fig2d.ph, np.log10(fig2d_dic["dic"]), label="DIC")
ax.plot(fig2d.ph, np.log10(1 / fig2d_dic["co2"]), label="1/CO2aq")
ax.plot(
    fig2d.ph,
    np.log10(fig2d_dic.grads["CO2"]["dic"] * fig2d["dic"] / fig2d["co2"]),
    label="RF",
)
ax.legend()
# ax.set_ylim([0, 20])
