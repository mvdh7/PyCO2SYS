# %%
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

co2s = pyco2.sys(
    ph=np.linspace(6, 9),
    pco2=np.vstack(np.linspace(1, 5000)),
    s=34.8,
    t=18.3,
).solve("revelle")
co2s2 = pyco2.sys(
    ph=co2s.ph,
    ta=2250,
    s=34.8,
    t=18.3,
).solve(["pco2", "co2", "hco3", "co3"])
co2s3 = pyco2.sys(
    pco2=np.linspace(1, 3000, num=3000),
    ta=2250,
    s=34.8,
    t=18.3,
).solve(["revelle", "dic"])
co2s3.get_grads(["CO2", "HCO3", "CO3", "dic"], "pCO2")

fig, ax = plt.subplots(dpi=300)
cf = ax.contourf(co2s.ph, co2s.pco2.ravel(), co2s.revelle)
ax.plot(co2s2.ph, co2s2.pco2, c="w")
ax.set_ylim(0, 5000)
plt.colorbar(cf)

fig, ax = plt.subplots(dpi=300)
ax.plot(co2s2.pco2, co2s2.co2, label="CO2")
ax.plot(co2s2.pco2, co2s2.hco3, label="HCO3")
ax.plot(co2s2.pco2, co2s2.co3, label="CO3")
ax.set_xlim(0, 3000)
ax.legend()

# %%
fig, ax = plt.subplots(dpi=300)
ax.plot(co2s3.pco2, co2s3.grads["CO2"]["pCO2"], label="CO2")
ax.plot(co2s3.pco2, co2s3.grads["HCO3"]["pCO2"], label="HCO3")
ax.plot(co2s3.pco2, -co2s3.grads["CO3"]["pCO2"], label="-CO3")
ax.plot(co2s3.pco2, co2s3.grads["dic"]["pCO2"], label="DIC")
# ax.plot(
#     co2s3.pco2,
#     co2s3.grads["CO2"]["pCO2"]
#     + co2s3.grads["HCO3"]["pCO2"]
#     + co2s3.grads["CO3"]["pCO2"],
#     label="Total",
#     c="k",
#     ls=":",
# )
ax.axhline(0, c="k", lw=0.8)
# ax.set_xlim(0, 500)
ax.set_ylim(0, 0.2)
ax.legend()
# ax2 = ax.twinx()
# # ax2.plot(co2s3.pco2, co2s3.revelle, c="k")
# ax.plot(co2s3.pco2, co2s3.pco2 / co2s3.dic, c="pink")
# ax2.plot(
#     co2s3.pco2,
#     co2s3.grads["dic"]["pCO2"] * co2s3.pco2 / co2s3.dic,
#     c="k",
# )
ax.grid(alpha=0.2)

# %%
fig, ax = plt.subplots(dpi=300)
# ax.plot(co2s3.pco2, co2s3.dic / (co2s3.pco2 * co2s3.grads["dic"]["pCO2"]))
ax.plot(co2s3.pco2, co2s3.dic, label="dic")
ax.plot(co2s3.pco2, 1 / co2s3.pco2, label="1/pco2")
ax.plot(co2s3.pco2, 1 / co2s3.grads["dic"]["pCO2"], label="1/grad")
ax.legend()
ax.set_yscale("log")

# %%
fig, ax = plt.subplots(dpi=300)
ax.plot(co2s3.pco2, np.log10(1 / co2s3.grads["dic"]["pCO2"]), label="dpCO2/dDIC")
# ax.plot(co2s3.pco2, co2s3.dic / co2s3.pco2, label="DIC/pCO2")
ax.plot(co2s3.pco2, np.log10(co2s3.dic), label="DIC")
ax.plot(co2s3.pco2, -np.log10(co2s3.pco2), label="pCO2")
ax.plot(co2s3.pco2, np.log10(co2s3.revelle), label="Revelle", c="k")
ax.plot(
    co2s3.pco2,
    np.log10(co2s3.revelle) - np.log10(co2s3.dic),
    label="Egleston",
    c="k",
    ls="-.",
)
# ax.set_ylim(0, 20)
ax.legend()
ax.grid(alpha=0.2)
# ax.set_xscale("log")

# %%
fig, ax = plt.subplots(dpi=300)
ax.plot(
    # co2s3.grads["CO3"]["pCO2"] - co2s3.grads["CO2"]["pCO2"],
    co2s3.pco2,
    co2s3.co3,
    # co2s3.co3 - co2s3.co2,
    # co2s3.revelle,
    label="CO3",
)
ax.plot(co2s3.pco2, co2s3.co2, label="CO2")
ax.plot(co2s3.pco2, co2s3.hco3, label="HCO3")
ax.plot(co2s3.pco2, co2s3.dic, c="k", label="DIC")
ax2 = ax.twinx()
# ax2.plot(co2s3.pco2, co2s3.revelle, c="r", label="Revelle")
ax.grid(alpha=0.2)
ax.legend()

# %%
co2s = pyco2.sys(
    pco2=600 * np.array([1, 1.01]), alkalinity=2250, s=34.8, t=18.3
).solve()
print(co2s.revelle)
print(co2s.dic, np.diff(co2s.dic), 1 / (100 * np.diff(co2s.dic) / co2s.dic[0]))
print(co2s.co3, np.diff(co2s.co3))
print(co2s.hco3, np.diff(co2s.hco3))
print(co2s.co2, np.diff(co2s.co2))
