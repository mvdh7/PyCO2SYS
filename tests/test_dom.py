from autograd import numpy as np
import PyCO2SYS as pyco2
from PyCO2SYS.solve import dom

# Test values
pH = np.linspace(-2, 14, num=100)
pHstep = np.mean(np.diff(pH))
# pH = 8.0#np.array([8.0])

sw = {}
sw["Hfree"] = 10.0**-pH
sw["OH"] = 1e-14 / sw["Hfree"]
sw["SO4"] = np.full_like(pH, 0.0)
sw["HSO4"] = np.full_like(pH, 0.0)
sw["F"] = np.full_like(pH, 0.0)
sw["CO3"] = np.full_like(pH, 0.0)
sw["HCO3"] = np.full_like(pH, 0.0)
sw["BOH4"] = np.full_like(pH, 0.0)
sw["H3SiO4"] = np.full_like(pH, 0.0)
sw["H2PO4"] = np.full_like(pH, 0.0)
sw["HPO4"] = np.full_like(pH, 0.0)
sw["PO4"] = np.full_like(pH, 0.0)
sw["NH4"] = np.full_like(pH, 0.0)
sw["HS"] = np.full_like(pH, 0.0)

salinity = np.full_like(pH, 35.0)
temperature = 25
pressure = 0
rc = None

c_ions, z_ions = dom.get_ions(sw, salinity, temperature, pressure)
ionic_strength = dom.get_ionic_strength(c_ions, z_ions)
c_H = 10**-pH

# Do calculations - fulvic
log10_chi_fulvic = dom.solve_chi(c_ions, z_ions, ionic_strength, dom.nd_fulvic)
chi_fulvic = 10.0**log10_chi_fulvic
Q_H_fulvic = dom.nica(c_H, chi_fulvic, dom.nd_fulvic)
# Do calculations - humic
log10_chi_humic = dom.solve_chi(c_ions, z_ions, ionic_strength, dom.nd_humic)
chi_humic = 10.0**log10_chi_humic
Q_H_humic = dom.nica(c_H, chi_humic, dom.nd_humic)

#%% Quick viz
from matplotlib import pyplot as plt

fix, axs = plt.subplots(nrows=3, ncols=2, dpi=300, figsize=(6, 7.5))

ax = axs[0, 0]
ax.text(0, 1.05, "(a)", transform=ax.transAxes)
ax.plot(pH, Q_H_fulvic, label="Fulvic")
ax.plot(pH, Q_H_humic, label="Humic")
# Q_H_constantchi = dom.nica(c_H, 10, dom.nd_fulvic)
# ax.plot(pH, Q_H_constantchi, label="$\chi$ constant")
ax.set_xlabel("pH")
ax.set_ylabel("$Q_\mathrm{H}$")
ax.legend()

ax = axs[1, 0]
ax.text(0, 1.05, "(b)", transform=ax.transAxes)
ax.plot(pH[1:], -np.diff(Q_H_fulvic) / pHstep, label="Fulvic")
ax.plot(pH[1:], -np.diff(Q_H_humic) / pHstep, label="Humic")
# ax.plot(pH[1:], -np.diff(Q_H_constantchi) / pHstep, label="$\chi$ constant")
ax.set_xlabel("pH")
ax.set_ylabel("d$Q_\mathrm{H}$ / dpH")
# ax.legend()

ax = axs[0, 1]
ax.text(0, 1.05, "(d)", transform=ax.transAxes)
ax.axhline(0, c="k", lw=0.8)
ax.plot(
    pH,
    dom.charge_balance(log10_chi_fulvic, c_ions, z_ions, ionic_strength, dom.nd_fulvic)
    * 1e15,
)
ax.plot(
    pH,
    dom.charge_balance(log10_chi_humic, c_ions, z_ions, ionic_strength, dom.nd_humic)
    * 1e15,
)
ax.set_xlabel("pH")
ax.set_ylabel("Charge balance at\nfitted $\chi$ × 10$^{15}$")

ax = axs[1, 1]
ax.text(0, 1.05, "(e)", transform=ax.transAxes)
l10 = np.linspace(-5, 5, num=100)
i = 62
ax.plot(
    l10,
    dom.charge_balance(
        l10, np.array([c_ions[i]]), z_ions, ionic_strength[i], dom.nd_fulvic
    ),
)
ax.plot(
    l10,
    dom.charge_balance(
        l10, np.array([c_ions[i]]), z_ions, ionic_strength[i], dom.nd_humic
    ),
)
ax.axhline(0, c="k", lw=0.8)
ax.set_ylim(np.array([-1, 1]) * 50)
ax.set_title("pH = {:.1f}".format(pH[i]))
ax.set_xlabel("log$_{10}$ $\chi$")
ax.set_ylabel("Charge balance")

ax = axs[2, 0]
ax.text(0, 1.05, "(c)", transform=ax.transAxes)
ax.plot(pH, 10.0**log10_chi_fulvic)
ax.plot(pH, 10.0**log10_chi_humic)
ax.set_xlabel("pH")
ax.set_ylabel("$\chi$")

ax = axs[2, 1]
ax.text(0, 1.05, "(f)", transform=ax.transAxes)
ax.plot(pH, -np.log10(chi_fulvic * c_H) - pH)
ax.plot(pH, -np.log10(chi_humic * c_H) - pH)
# ax.axline((0, 0), slope=1, c="k", lw=0.8)
ax.set_xlabel("pH")
ax.set_ylabel("p($\chi c_\mathrm{H}$) $-$ pH")

plt.tight_layout()
plt.savefig("tests/test_dom.png")

#%%
r = pyco2.sys(par1=2300, par2=2100, par1_type=1, par2_type=2, salinity=35)
print(r["pH"])
