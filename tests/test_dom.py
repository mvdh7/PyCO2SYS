from autograd import numpy as np
from PyCO2SYS import salts
from PyCO2SYS.solve import dom
import gsw

# Test values
pH = np.linspace(-2, 14, num=100)
pHstep = np.mean(np.diff(pH))

# sw = {}
# sw["Hfree"] = 10.0**-pH
# sw["OH"] = 1e-14 / sw["Hfree"]
# sw["SO4"] = np.full_like(pH, 0.0)
# sw["HSO4"] = np.full_like(pH, 0.0)
# sw["F"] = np.full_like(pH, 0.0)
# sw["CO3"] = np.full_like(pH, 0.0)
# sw["HCO3"] = np.full_like(pH, 0.0)
# sw["BOH4"] = np.full_like(pH, 0.0)
# sw["H3SiO4"] = np.full_like(pH, 0.0)
# sw["H2PO4"] = np.full_like(pH, 0.0)
# sw["HPO4"] = np.full_like(pH, 0.0)
# sw["PO4"] = np.full_like(pH, 0.0)
# sw["NH4"] = np.full_like(pH, 0.0)
# sw["HS"] = np.full_like(pH, 0.0)

# salinity = np.full_like(pH, 0)
# temperature = 25
# pressure = 0
# rc = None

# c_ions, z_ions = dom.get_ions(sw, salinity, temperature, pressure)
# ionic_strength = dom.get_ionic_strength(c_ions, z_ions)

#%%

c_H = 10.0**-pH
c_OH = 1e-14 / c_H
c_Na = np.full_like(c_H, 1.0)
c_Ca = np.full_like(c_H, 0.1)
c_Cl = c_Na + c_H + 2 * c_Ca - c_OH
assert np.all(c_H + c_Na + 2 * c_Ca - c_OH - c_Cl == 0)
assert np.all(c_Cl >= 0)
c_ions = np.array([c_H, c_OH, c_Na, c_Cl, c_Ca]).transpose()
z_ions = np.array([1, -1, 1, -1, 2])
ionic_strength = dom.get_ionic_strength(c_ions, z_ions)

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
ax.plot(pH, Q_H_fulvic, label="Fulvic")
ax.plot(pH, Q_H_humic, label="Humic")
# Q_H_constantchi = dom.nica(c_H, 10, dom.nd_fulvic)
# ax.plot(pH, Q_H_constantchi, label="$\chi$ constant")
ax.set_xlabel("pH")
ax.set_ylabel("$Q_\mathrm{H}$")
ax.legend()

ax = axs[1, 0]
ax.plot(pH[1:], -np.diff(Q_H_fulvic) / pHstep, label="Fulvic")
ax.plot(pH[1:], -np.diff(Q_H_humic) / pHstep, label="Humic")
# ax.plot(pH[1:], -np.diff(Q_H_constantchi) / pHstep, label="$\chi$ constant")
ax.set_xlabel("pH")
ax.set_ylabel("d$Q_\mathrm{H}$ / dpH")
ax.legend()

ax = axs[0, 1]
ax.axhline(0, c="k", lw=0.8)
ax.plot(
    pH,
    dom.charge_balance(log10_chi_fulvic, c_ions, z_ions, ionic_strength, dom.nd_fulvic),
)
ax.plot(
    pH,
    dom.charge_balance(log10_chi_humic, c_ions, z_ions, ionic_strength, dom.nd_humic),
)
ax.set_xlabel("pH")
ax.set_ylabel("Charge balance at best $\chi$")

ax = axs[1, 1]
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
ax.plot(pH, log10_chi_fulvic)
ax.plot(pH, log10_chi_humic)
ax.set_xlabel("pH")
ax.set_ylabel("log$_{10}$ $\chi$")

plt.tight_layout()