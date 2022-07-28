from autograd import numpy as np
from PyCO2SYS.solve import dom


# Test values
temperature_K = 298.15
pH = np.linspace(-2, 14, num=100)
pHstep = np.mean(np.diff(pH))
c_H = 10.0**-pH
c_OH = 1e-14 / c_H
c_Na = np.full_like(c_H, 1.0)
c_Ca = np.full_like(c_H, 0.1)
c_Cl = c_Na + c_H + 2 * c_Ca - c_OH
assert np.all(c_H + c_Na + 2 * c_Ca - c_OH - c_Cl == 0)
assert np.all(c_Cl >= 0)
c_ions = np.array([c_H, c_OH, c_Na, c_Cl, c_Ca]).transpose()
z_ions = np.array([1, -1, 1, -1, 2])
ionic_strength = 0.5 * np.sum(c_ions * z_ions**2, axis=1)

# Do calculations
nd = dom.nd_fulvic
log10_chi = dom.solve_chi(c_ions, z_ions, ionic_strength, nd)
chi = 10.0**log10_chi
psi = dom.chi_to_psi(chi, temperature_K - 273.15)
Q_H = dom.nica(c_H, chi, nd)

#%% Quick viz
from matplotlib import pyplot as plt

fix, axs = plt.subplots(nrows=3, ncols=2, dpi=300, figsize=(6, 7.5))

ax = axs[0, 0]
ax.plot(pH, Q_H, label="$\chi$ solved")
Q_H_constantchi = dom.nica(c_H, 10, dom.nd_fulvic)
ax.plot(pH, Q_H_constantchi, label="$\chi$ constant")
ax.set_xlabel("pH")
ax.set_ylabel("$Q_\mathrm{H}$")
ax.legend()

ax = axs[1, 0]
ax.plot(pH[1:], -np.diff(Q_H) / pHstep, label="$\chi$ solved")
ax.plot(pH[1:], -np.diff(Q_H_constantchi) / pHstep, label="$\chi$ constant")
ax.set_xlabel("pH")
ax.set_ylabel("d$Q_\mathrm{H}$ / dpH")
ax.legend()

ax = axs[0, 1]
ax.axhline(0, c="k", lw=0.8)
ax.plot(pH, dom.charge_balance(log10_chi, c_ions, z_ions, ionic_strength, nd))
ax.set_xlabel("pH")
ax.set_ylabel("Charge balance at best $\chi$")

ax = axs[1, 1]
l10 = np.linspace(-5, 5, num=100)
i = 62
ax.plot(
    l10, dom.charge_balance(l10, np.array([c_ions[i]]), z_ions, ionic_strength[i], nd)
)
ax.axhline(0, c="k", lw=0.8)
ax.set_ylim(np.array([-1, 1]) * 50)
ax.set_title("pH = {:.1f}".format(pH[i]))
ax.set_xlabel("log$_{10}$ $\chi$")
ax.set_ylabel("Charge balance")

ax = axs[2, 0]
ax.plot(pH, log10_chi)
ax.set_xlabel("pH")
ax.set_ylabel("log$_{10}$ $\chi$")

plt.tight_layout()
