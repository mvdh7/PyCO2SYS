import numpy as np
from matplotlib import pyplot as plt
import PyCO2SYS as pyco2

# Define input conditions
dic = np.linspace(1600, 3200, 1601)
ta = 2250
tmp = 25
sal = 35
prs = 0
si = 0
phos = 0

# Run CO2SYS
cdict = pyco2.sys(
    ta,
    dic,
    1,
    2,
    salinity=sal,
    temperature=tmp,
    pressure=prs,
    total_silicate=si,
    total_phosphate=phos,
    opt_pH_scale=1,
    opt_k_carbonic=10,  # not sure which options were used by ESM10...
    opt_k_fluoride=1,
    opt_total_borate=2,
    opt_buffers_mode=1,
)

# Recreate ESM10 Fig. 2
fvars = ["gamma_dic", "gamma_alk", "beta_dic", "beta_alk", "omega_dic", "omega_alk"]
fmults = [1, -1, 1, -1, -1, 1]
fclrs = ["#453c90", "#b84690", "#b84690", "#007831", "#df0023", "#40b4b6"]
flabels = [
    "$\\gamma_\mathrm{DIC}$",
    "$-\\gamma_\mathrm{Alk}$",
    "$\\beta_\mathrm{DIC}$",
    "$-\\beta_\mathrm{Alk}$",
    "$-\\omega_\mathrm{DIC}$",
    "$\\omega_\mathrm{Alk}$",
]
fig, ax = plt.subplots(dpi=300)
for i, fvar in enumerate(fvars):
    ax.plot(dic * 1e-3, cdict[fvar] * fmults[i] * 1e3, c=fclrs[i], label=flabels[i])
ax.legend(edgecolor="k")
ax.set_xlim([1.6, 3.2])
ax.set_ylim([0.1, 1.0])
ax.set_title("Buffer factors at Alk = {} mM".format(ta * 1e-3))
ax.set_xlabel("DIC (mM)")
ax.set_ylabel("buffer factor (mM)")
ax.grid(alpha=0.4)
plt.savefig("validate/figures/buffers_ESM10.png")
