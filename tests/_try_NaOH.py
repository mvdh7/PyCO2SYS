# %%
import numpy as np
from matplotlib import pyplot as plt

import PyCO2SYS as pyco2

add_NaOH = np.linspace(0, 3000, num=50)  # µmol/kg
co2s = pyco2.sys(
    dic=2050,
    alkalinity=2250 + add_NaOH,
)
co2s.solve(["pH", "CO3", "HCO3", "CO2"])

fig, ax = plt.subplots(dpi=300)
ax.plot(add_NaOH * 1e-3, 1e-3 * co2s.alkalinity, label="TA", c="#934D20")
ax.plot(add_NaOH * 1e-3, 1e-3 * co2s.CO3, label="[CO$_3^{2-}$]", c="#4F71BE")
ax.plot(add_NaOH * 1e-3, 1e-3 * co2s.HCO3, label="[HCO$_3^-$]", c="#DE8344")
ax.plot(add_NaOH * 1e-3, 1e-3 * co2s.CO2, label="[CO$_2$(aq)]", c="#A5A5A5")
ax.plot(0, 0, label="pH", c="#2D4374")
ax.grid(alpha=0.3)
ax.set_ylabel("$x$ / mmol kg$^{-1}$")
ax.legend()
ax2 = ax.twinx()
ax2.plot(add_NaOH * 1e-3, co2s.pH, c="#2D4374")
ax2.set_ylabel("pH")
ax.set_xlabel("Added NaOH / mmol kg$^{-1}$")
ax.set_title(
    f"Start: TA = {co2s.alkalinity[0]:.0f}, DIC = {co2s.dic:.0f} " + "µmol kg$^{-1}$"
)
