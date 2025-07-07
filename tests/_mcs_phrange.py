# %%
import numpy as np

import PyCO2SYS as pyco2

co2s_hi = pyco2.sys(
    # pH=[8.2, 8.1],
    pH=8.15,
    alkalinity=2400,
    temperature=[11, 21],
)
co2s_lo = pyco2.sys(
    # pH=[7.9, 7.8],
    pH=7.85,
    alkalinity=2400,
    temperature=[11, 21],
)
print(co2s_hi["H"] * 1e9)
print(co2s_lo["H"] * 1e9)

print(co2s_hi["co3"], np.diff(co2s_hi["co3"]))
print(co2s_lo["co3"], np.diff(co2s_lo["co3"]))

print(co2s_hi["oa"], np.diff(co2s_hi["oa"]))
print(co2s_lo["oa"], np.diff(co2s_lo["oa"]))
