# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import PyCO2SYS as pyco2


# from PyCO2SYS.equilibria.p1atm import pk_HCO3_total_MMB25


def pk_HCO3_total_MMB25(temperature, salinity):
    """Carbonic acid dissociation constants with K2 following MMB25.
    K1 should come from WMW14.
    Used when opt_k_carbonic = 19.
    """
    TempK = temperature + 273.15
    Sal = salinity
    pK2 = (
        5.1703
        + 2136.77 / TempK
        - 177788 / TempK**2
        - 0.4457 * np.sqrt(Sal) / (1 + 1.11 * np.sqrt(Sal))
        + 0.0674 * Sal / np.log(TempK)
        - 0.0008238 * np.sqrt(Sal) * TempK
    )
    return pK2


cv = pd.read_csv("tests/data/check_values_pK2_MMB25.csv")
co2s = pyco2.sys(
    s=cv.Salinity.values,
    t=cv.Temp_C.values,
    opt_k_carbonic=19,
).solve("pk_HCO3_total_1atm")

pk2_direct = pk_HCO3_total_MMB25(cv.Temp_C.values, cv.Salinity.values)

pk2_diff = pk2_direct - cv.param_pK2

fig, ax = plt.subplots()
# ax.scatter(cv.param_pK2, co2s.pk_HCO3_total_1atm - cv.param_pK2)
ax.scatter(cv.Salinity, pk2_direct - cv.param_pK2)
ax.scatter(cv.Salinity.iloc[-1], pk2_direct[-1] - cv.param_pK2.iloc[-1])

# def test_MMB25():


def fitting_function(t_s, a, b, c, d, e, f):
    """Carbonic acid dissociation constants with K2 following MMB25.
    K1 should come from WMW14.
    Used when opt_k_carbonic = 19.
    """
    temperature, salinity = t_s
    TempK = temperature + 273.15
    Sal = salinity
    pK2 = (
        a
        + b / TempK
        + c / TempK**2
        + d * np.sqrt(Sal) / (1 + 1.11 * np.sqrt(Sal))
        + e * Sal / np.log(TempK)
        + f * np.sqrt(Sal) * TempK
    )
    return pK2


pk_ff = fitting_function(
    [cv.Temp_C.values, cv.Salinity.values],
    5.1703,
    2136.77,
    -177788,
    -0.4457,
    0.0674,
    -0.0008238,
)
cf = curve_fit(
    fitting_function,
    [cv.Temp_C.values, cv.Salinity.values],
    cv.param_pK2.values,
)
