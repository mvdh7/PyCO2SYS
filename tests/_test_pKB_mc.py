# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def fitting_function(t_s, a, b, c, d, e, f, g, h, i):
    Temp_K, Sal = t_s
    pKB = (
        +a
        + b * np.sqrt(Sal)
        + c / Temp_K
        + d * np.sqrt(Sal) / Temp_K
        + e * np.sqrt(Sal) / (1 + np.sqrt(Sal))
        + f * np.log(Temp_K)
        + g * np.sqrt(Sal) / np.log(Temp_K)
        + h * np.sqrt(Sal) * Temp_K
        + i * Sal * Temp_K
    )
    return pKB


df = pd.read_excel("tests/data/check_values_pKB_MMB26.xlsx")
noise_scale = np.sqrt(np.mean((df.pKB_combined_exp - df.pKB_MMB26) ** 2))
npts = len(df.pKB_combined_exp.values)

rng = np.random.default_rng(1)

all_fits = []
for _ in range(1000):
    noise = rng.normal(scale=noise_scale, size=npts)
    cf = curve_fit(
        fitting_function,
        [df.Temp_K, df.Sal.values],
        df.pKB_MMB26.values + noise,
        method="lm",
    )
    all_fits.append(cf[0])
all_fits = np.array(all_fits)
uncertainty_matrix = np.cov(all_fits, rowvar=False)

# %%
fig, ax = plt.subplots()
ax.scatter(uncertainty_matrix.ravel(), cf[1].ravel())
ax.axline((0, 0), slope=1)
