from time import time

import numpy as np
import pandas as pd

import PyCO2SYS as pyco2

gfile = "C:/Users/mphum/Downloads/GLODAPv2.2021_Merged_Master_File.csv"
glodap = pd.read_csv(gfile, na_values=-9999)
kwargs = {
    "par1": glodap.G2tco2.to_numpy(),
    "par2": glodap.G2talk.to_numpy(),
    "par1_type": 2,
    "par2_type": 1,
    "salinity": glodap.G2salinity.to_numpy(),
    "temperature": glodap.G2temperature.to_numpy(),
    "pressure": glodap.G2pressure.to_numpy(),
    "temperature_out": glodap.G2temperature.to_numpy(),
    "pressure_out": glodap.G2pressure.to_numpy(),
    "total_silicate": glodap.G2silicate.to_numpy(),
    "total_phosphate": glodap.G2phosphate.to_numpy(),
    "opt_k_carbonic": 10,
    "opt_buffers_mode": 2,
}
# Then run in console:
# %timeit pyco2.sys(**kwargs)

times = []
for i in range(7):
    print(i)
    go = time()
    pyco2.sys(**kwargs)
    this_time = time() - go
    times.append(this_time)

print(np.mean(times))
print(np.std(times))
