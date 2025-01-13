# %%
import tracemalloc
import warnings
from datetime import datetime

import numpy as np

import PyCO2SYS as pyco2

rng = np.random.default_rng(1)
shape = (1000, 1000)
kwargs = {
    "par1": rng.normal(loc=2200, scale=1000, size=shape),
    "par2": rng.normal(loc=2000, scale=100, size=shape),
    "par1_type": 1,
    "par2_type": 2,
    "temperature": rng.uniform(low=-2, high=40, size=shape),
    "salinity": rng.uniform(low=0, high=50, size=shape),
    "pressure": rng.uniform(low=0, high=10000, size=shape),
    "total_silicate": rng.uniform(low=0, high=10, size=shape),
    "total_phosphate": rng.uniform(low=0, high=1100, size=shape),
}
with warnings.catch_warnings(action="ignore"):
    tracemalloc.start()
    print(tracemalloc.get_traced_memory())
    start = datetime.now()
    results = pyco2.sys(**kwargs)
    print(tracemalloc.get_traced_memory())
    tracemalloc.stop()
    print(datetime.now() - start)
    start = datetime.now()
    results = pyco2.sys(**kwargs)
    print(datetime.now() - start)
