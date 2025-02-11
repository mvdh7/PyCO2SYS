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
timer = []
memory = []
for _ in range(10):
    print("")
    with warnings.catch_warnings(action="ignore"):
        tracemalloc.start()
        memory_start = tracemalloc.get_traced_memory()
        print("Initial memory:")
        print(f" -  now: {memory_start[0] / (1024 * 1024):.1f} MB")
        print(f" - peak: {memory_start[1] / (1024 * 1024):.1f} MB")
        start = datetime.now()
        results = pyco2.sys(**kwargs)
        memory_end = tracemalloc.get_traced_memory()
        print("Memory after pyco2.sys:")
        print(f" -  now: {memory_end[0] / (1024 * 1024):.1f} MB")
        print(
            f" - peak: {memory_end[1] / (1024 * 1024):.1f} MB"
            + f" (now × {memory_end[1] / memory_end[0]:.1f})"
        )
        tracemalloc.stop()
        print("Time to run pyco2.sys once:")
        time_end = datetime.now() - start
        print(time_end)
        print(f"Number of elements in results: {len(results)}")
        timer.append(time_end)
        memory.append(memory_end)
memory = np.array(memory)
memory_mean = memory.mean(axis=0) / (1024 * 1024)
memory_std = memory.std(axis=0) / (1024 * 1024)
timer = np.array([t.total_seconds() for t in timer])
print("")
print(f"Peak memory  = {memory_mean[1]:.1f} ± {memory_std[1]:.3f} MB")
print(f"Final memory = {memory_mean[0]:.1f} ± {memory_std[0]:.3f} MB")
print(f"Run time   = {timer.mean():.2f} ± {timer.std():.2f} s")

# %% Results from 2015-02-11
# Peak memory  = 13132.5 ± 0.007 MB
# Final memory = 671.8 ± 0.006 MB
# Run time   = 75.47 ± 8.17 s
