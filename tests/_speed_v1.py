# %%
import tracemalloc
import warnings
from datetime import datetime

import numpy as np

import PyCO2SYS as pyco2

rng = np.random.default_rng(1)
shape = (1200, 1200)
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
for n in range(10):
    print(n)
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

# %% Results from 2025-02-11 (1000, 1000)
# Peak memory  = 13132.5 ± 0.007 MB
# Final memory = 671.8 ± 0.006 MB
# Run time   = 75.47 ± 8.17 s

# from 2026-02-16 (100, 100) after confirming similar to above
# Peak memory  = 117.4 ± 0.050 MB
# Final memory = 7.0 ± 0.047 MB
# Run time   = 0.66 ± 0.02 s

# (200, 200)
# Peak memory  = 456.5 ± 0.050 MB
# Final memory = 27.2 ± 0.047 MB
# Run time   = 1.05 ± 0.02 s

# (300, 300)
# Peak memory  = 1022.0 ± 0.050 MB
# Final memory = 60.8 ± 0.047 MB
# Run time   = 1.73 ± 0.02 s

# (400, 400)
# Peak memory  = 1813.4 ± 0.050 MB
# Final memory = 107.7 ± 0.047 MB
# Run time   = 2.74 ± 0.03 s

# (500, 500)
# Peak memory  = 2830.9 ± 0.050 MB
# Final memory = 168.2 ± 0.047 MB
# Run time   = 4.05 ± 0.09 s

# (600, 600)
# Peak memory  = 4731.0 ± 0.046 MB
# Final memory = 242.0 ± 0.044 MB
# Run time   = 6.64 ± 0.12 s

# (700, 700)
# Peak memory  = 6437.5 ± 0.046 MB
# Final memory = 329.3 ± 0.044 MB
# Run time   = 8.83 ± 0.11 s

# (800, 800)
# Peak memory  = 8406.6 ± 0.046 MB
# Final memory = 430.0 ± 0.044 MB
# Run time   = 12.84 ± 0.83 s

# (900, 900)
# Peak memory  = 11376.3 ± 0.047 MB
# Final memory = 544.2 ± 0.042 MB
# Run time   = 30.70 ± 4.72 s

# (10, 10)
# Peak memory  = 5.0 ± 0.048 MB
# Final memory = 0.4 ± 0.046 MB
# Run time   = 0.39 ± 0.01 s

# (1, 1)
# Peak memory  = 4.1 ± 0.048 MB
# Final memory = 0.3 ± 0.048 MB
# Run time   = 0.53 ± 0.01 s

# (1200, 1200)


# Data plotted and put into arrays in _plot_speed_osm.py
