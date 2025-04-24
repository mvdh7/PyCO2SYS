# %%
import importlib
import tracemalloc
import warnings
from datetime import datetime
from sys import path

import numpy as np

import PyCO2SYS as pyco2

ppath = "/Users/matthew/github/PyCO2SYS"
if ppath not in path:
    path.append(ppath)

rng = np.random.default_rng(1)
shape = (1000, 1000)
kwargs = {
    "alkalinity": rng.normal(loc=2200, scale=1000, size=shape),
    "dic": rng.normal(loc=2000, scale=100, size=shape),
    "temperature": rng.uniform(low=-2, high=40, size=shape),
    "salinity": rng.uniform(low=0, high=50, size=shape),
    "pressure": rng.uniform(low=0, high=10000, size=shape),
    "total_silicate": rng.uniform(low=0, high=10, size=shape),
    "total_phosphate": rng.uniform(low=0, high=1100, size=shape),
}
with warnings.catch_warnings(action="ignore"):
    tracemalloc.start()
    memory_start = tracemalloc.get_traced_memory()
    print("Initial memory:")
    print(f" -  now: {memory_start[0] / (1024 * 1024):.1f} MB")
    print(f" - peak: {memory_start[1] / (1024 * 1024):.1f} MB")
    start = datetime.now()
    results = pyco2.sys(**kwargs)
    memory_init = tracemalloc.get_traced_memory()
    print("Memory after initialising:")
    print(f" -  now: {memory_init[0] / (1024 * 1024):.1f} MB")
    print(
        f" - peak: {memory_init[1] / (1024 * 1024):.1f} MB"
        + f" (now × {memory_init[1] / memory_init[0]:.1f})"
    )
    results.solve("pH")
    memory_end = tracemalloc.get_traced_memory()
    print("Memory after solving:")
    print(f" -  now: {memory_end[0] / (1024 * 1024):.1f} MB")
    print(
        f" - peak: {memory_end[1] / (1024 * 1024):.1f} MB"
        + f" (now × {memory_end[1] / memory_end[0]:.1f})"
    )
    tracemalloc.stop()
    print("Times to run pyco2.sys:")
    print(datetime.now() - start)
    start = datetime.now()
    results = pyco2.sys(**kwargs)
    results.solve("pH")
    print(datetime.now() - start)
    print(memory_end)
    print(f"Number of elements in results: {len(results)}")

# %% Results of tests run on 2025-02-11
memory = np.array(
    [
        [61519538, 146228027],
        [61518761, 162224799],
        [61521024, 186228997],
        [61519035, 178229430],
        [61521049, 146225878],
        [61521190, 162229031],
        [61521109, 170226075],
        [61520897, 178230025],
        [61518485, 162226856],
        [61520524, 162226256],
    ]
)
timer_first = np.array(
    [
        t.split(":0")[-1]
        for t in [
            "0:00:03.398100",
            "0:00:03.414389",
            "0:00:03.369940",
            "0:00:03.412270",
            "0:00:03.365847",
            "0:00:03.378658",
            "0:00:03.372328",
            "0:00:03.451542",
            "0:00:03.384237",
            "0:00:03.420039",
        ]
    ]
).astype(float)
timer_second = np.array(
    [
        t.split(":0")[-1]
        for t in [
            "0:00:01.509018",
            "0:00:01.476695",
            "0:00:01.576584",
            "0:00:01.485075",
            "0:00:01.588812",
            "0:00:01.540104",
            "0:00:01.459591",
            "0:00:01.536936",
            "0:00:01.466345",
            "0:00:01.537466",
        ]
    ]
).astype(float)
memory_mean = memory.mean(axis=0) / (1024 * 1024)
memory_std = memory.std(axis=0) / (1024 * 1024)
print(f"Peak memory  = {memory_mean[1]:.1f} ± {memory_std[1]:.3f} MB")
print(f"Final memory =  {memory_mean[0]:.1f} ±  {memory_std[0]:.3f} MB")
print(f"First run time   = {timer_first.mean():.2f} ± {timer_first.std():.2f} s")
print(f"Second run time  = {timer_second.mean():.2f} ± {timer_second.std():.2f} s")
print(
    f"Compilation time = {timer_first.mean() - timer_second.mean():.2f}"
    + f" ± {np.hypot(timer_first.std(), timer_second.std()):.2f} s"
)

# %% pH only results
memory = np.array(
    [
        [27630842, 106248829],
        [27638208, 106254657],
        [27712953, 106109560],
        [27709767, 106108564],
        [27711603, 106109912],
        [27713688, 106110798],
        [27712048, 106107238],
        [27711826, 106107603],
        [27711108, 106107821],
        [27710845, 106105917],
    ]
)
timer_first = np.array(
    [
        t.split(":0")[-1]
        for t in [
            "0:00:01.525704",
            "0:00:01.524722",
            "0:00:01.574223",
            "0:00:01.581946",
            "0:00:01.562645",
            "0:00:01.581062",
            "0:00:01.564034",
            "0:00:01.580977",
            "0:00:01.599215",
            "0:00:01.589658",
        ]
    ]
).astype(float)
timer_second = np.array(
    [
        t.split(":0")[-1]
        for t in [
            "0:00:00.471721",
            "0:00:00.460880",
            "0:00:00.464526",
            "0:00:00.469261",
            "0:00:00.460154",
            "0:00:00.483310",
            "0:00:00.474302",
            "0:00:00.479408",
            "0:00:00.486146",
            "0:00:00.484206",
        ]
    ]
).astype(float)
memory_mean = memory.mean(axis=0) / (1024 * 1024)
memory_std = memory.std(axis=0) / (1024 * 1024)
print(f"Peak memory  = {memory_mean[1]:.1f} ± {memory_std[1]:.3f} MB")
print(f"Final memory =  {memory_mean[0]:.1f} ±  {memory_std[0]:.3f} MB")
print(f"First run time   = {timer_first.mean():.2f} ± {timer_first.std():.2f} s")
print(f"Second run time  = {timer_second.mean():.2f} ± {timer_second.std():.2f} s")
print(
    f"Compilation time = {timer_first.mean() - timer_second.mean():.2f}"
    + f" ± {np.hypot(timer_first.std(), timer_second.std()):.2f} s"
)
