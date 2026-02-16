# %%
import importlib
import tracemalloc
import warnings
from datetime import datetime
from sys import path

import jax
import numpy as np

import PyCO2SYS as pyco2

rng = np.random.default_rng(1)
shape = (1200, 1200)
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
    results.solve(results.keys_all())
    memory_end = tracemalloc.get_traced_memory()
    print("Memory after solving:")
    print(f" -  now: {memory_end[0] / (1024 * 1024):.1f} MB")
    print(
        f" - peak: {memory_end[1] / (1024 * 1024):.1f} MB"
        + f" (now × {memory_end[1] / memory_end[0]:.1f})"
    )
    tracemalloc.stop()
    print("Times to run pyco2.sys:")
    t1 = datetime.now() - start
    print(t1)
    start = datetime.now()
    results = pyco2.sys(**kwargs)
    results.solve(results.keys_all())
    t2 = datetime.now() - start
    print(t2)
    print(memory_end)
    print(f"Number of elements in results: {len(results)}")

with open("tests/_speed_v2_results_mem.txt", "a") as f:
    f.write("[{}, {}],".format(*memory_end))
    f.write("\n")
with open("tests/_speed_v2_results_t1.txt", "a") as f:
    f.write('"' + str(t1) + '",')
    f.write("\n")
with open("tests/_speed_v2_results_t2.txt", "a") as f:
    f.write('"' + str(t2) + '",')
    f.write("\n")

# # %% Try get_func_of approach
# with warnings.catch_warnings(action="ignore"):
#     tracemalloc.start()
#     memory_start = tracemalloc.get_traced_memory()
#     print("Initial memory:")
#     print(f" -  now: {memory_start[0] / (1024 * 1024):.1f} MB")
#     print(f" - peak: {memory_start[1] / (1024 * 1024):.1f} MB")
#     start = datetime.now()
#     get_pH = results._get_func_of("pH")
#     get_pH_kwargs = kwargs.copy()
#     get_pH_kwargs.update(dict(total_ammonia=0, total_sulfide=0, total_nitrite=0))
#     memory_init = tracemalloc.get_traced_memory()
#     print("Memory after initialising:")
#     print(f" -  now: {memory_init[0] / (1024 * 1024):.1f} MB")
#     print(
#         f" - peak: {memory_init[1] / (1024 * 1024):.1f} MB"
#         + f" (now × {memory_init[1] / memory_init[0]:.1f})"
#     )
#     pH = get_pH(**get_pH_kwargs)
#     memory_end = tracemalloc.get_traced_memory()
#     print("Memory after solving:")
#     print(f" -  now: {memory_end[0] / (1024 * 1024):.1f} MB")
#     print(
#         f" - peak: {memory_end[1] / (1024 * 1024):.1f} MB"
#         + f" (now × {memory_end[1] / memory_end[0]:.1f})"
#     )
#     tracemalloc.stop()
#     print("Time to run pyco2.sys:")
#     print(datetime.now() - start)
#     print(memory_end)

# %% Tests for OSM 2026-02-16 --- solve for pH only, 1000x1000
memory = np.array(
    [
        [27982932, 152559370],
        [27984395, 152559915],
        [27981363, 152561457],
        [27977747, 152560816],
        [27977487, 152559253],
        [27982434, 152559833],
        [27981646, 152559338],
        [27982143, 152561341],
        [27981900, 152560303],
        [27981451, 152560369],
    ]
)
timer_first = np.array(
    [
        t.split(":0")[-1]
        for t in [
            "0:00:07.428858",
            "0:00:07.311899",
            "0:00:07.397539",
            "0:00:07.418438",
            "0:00:07.289359",
            "0:00:07.300932",
            "0:00:07.354765",
            "0:00:07.469507",
            "0:00:07.327888",
            "0:00:07.966272",
        ]
    ]
).astype(float)
timer_second = np.array(
    [
        t.split(":0")[-1]
        for t in [
            "0:00:04.286776",
            "0:00:04.618128",
            "0:00:04.314287",
            "0:00:04.308043",
            "0:00:04.352872",
            "0:00:04.685240",
            "0:00:04.308587",
            "0:00:04.293670",
            "0:00:04.257207",
            "0:00:04.605184",
        ]
    ]
).astype(float)
memory_mean = memory.mean(axis=0) / (1024 * 1024)
memory_std = memory.std(axis=0) / (1024 * 1024)
print(f"Peak memory  = {memory_mean[1]:.1f} ± {memory_std[1]:.3f} MB")
print(f"Final memory =  {memory_mean[0]:.1f} ±  {memory_std[0]:.3f} MB")
print(f"First run time   = {timer_first.mean():.2f} ± {timer_first.std():.2f} s")
print(f"Second run time  = {timer_second.mean():.3f} ± {timer_second.std():.3f} s")
print(
    f"Compilation time = {timer_first.mean() - timer_second.mean():.2f}"
    + f" ± {np.hypot(timer_first.std(), timer_second.std()):.2f} s"
)

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
print(f"Second run time  = {timer_second.mean():.3f} ± {timer_second.std():.3f} s")
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
