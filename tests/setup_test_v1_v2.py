import pickle

import numpy as np

import PyCO2SYS as pyco2

rng = np.random.default_rng(7)

npts = 1000
kwargs = {
    "par1": np.array([0, *rng.uniform(low=-1000, high=3000, size=npts)]),
    "par1_type": 1,
    "par2": np.array([*rng.uniform(low=0, high=3000, size=npts), 0]),
    "par2_type": 2,
    "temperature": np.array([0, *rng.uniform(low=-2, high=40, size=npts)]),
    "salinity": np.array([0, *rng.uniform(low=0, high=50, size=npts)]),
    "pressure": np.array([0, *rng.uniform(low=0, high=10000, size=npts)]),
    "total_silicate": np.array([0, *rng.uniform(low=0, high=100, size=npts)]),
    "total_phosphate": np.array([0, *rng.uniform(low=0, high=100, size=npts)]),
    "total_sulfide": np.array([0, *rng.uniform(low=0, high=100, size=npts)]),
    "total_ammonia": np.array([0, *rng.uniform(low=0, high=100, size=npts)]),
    "opt_k_carbonic": 10,
    "opt_pressured_kCO2": 1,
    "pressure_atmosphere": np.array([1, *rng.uniform(low=0.5, high=1.5, size=npts)]),
}
results = pyco2.sys(**kwargs)
with open("data/test_v1_v2.pkl", "wb") as f:
    pickle.dump(results, f)
