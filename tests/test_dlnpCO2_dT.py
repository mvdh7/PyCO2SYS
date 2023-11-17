import PyCO2SYS as pyco2
import numpy as np

# Simulate arguments
rng = np.random.default_rng(1)
npts = 1000
alkalinity = rng.normal(loc=2300, scale=50, size=npts)
dic = rng.normal(loc=2150, scale=50, size=npts)
salinity = rng.uniform(low=0, high=40, size=npts)
temperature = rng.uniform(low=-2, high=35, size=npts)
pressure = rng.uniform(low=0, high=10000, size=npts)
temperature_out = rng.uniform(low=-2, high=35, size=npts)
pressure_out = rng.uniform(low=0, high=10000, size=npts)
total_silicate = rng.uniform(low=0, high=100, size=npts)
total_phosphate = rng.uniform(low=0, high=100, size=npts)
total_ammonia = rng.uniform(low=0, high=100, size=npts)
total_sulfide = rng.uniform(low=0, high=100, size=npts)
opt_k_carbonic = rng.integers(low=1, high=18, size=npts)

# Solve
results = pyco2.sys(
    par1=alkalinity,
    par2=dic,
    par1_type=1,
    par2_type=2,
    salinity=salinity,
    temperature=temperature,
    pressure=pressure,
    temperature_out=temperature_out,
    pressure_out=pressure_out,
    total_silicate=total_silicate,
    total_phosphate=total_phosphate,
    total_ammonia=total_ammonia,
    total_sulfide=total_sulfide,
    opt_k_carbonic=opt_k_carbonic,
    opt_pressured_kCO2=0,
    grads_of=["pCO2"],
    grads_wrt=["temperature"],
)

# Extract values
dlnpCO2_dT_autograd = results["dlnpCO2_dT"]
dlnpCO2_dT_ffd = results["d_pCO2__d_temperature"] / results["pCO2"]
diff = dlnpCO2_dT_autograd - dlnpCO2_dT_ffd
abs_diff = np.abs(diff)
max_abs_diff = np.max(abs_diff)
diff_pct = 200 * diff / (dlnpCO2_dT_autograd + dlnpCO2_dT_ffd)
abs_diff_pct = np.abs(diff_pct)
max_abs_diff_pct = np.max(abs_diff_pct)


def test_dlnpCO2_dT():
    """Are all of the autograd values the same as the forward finite difference values
    to within 0.02%?
    """
    assert np.allclose(dlnpCO2_dT_autograd, dlnpCO2_dT_ffd, rtol=0.02 / 100, atol=0)


test_dlnpCO2_dT()
