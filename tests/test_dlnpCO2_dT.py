import PyCO2SYS as pyco2
import numpy as np

# Simulate arguments
rng = np.random.default_rng(1)
npts = 10_000
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
opt_pressured_kCO2 = rng.integers(low=0, high=2, size=npts)

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
    opt_pressured_kCO2=opt_pressured_kCO2,
    grads_of=["pCO2", "pCO2_out"],
    grads_wrt=["temperature", "temperature_out"],
)

# Extract values - input conditions
dlnpCO2_dT_autograd = results["dlnpCO2_dT"]
dlnpCO2_dT_ffd = results["d_pCO2__d_temperature"] / results["pCO2"]
diff = dlnpCO2_dT_autograd - dlnpCO2_dT_ffd
abs_diff = np.abs(diff)
max_abs_diff = np.max(abs_diff)
diff_pct = 200 * diff / (dlnpCO2_dT_autograd + dlnpCO2_dT_ffd)
abs_diff_pct = np.abs(diff_pct)
max_abs_diff_pct = np.max(abs_diff_pct)

# Extract values - output conditions
out_dlnpCO2_dT_autograd = results["dlnpCO2_dT_out"]
out_dlnpCO2_dT_ffd = results["d_pCO2_out__d_temperature_out"] / results["pCO2_out"]
out_diff = out_dlnpCO2_dT_autograd - out_dlnpCO2_dT_ffd
out_abs_diff = np.abs(out_diff)
out_max_abs_diff = np.max(out_abs_diff)
out_diff_pct = 200 * out_diff / (out_dlnpCO2_dT_autograd + out_dlnpCO2_dT_ffd)
out_abs_diff_pct = np.abs(out_diff_pct)
out_max_abs_diff_pct = np.max(out_abs_diff_pct)


def test_dlnpCO2_dT():
    """Are all of the autograd values the same as the forward finite difference values
    to within
        - 0.03% where opt_pressured_kCO2 is 0,
        - 4.00% where opt_pressured_kCO2 is 1,
        - 1e-3 across all values,
        - 1e-6 in the mean across all values, and
        - 0.01% in the mean across all values,
    under input and output conditions?
    """
    L = opt_pressured_kCO2 == 0
    assert np.allclose(
        dlnpCO2_dT_autograd[L], dlnpCO2_dT_ffd[L], rtol=0.03 / 100, atol=0
    )
    assert np.allclose(
        out_dlnpCO2_dT_autograd[L], out_dlnpCO2_dT_ffd[L], rtol=0.03 / 100, atol=0
    )
    assert np.allclose(
        dlnpCO2_dT_autograd[~L], dlnpCO2_dT_ffd[~L], rtol=4 / 100, atol=0
    )
    assert np.allclose(
        out_dlnpCO2_dT_autograd[~L], out_dlnpCO2_dT_ffd[~L], rtol=4 / 100, atol=0
    )
    assert np.allclose(dlnpCO2_dT_autograd, dlnpCO2_dT_ffd, rtol=0, atol=1e-3)
    assert np.allclose(out_dlnpCO2_dT_autograd, out_dlnpCO2_dT_ffd, rtol=0, atol=1e-3)
    assert np.mean(abs_diff) < 1e-6
    assert np.mean(out_abs_diff) < 1e-6
    assert np.mean(abs_diff_pct) < 0.01
    assert np.mean(out_abs_diff_pct) < 0.01


# test_dlnpCO2_dT()
