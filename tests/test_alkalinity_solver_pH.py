import numpy as np
import PyCO2SYS as pyco2

# Set conditions
alkalinity = 0
dic = 2500

# Get all pHs from TA and DIC
opt_pH_scale = np.array([1, 2, 3, 4])
results = pyco2.sys(alkalinity, dic, 1, 2, opt_pH_scale=np.array([1, 2, 3, 4]))


# Check they're consistent regardless of input scale
def test_pH_scale_consistency():
    assert np.all(np.isclose(results["pH_total"], results["pH"][opt_pH_scale == 1]))
    assert np.all(np.isclose(results["pH_sws"], results["pH"][opt_pH_scale == 2]))
    assert np.all(np.isclose(results["pH_free"], results["pH"][opt_pH_scale == 3]))
    assert np.all(np.isclose(results["pH_nbs"], results["pH"][opt_pH_scale == 4]))


# test_pH_scale_consistency()
