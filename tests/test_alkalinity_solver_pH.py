import copy
import numpy as np
import PyCO2SYS as pyco2

# Set conditions
alkalinity = 0
dic = 2500

# Get all pHs from TA and DIC
opt_pH_scale = np.array([1, 2, 3, 4])
results_original = pyco2.CO2SYS_nd(alkalinity, dic, 1, 2, opt_pH_scale=opt_pH_scale)

# Switch to fixed alkalinity function
alk_func = copy.deepcopy(pyco2.solve.get.TAfromTCpH)
pyco2.solve.get.TAfromTCpH = pyco2.solve.get.TAfromTCpH_fixed

# Get all pHs from TA and DIC, again
results_fixed = pyco2.CO2SYS_nd(
    alkalinity, dic, 1, 2, opt_pH_scale=np.array([1, 2, 3, 4])
)

# Check they're consistent regardless of input scale
def test_pH_scale_consistency():
    assert np.all(
        np.isclose(results_fixed["pH_total"], results_fixed["pH"][opt_pH_scale == 1])
    )
    assert np.all(
        np.isclose(results_fixed["pH_sws"], results_fixed["pH"][opt_pH_scale == 2])
    )
    assert np.all(
        np.isclose(results_fixed["pH_free"], results_fixed["pH"][opt_pH_scale == 3])
    )
    assert np.all(
        np.isclose(results_fixed["pH_nbs"], results_fixed["pH"][opt_pH_scale == 4])
    )


test_pH_scale_consistency()


# Revert to original alkalinity function
pyco2.solve.get.TAfromTCpH = alk_func
