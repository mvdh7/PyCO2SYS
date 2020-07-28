import numpy as np
import PyCO2SYS as pyco2

kwargs = {
    "salinity": 32,
    "k_carbonic_1": 1e-6,
}

results = pyco2.CO2SYS_nd(
    np.linspace(2000, 2100, 11), np.vstack(np.linspace(2300, 2400, 21)), 2, 1, **kwargs
)

grads_of = ["pH", "k_carbonic_1"]
grads_wrt = ["par1", "par2", "k_carbonic_1", "pk_carbonic_1", "temperature"]

co2derivs, dxs = pyco2.uncertainty.forward_nd(results, grads_of, grads_wrt, **kwargs)

# test_nd = pyco2.CO2SYS_nd(**results_forward)
