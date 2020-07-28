import numpy as np
import PyCO2SYS as pyco2

results = pyco2.CO2SYS_nd(
    np.linspace(2000, 2100, 11), np.vstack(np.linspace(2300, 2400, 21)), 2, 1,
)

grads_of = ["pH"]
grads_wrt = ["par1", "par2"]

results_forward = pyco2.uncertainty.forward_nd(results, grads_of, grads_wrt)
