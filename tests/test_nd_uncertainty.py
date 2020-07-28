import numpy as np
import PyCO2SYS as pyco2

# Calculate initial results
kwargs = {
    "salinity": 32,
    "k_carbonic_1": 1e-6,
}
results = pyco2.CO2SYS_nd(
    np.linspace(2000, 2100, 11), np.vstack(np.linspace(2300, 2400, 21)), 2, 1, **kwargs
)

# Get gradients
grads_of = ["pH", "k_carbonic_1"]
grads_wrt = ["par1", "par2", "k_carbonic_1", "pk_carbonic_1", "temperature"]
CO2SYS_derivs, dxs = pyco2.uncertainty.forward_nd(
    results, grads_of, grads_wrt, **kwargs
)

# Do independent uncertainty propagation
uncertainties_into = ["pH", "isocapnic_quotient", "dic"]
uncertainties_from = {
    "par1": 2,
    "par2": 2,
    "pk_carbonic_1": 0.02,
}
uncertainties, components = pyco2.uncertainty.propagate_nd(
    results, uncertainties_into, uncertainties_from, **kwargs,
)

# Try out the standard uncertainties of OEDG18
uncertainties_pk, components_pk = pyco2.uncertainty.propagate_nd(
    results, uncertainties_into, pyco2.uncertainty.pKs_OEDG18, **kwargs
)
