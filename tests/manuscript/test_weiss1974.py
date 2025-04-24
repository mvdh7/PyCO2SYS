# %%
import numpy as np

import PyCO2SYS as pyco2

# Import data from W74 Table III
data = np.genfromtxt("tests/manuscript/data/weiss1974_tableIII.csv", delimiter=",")
salinity = data[0, 1:]
temperature = np.vstack(data[1:, 0])
kCO2_W74 = data[1:, 1:]

# Calculate kCO2 with PyCO2SYS
kCO2_pyco2 = np.array(
    np.round((10 ** -pyco2.equilibria.p1atm.pk_CO2_W74(temperature, salinity)) * 1e2, 3)
)
kCO2_pyco2[0, :2] = np.nan


def test_kCO2_W74():
    assert np.all(np.isclose(kCO2_W74, kCO2_pyco2, rtol=0, atol=1e-5, equal_nan=True))


# test_kCO2_W74()
