import PyCO2SYS as pyco2, numpy as np

# Import data from W74 Table III
data = np.genfromtxt("manuscript/data/weiss1974_tableIII.csv", delimiter=",")
salinity = data[0, 1:]
temperature = np.vstack(data[1:, 0])
temperature_K = temperature + 273.15
kCO2_W74 = data[1:, 1:]

# Calculate kCO2 with PyCO2SYS
kCO2_pyco2 = np.round(pyco2.equilibria.p1atm.kCO2_W74(temperature_K, salinity) * 1e2, 3)
kCO2_pyco2[0, :2] = np.nan


def test_kCO2_W74():
    assert np.all(np.isclose(kCO2_W74, kCO2_pyco2, rtol=0, atol=1e-5, equal_nan=True))


# test_kCO2_W74()
