import PyCO2SYS as pyco2
import numpy as np

results = pyco2.sys(temperature=0, salinity=35, opt_k_carbonic=18)
pK1 = -np.log10(results["k_carbonic_1"])
pK2 = -np.log10(results["k_carbonic_2"])


def test_PLR18():
    """Compare against check values from PLR18 Table 3."""
    assert np.round(pK1, 4) == 6.1267
    assert np.round(pK2, 4) == 9.3940


# test_PLR18()
