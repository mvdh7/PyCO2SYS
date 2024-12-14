import numpy as np

from PyCO2SYS import CO2System

sys = CO2System(values=dict(temperature=0, salinity=35), opts=dict(opt_k_carbonic=18))
sys.solve(["k_H2CO3", "k_HCO3"])
pK1 = -np.log10(sys.values["k_H2CO3"])
pK2 = -np.log10(sys.values["k_HCO3"])


def test_PLR18():
    """Compare against check values from PLR18 Table 3."""
    assert np.round(pK1, 4) == 6.1267
    assert np.round(pK2, 4) == 9.3940


# test_PLR18()
