# %%
import numpy as np

import PyCO2SYS as pyco2


co2s = pyco2.sys(temperature=0, salinity=35, opt_k_carbonic=18)
co2s.solve(["pk_H2CO3", "pk_HCO3"])


def test_PLR18():
    """Compare against check values from PLR18 Table 3."""
    assert np.round(co2s.pk_H2CO3, 4) == 6.1267
    assert np.round(co2s.pk_HCO3, 4) == 9.3940


# test_PLR18()
