# %%
import numpy as np

import PyCO2SYS as pyco2


def test_co3_alkalinity_timeout():
    # co3 = 700, talk = 2000 is impossible
    # solver should timeout rather than hang infinitely,
    # and return a NaN there, while still returning a value
    # for co3 = 600.
    co2s = pyco2.sys(
        co3=[600, 700],
        talk=2000,
    )
    assert ~np.isnan(co2s.ph[0])
    assert np.isnan(co2s.ph[1])


# test_co3_alkalinity_timeout()
