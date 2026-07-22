# %%
import numpy as np

import PyCO2SYS as pyco2


def test_bh_prop():
    # The covariance matrix for fitted bh from Humphreys (2024) should
    # be assigned automatically when adjusting fCO2 with that method
    co2a = pyco2.sys(fco2=[400, 500]).adjust(t=10).prop("fco2")
    assert co2a.u.fco2.shape == (2, 2)
    assert ((0.2 < co2a.u.fco2) & (co2a.u.fco2 < 0.5)).all()
    assert np.isclose(co2a.u.fco2[0, 1], co2a.u.fco2[1, 0])


# test_bh_prop()
