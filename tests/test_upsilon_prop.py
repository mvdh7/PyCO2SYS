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


def test_bl_prop():
    # The uncertainty in bl for the Takahashi et al. (1993) linear fit
    # as calculated by Humphreys (2024) should be assigned automatically
    # when adjusting fCO2 with that method
    co2a = pyco2.sys(fco2=[400, 500]).adjust(t=26, method_fCO2=5).prop("fco2")
    assert co2a.u.fco2.shape == (2, 2)
    L = co2a.u.fco2 > 0.1
    assert ((0.9 < co2a.u.fco2[L]) & (co2a.u.fco2[L] < 1.5)).all()
    assert np.isclose(co2a.u.fco2[0, 1], co2a.u.fco2[1, 0])


# test_bh_prop()
# test_bl_prop()
