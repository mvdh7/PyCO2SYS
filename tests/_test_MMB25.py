# %%
import numpy as np
import pandas as pd

import PyCO2SYS as pyco2


def test_MMB25():
    cv = pd.read_csv("tests/data/check_values_pK2_MMB25.csv")
    co2s = pyco2.sys(
        s=cv.Salinity.values,
        t=cv.Temp_K.values - 273.15,
        opt_k_carbonic=19,
    ).solve("pk_HCO3_total_1atm")
    assert np.allclose(
        co2s.pk_HCO3_total_1atm - cv.param_pK2.values,
        0,
        rtol=0,
        atol=1e-12,
    )
