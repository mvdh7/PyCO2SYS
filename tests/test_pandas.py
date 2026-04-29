# %%
import numpy as np
import pandas as pd

import PyCO2SYS as pyco2


def test_pandas():
    dic_overwrite = 2000
    wtf = 3
    df = pd.DataFrame(
        {
            "tco2": [2000, 2100, 2200],
            "PCO2": [400, 450, 485],
            "wtf": wtf,
            "extra": 45,
        }
    )
    co2s = pyco2.sys(
        data=df,
        total_silicate=1.5,
        opt_k_carbonic=9,
        t="wtf",
        dic=dic_overwrite,
    )
    assert isinstance(co2s, pyco2.CO2System)
    assert "extra" in co2s.ignored
    assert np.allclose(co2s.temperature, wtf)
    assert co2s.dic == dic_overwrite
    pH = co2s.to_pandas("pH")
    assert isinstance(pH, pd.Series)
    df_results = co2s.to_pandas(["pH", "ta"])
    assert isinstance(df_results, pd.DataFrame)


# test_pandas()
