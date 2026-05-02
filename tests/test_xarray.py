# %%
import numpy as np
import xarray as xr

import PyCO2SYS as pyco2


def test_xarray_broadcast():
    ds = xr.Dataset(
        {
            "temperature": ("dim_t", np.arange(0, 35)),
            "salinity": ("dim_s", np.arange(30, 40)),
        }
    )
    co2s = pyco2.sys(data=ds)
    pk_CO2 = co2s.to_xarray("pk_CO2")
    assert pk_CO2.shape == (ds.temperature * ds.salinity).shape
    ds_results = co2s.to_xarray(["pk_H2CO3", "pk_HCO3"])
    assert (
        ds_results.pk_H2CO3.shape == ds_results.pk_HCO3.shape == pk_CO2.shape
    )


def test_xarray():
    dic_overwrite = 2000
    wtf = 3
    ds = xr.Dataset(
        {
            "tco2": ("dim", [2000, 2100, 2200]),
            "PCO2": ("dim", [400, 450, 485]),
            "wtf": wtf,
            "extra": 45,
        }
    )
    co2s = pyco2.sys(
        data=ds,
        total_silicate=1.5,
        opt_k_carbonic=9,
        t="wtf",
        dic=dic_overwrite,
    )
    assert isinstance(co2s, pyco2.CO2System)
    assert co2s.ignored == {"extra"}
    assert np.allclose(co2s.temperature, wtf)
    assert co2s.dic == dic_overwrite
    pH = co2s.to_xarray("pH")
    assert isinstance(pH, xr.DataArray)
    ds_results = co2s.to_xarray(["pH", "ta"])
    assert isinstance(ds_results, xr.Dataset)


# test_xarray_broadcast()
# test_xarray()
