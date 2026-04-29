# %%
import numpy as np
import xarray as xr

import PyCO2SYS as pyco2


ds = xr.Dataset(
    {
        "temperature": ("dim_t", np.arange(0, 35)),
        "salinity": ("dim_s", np.arange(30, 40)),
    }
)
co2s = pyco2.sys(data=ds)
pk_CO2 = co2s.to_xarray("pk_CO2")
ds_results = co2s.to_xarray(["pk_H2CO3", "pk_HCO3"])
