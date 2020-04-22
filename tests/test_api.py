from PyCO2SYS.api import CO2SYS_wrap as CO2SYS


def test_CO2sys_api():
    import pandas as pd

    output = CO2SYS(dic=2000, alk=2300)

    isinstance(output, pd.DataFrame)
    assert output.shape == (1, 112)


def test_CO2sys_xarray():
    import xarray as xr

    dic = xr.DataArray(
        data=[[2001, 2100], [2200, 2040]],
        coords={'lat': [30, 40], 'lon': [20, 30]},
        dims=['lat', 'lon'],
        name='dic'
    )

    output = CO2SYS(dic=dic, pco2=430)
    assert isinstance(output, xr.Dataset)
    assert isinstance(output.TAlk, xr.DataArray)
