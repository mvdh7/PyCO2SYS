from PyCO2SYS.api import CO2SYS_wrap as CO2SYS


def test_CO2sys_api():
    import pandas as pd

    output = CO2SYS(dic=2000, alk=2300)

    isinstance(output, pd.DataFrame)
    assert output.shape[0] == 1


def test_CO2sys_api_vector():
    import pandas as pd
    import numpy as np

    output = CO2SYS(dic=np.linspace(2000, 2300, 11), alk=2300)

    isinstance(output, pd.DataFrame)
    assert output.shape[0] == 11


def test_CO2sys_raise_error():
    try:
        output = CO2SYS(dic=2000)
        output = Exception
    except KeyError:
        output = None
    except Exception as e:
        output = e

    if output is not None:
        raise output("Test should fail if no KeyError is not passed")


def test_CO2sys_xarray():
    import xarray as xr

    dic = xr.DataArray(
        data=[[2001, 2100], [2200, 2040]],
        coords={"lat": [30, 40], "lon": [20, 30]},
        dims=["lat", "lon"],
        name="dic",
    )

    output = CO2SYS(dic=dic, pco2=430)
    assert isinstance(output, xr.Dataset)
    assert isinstance(output.TAlk, xr.DataArray)
    assert output.TCO2.shape == (
        2,
        2,
    )


# test_CO2sys_api()
# test_CO2sys_api_vector()
# test_CO2sys_raise_error()
# test_CO2sys_xarray()
