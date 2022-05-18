# Not included yet as this part is neither complete nor documented and requires dask

# import PyCO2SYS as pyco2
# import numpy as np, xarray as xr

# # Generate arguments
# arrsize = (1000, 1000)
# rng = np.random.default_rng(1)
# np_alkalinity = rng.normal(loc=2300, scale=100, size=arrsize)
# np_dic = rng.normal(loc=2100, scale=100, size=arrsize)
# np_salinity = rng.normal(loc=35, scale=1, size=arrsize)
# sys_kwargs = {"temperature": 10, "salinity": np_salinity}

# # Get pH with pyco2.sys
# sys_pH = pyco2.sys(
#     par1=np_alkalinity,
#     par2=np_dic,
#     par1_type=1,
#     par2_type=2,
#     opt_k_carbonic=10,
#     **sys_kwargs
# )["pH_free"]

# # Get pH with NumPy arrays and the minimal function
# np_pH = pyco2.minimal.pH_from_alkalinity_dic(np_alkalinity, np_dic, **sys_kwargs)

# # Convert to xarray DataArrays
# lat = np.linspace(-90, 90, num=arrsize[0])
# lon = np.linspace(-180, 180, num=arrsize[1])
# xr_alkalinity = xr.DataArray(np_alkalinity, coords=[lat, lon], dims=["lat", "lon"])
# xr_dic = xr.DataArray(np_dic, coords=[lat, lon], dims=["lat", "lon"])
# xr_kwargs = sys_kwargs.copy()
# xr_kwargs["salinity"] = xr.DataArray(
#     np_salinity, coords=[lat, lon], dims=["lat", "lon"]
# )

# # Get pH with xarray DataArrays
# xr_pH = xr.apply_ufunc(
#     pyco2.minimal.pH_from_alkalinity_dic, xr_alkalinity, xr_dic, kwargs=xr_kwargs
# )

# # Chunk vectors
# chunks = {"lat": 100, "lon": 100}
# ch_alkalinity = xr_alkalinity.chunk(chunks)
# ch_dic = xr_dic.chunk(chunks)
# ch_kwargs = xr_kwargs.copy()
# ch_kwargs["salinity"] = xr_kwargs["salinity"].chunk(chunks)

# # Get pH, parallelised
# ch_pH = xr.apply_ufunc(
#     pyco2.minimal.pH_from_alkalinity_dic,
#     xr_alkalinity,
#     xr_dic,
#     kwargs=ch_kwargs,
#     dask="parallelized",
# )


# def test_pH_from_alkalinity_dic():
#     """Does pH computed with all the different methods agree?"""
#     assert np.allclose(sys_pH, np_pH, rtol=0, atol=1e-12)
#     assert np.allclose(sys_pH, xr_pH, rtol=0, atol=1e-12)
#     assert np.allclose(sys_pH, ch_pH, rtol=0, atol=1e-12)


# # test_pH_from_alkalinity_dic()
