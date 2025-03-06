# %%
import xarray as xr
from jax import numpy as np

from PyCO2SYS.engine import da_to_array


def prep(dims_a, dims_b):
    a = np.array([[0, 1, 2], [3, 4, 5]])
    b = a.transpose()
    xa = xr.DataArray(a, dims=dims_a)
    xb = xr.DataArray(b, dims=dims_b)
    data = xr.Dataset({"xa": xa, "xb": xb})
    xr_dims = list(data.sizes.keys())
    return xa, xb, xr_dims


def test_same_dims():
    xa, xb, xr_dims = prep(("a1", "a2"), ("a2", "a1"))
    na = da_to_array(xa, xr_dims)
    nb = da_to_array(xb, xr_dims)
    assert na.shape == (2, 3)
    assert na.shape == nb.shape
    assert np.all(na == nb)


def test_one_overlap():
    xa, xb, xr_dims = prep(("a1", "a2"), ("a2", "b1"))
    na = da_to_array(xa, xr_dims)
    nb = da_to_array(xb, xr_dims)
    assert na.shape == (2, 3, 1)
    assert nb.shape == (1, 3, 2)


def test_no_overlap():
    xa, xb, xr_dims = prep(("a1", "a2"), ("b1", "b2"))
    na = da_to_array(xa, xr_dims)
    nb = da_to_array(xb, xr_dims)
    assert na.shape == (2, 3, 1, 1)
    assert nb.shape == (1, 1, 3, 2)


# test_same_dims()
# test_one_overlap()
# test_no_overlap()
