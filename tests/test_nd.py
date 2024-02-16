import numpy as np, PyCO2SYS as pyco2

# Set up for basic conditioning check
par1c = np.random.normal(size=(5, 4, 3))
par2c = [10, 11, 12]
par3c = np.random.normal(size=(5, 4, 3))
par4c = np.random.normal(size=(3, 4, 5))
par5c = 12.2
par6c = "test"
par7c = np.random.normal(size=(5, 1, 3))
par8c = ["test"]
inputs = {
    "PAR1": par1c,
    "PAR2": par2c,
    "par3": par3c,
    "par4": par4c,  # this one should make it fail
    "par5": par5c,
    "par6": par6c,
    "par7": par7c,
    "par8": par8c,
}
inputs_no_4 = {k: v for k, v in inputs.items() if k != "par4"}


def test_conditioning():
    # Check the working case
    cond = pyco2.engine.nd.condition(inputs_no_4, to_shape=(10, 5, 4, 3))
    assert np.shape(cond["PAR1"]) == (10, 5, 4, 3)
    assert np.shape(cond["PAR2"]) == (10, 5, 4, 3)
    assert np.shape(cond["par3"]) == (10, 5, 4, 3)
    assert np.isscalar(cond["par5"])
    assert np.isscalar(cond["par6"])
    assert np.shape(cond["par7"]) == (10, 5, 4, 3)
    assert np.shape(cond["par8"]) == (10, 5, 4, 3)
    # Check the non-working case (should print out a "PyCO2SYS error")
    cond_4 = pyco2.engine.nd.condition(inputs)
    assert cond_4 is None


# test_conditioning()


# %% Now move on to the main function
par1 = [2300, 2150, 8.3]
par2 = [2150]
par1_type = [1, 1, 3]
par2_type = 2
kwargs = {
    "salinity": [[35, 31, 0.8], [34, 34, 34]],
    "total_sulfate": [[3], [5]],
    "k_carbonic_1": [1e-6, 1.1e-6, 1.2e-6],
    "temperature_out": 0,
}
co2nd = pyco2.sys(par1, par2, par1_type, par2_type, **kwargs)


def test_nd_misc():
    assert (
        np.shape(co2nd["k_carbonic_2"])
        == pyco2.engine.nd.broadcast1024(
            par1, par2, par1_type, par2_type, *kwargs.values()
        ).shape
    )


# test_nd_misc()


# Test with all scalar inputs
co2nd_scalar = pyco2.CO2SYS_nd(2300, 2150, 1, 2)


def test_scalars():
    assert np.all([np.size(v) == 1 for v in co2nd_scalar.values()])


# test_scalars()


# Test with TA/DIC grid
par1g = np.linspace(2100, 2400, 11)
par2g = np.vstack(np.linspace(2000, 2300, 11))
par1g_type = 1
par2g_type = 2
co2nd_grid = pyco2.CO2SYS_nd(par1g, par2g, par1g_type, par2g_type)


def test_grid():
    assert (
        np.shape(co2nd_grid["alkalinity"])
        == np.shape(co2nd_grid["dic"])
        == np.broadcast(par1g, par2g).shape
    )


# test_grid()
