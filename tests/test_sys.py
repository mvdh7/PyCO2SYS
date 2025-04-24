# %%
import PyCO2SYS as pyco2


def test_no_kwargs():
    co2s = pyco2.sys()
    co2s.solve()
    assert isinstance(co2s, pyco2.CO2System)


# test_no_kwargs()
