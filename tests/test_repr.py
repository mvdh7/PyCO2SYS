# %%
import PyCO2SYS as pyco2


def test_repr():
    co2s = pyco2.sys()
    assert isinstance(co2s.__repr__(), str)
    assert isinstance(co2s.opts.__repr__(), str)
    co2s = pyco2.sys(fco2=400)
    assert isinstance(co2s.__repr__(), str)
    assert isinstance(co2s.opts.__repr__(), str)
    co2s = pyco2.sys(fco2=400, ph=8)
    assert isinstance(co2s.__repr__(), str)
    assert isinstance(co2s.opts.__repr__(), str)
    co2a = co2s.adjust(t=15)
    assert isinstance(co2a.__repr__(), str)
    assert isinstance(co2a.opts.__repr__(), str)


# test_repr()
