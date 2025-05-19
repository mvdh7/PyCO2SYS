# %%
import numpy as np

from PyCO2SYS import CO2System


def test_adjust_dic_alkalinity():
    sys = CO2System(dic=2100, alkalinity=2300, temperature=np.array([10, 20]))
    sys.solve("pH")
    sysa = sys.adjust(temperature=np.array([20, 10]))
    assert np.allclose(sys["pH"], sysa["pH"][::-1])


def test_adjust_pCO2():
    sys = CO2System(pCO2=100, temperature=10)
    for method in range(1, 7):
        if method == 4:
            sysa = sys.adjust(temperature=11, method_fCO2=method, bh=30000)
        else:
            sysa = sys.adjust(temperature=11, method_fCO2=method)
        assert np.isclose(sys["pCO2"] * 1.04, sysa["pCO2"], rtol=0, atol=1)


# test_adjust_dic_alkalinity()
# test_adjust_pCO2()
