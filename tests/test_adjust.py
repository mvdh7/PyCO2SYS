# %%
import numpy as np

from PyCO2SYS import CO2System


def test_adjust_dic_pH():
    co2s = CO2System(dic=2100, pH=8.1, temperature=np.array([10, 20]))
    co2s.adjust(temperature=np.array([20, 10])).solve("alkalinity")
    # just checks that this runs without errors


def test_adjust_pCO2():
    sys = CO2System(pCO2=100, temperature=10)
    for method in range(1, 7):
        if method == 4:
            sysa = sys.adjust(temperature=11, method_fCO2=method, bh=30000)
        else:
            sysa = sys.adjust(temperature=11, method_fCO2=method)
        assert np.isclose(sys["pCO2"] * 1.04, sysa["pCO2"], rtol=0, atol=1)


# test_adjust_dic_pH()
# test_adjust_pCO2()
