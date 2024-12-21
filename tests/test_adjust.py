import numpy as np

from PyCO2SYS import CO2System


def test_adjust_dic_alkalinity():
    sys = CO2System(
        values=dict(dic=2100, alkalinity=2300, temperature=np.array([10, 20]))
    )
    sys.solve("pH")
    sysa = sys.adjust(temperature=np.array([20, 10]))
    assert np.allclose(sys.values["pH"], sysa.values["pH"][::-1])


def test_adjust_pCO2():
    sys = CO2System(values=dict(pCO2=100, temperature=10))
    for method in range(1, 7):
        print(method)
        if method == 4:
            sysa = sys.adjust(temperature=11, method_fCO2=method, bh_upsilon=30000)
        else:
            sysa = sys.adjust(temperature=11, method_fCO2=method)
        assert np.isclose(
            sys.values["pCO2"] * 1.04, sysa.values["pCO2"], rtol=0, atol=1
        )


# test_adjust_dic_alkalinity()
# test_adjust_pCO2()
