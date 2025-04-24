# %%
from PyCO2SYS.engine import parameters_core


def test_parameters_core():
    assert parameters_core[0] == "alkalinity"
    assert parameters_core[1] == "dic"
    assert parameters_core[2] == "pH"
    assert parameters_core[3] == "pCO2"
    assert parameters_core[4] == "fCO2"
    assert parameters_core[5] == "CO3"
    assert parameters_core[6] == "HCO3"
    assert parameters_core[7] == "CO2"
    assert parameters_core[8] == "xCO2"
    assert parameters_core[9] == "saturation_calcite"
    assert parameters_core[10] == "saturation_aragonite"
    assert len(parameters_core) == 11


# test_parameters_core()
