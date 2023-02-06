import numpy as np
import PyCO2SYS as pyco2


def test_total_magnesium():
    """Does the default salinity give the reference composition Mg value?"""
    assert np.isclose(pyco2.sys()["total_magnesium"], 0.0547421 * 1e6)


# test_total_magnesium()
