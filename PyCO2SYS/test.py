# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Test the internal consistency of PyCO2SYS and compare its results with those from 
other software tools.
"""

from autograd.numpy import hstack, isin, isscalar, meshgrid
from . import engine


# Define parameter type numbers and their names in the CO2SYS output dict
_partypes = {
    1: "TAlk",
    2: "TCO2",
    3: "pHin",
    4: "pCO2in",
    5: "fCO2in",
    6: "CO3in",
    7: "HCO3in",
    8: "CO2in",
}


def _rr_parcombos(par1type, par2type):
    """Generate all possible valid pairs of parameter type numbers excluding the input
    pair.
    """
    assert isscalar(par1type) & isscalar(par2type), "Both inputs must be scalar."
    # Get all possible combinations of parameter type numbers
    allpars = list(_partypes.keys())
    par1s, par2s = meshgrid(allpars, allpars)
    par1s = par1s.ravel()
    par2s = par2s.ravel()
    # Select only valid combinations and cut out input combination
    allIcases = engine.getIcase(par1s, par2s, checks=False)
    inputIcase = engine.getIcase(par1type, par2type, checks=False)
    valid = (par1s != par2s) & ~isin(allIcases, [45, 48, 58, inputIcase])
    par1s = par1s[valid]
    par2s = par2s[valid]
    # Icases = pyco2.engine.getIcase(par1s, par2s, checks=True)  # checks if all valid
    return par1s, par2s


def roundrobin(
    par1,
    par2,
    par1type,
    par2type,
    sal,
    temp,
    pres,
    si,
    phos,
    pHscale,
    k1k2,
    kso4,
    **kwargs
):
    """Solve the core marine carbonate system from given input parameters, then solve
    again from the results using every other possible combination of input pairs.
    """
    # Check all inputs are scalar
    nonscalar_message = "All inputs must be scalar."
    assert all(
        [isscalar(v) for k, v in locals().items() if k != "kwargs"]
    ), nonscalar_message
    if "kwargs" in locals().keys():
        assert all(
            [isscalar(v) for k, v in locals()["kwargs"].items()]
        ), nonscalar_message
    # Solve the MCS using the initial input pair
    args = (sal, temp, temp, pres, pres, si, phos, pHscale, k1k2, kso4)
    res0 = engine.CO2SYS(par1, par2, par1type, par2type, *args, **kwargs)
    # Extract the core variables
    res0core = hstack([res0[_partypes[i]] for i in range(1, 9)])
    # Generate new inputs, all combinations
    par1types, par2types = _rr_parcombos(0, 0)
    par1s = res0core[par1types - 1]
    par2s = res0core[par2types - 1]
    # Solve the MCS again but from all combinations
    res = engine.CO2SYS(par1s, par2s, par1types, par2types, *args, **kwargs)
    # Calculate differences from original to aid comparisons
    nodiffs = [
        "PAR1TYPE",
        "PAR2TYPE",
        "K1K2CONSTANTS",
        "KSO4CONSTANTS",
        "KSO4CONSTANT",
        "KFCONSTANT",
        "BORON",
        "pHSCALEIN",
        "buffers_mode",
    ]
    diff = {k: v - res0[k] if k not in nodiffs else v for k, v in res.items()}
    return res, diff
