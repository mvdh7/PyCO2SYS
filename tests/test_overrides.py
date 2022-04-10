# Test that using the internal overrides doesn't affect the results
import pandas as pd, PyCO2SYS as pyco2

# Import input conditions: "compare_MATLABv3_1_1.csv" was generated in MATLAB
# using "manuscript/compare_MATLABv3_1_1.m".
co2matlab = pd.read_csv("manuscript/results/compare_MATLABv3_1_1.csv")

# Convert constants options
co2matlab = co2matlab[co2matlab.KSO4CONSTANT < 3]
co2matlab["KSO4CONSTANTS"] = pyco2.convert.options_new2old(
    co2matlab["KSO4CONSTANT"].values, co2matlab["BORON"].values
)

# Run PyCO2SYS.CO2SYS under the same conditions
co2inputs = [
    co2matlab[var].values
    for var in [
        "PAR1",
        "PAR2",
        "PAR1TYPE",
        "PAR2TYPE",
        "SAL",
        "TEMPIN",
        "TEMPOUT",
        "PRESIN",
        "PRESOUT",
        "SI",
        "PO4",
        "pHSCALEIN",
        "K1K2CONSTANTS",
        "KSO4CONSTANTS",
        "NH3",
        "H2S",
        "KFCONSTANT",
    ]
]
co2py = pyco2.CO2SYS(*co2inputs, opt_buffers_mode=1)

# Get override input dicts from co2py and use them in CO2SYS
totals = pyco2.engine.dict2totals_umol(co2py)
equilibria_in, equilibria_out = pyco2.engine.dict2equilibria(co2py)
co2py_override = pyco2.CO2SYS(
    *co2inputs,
    opt_buffers_mode=1,
    totals=totals,
    equilibria_in=equilibria_in,
    equilibria_out=equilibria_out
)

# Compare results - should be ~identical
co2py_diff = {k: co2py_override[k] - co2py[k] for k in co2py if k != "opt_buffers_mode"}
co2py_diff = pd.DataFrame(co2py_diff)
co2py_diff_absmax = co2py_diff.abs().max()


def test_overrides():
    assert all(co2py_diff_absmax.values < 1e-11)


# test_overrides()
