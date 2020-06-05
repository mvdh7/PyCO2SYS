from time import time
import numpy as np
import pandas as pd
import PyCO2SYS as pyco2

# Import input conditions: "compare_MATLAB_extd.csv" was generated in MATLAB
# using "compare_MATLAB_extd.m".
co2matlab = pd.read_csv("validate/results/compare_MATLAB_extd.csv")

# Convert constants options
co2matlab["KSO4CONSTANTS"] = pyco2.convert.options_new2old(
    co2matlab["KSO4CONSTANT"].values, co2matlab["BORON"].values)

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
go = time()
co2py = pyco2.CO2SYS(*co2inputs, buffers_mode="auto")
print("PyCO2SYS runtime = {:.6f} s".format(time() - go))
co2py = pd.DataFrame(co2py)

# Compare the results
cvars = list(co2matlab.keys())
co2py_matlab = co2py.subtract(co2matlab)  # PyCO2SYS.CO2SYS vs MATLAB

# Get maximum absolute differences in each variable
mad_co2py_matlab = co2py_matlab.abs().max()

# Max. abs. diff. as a percentage
pmad_co2py_matlab = 100 * mad_co2py_matlab / co2matlab.mean()

def test_co2py_matlab():
    checkcols = [
        col
        for col in pmad_co2py_matlab.index
        if col not in ["RFin", "RFout", "PO4", "SAL", "SI", "H2S", "NH3"]
    ]
    assert np.all(
        (pmad_co2py_matlab[checkcols] < 1e-6).values
        | np.isnan(pmad_co2py_matlab[checkcols].values)
    )
