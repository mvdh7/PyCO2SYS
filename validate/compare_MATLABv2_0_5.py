from time import time
import pandas as pd
import numpy as np
import PyCO2SYS as pyco2

# Make sure we're using the original CO2SYS TA equation
pyco2.solve.get.TAfromTCpH = pyco2.solve.get.TAfromTCpH_original

# Import input conditions: "compare_MATLABv2_0_5.csv" was generated in MATLAB
# using "compare_MATLABv2_0_5.m".
co2matlab = pd.read_csv("validate/results/compare_MATLABv2_0_5.csv")

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
    ]
]
go = time()
co2py = pyco2.CO2SYS(*co2inputs, buffers_mode="auto")
print("PyCO2SYS runtime = {:.6f} s".format(time() - go))
co2py = pd.DataFrame(co2py)

# Also test the original CO2SYS clone
go = time()
DATA, HEADERS, _ = pyco2.original.CO2SYS(*co2inputs)
print("PyCO2SYS.original runtime = {:.6f} s".format(time() - go))
co2pyo = pd.DataFrame({header: DATA[:, h] for h, header in enumerate(HEADERS)})

# Compare the results
cvars = list(co2matlab.keys())
co2py_pyo = co2py.subtract(co2pyo)  # PyCO2SYS.CO2SYS vs PyCO2SYS.original.CO2SYS
co2py_matlab = co2py.subtract(co2matlab)  # PyCO2SYS.CO2SYS vs MATLAB
co2pyo_matlab = co2pyo.subtract(co2matlab)  # PyCO2SYS.original.CO2SYS vs MATLAB

# Get maximum absolute differences in each variable
mad_co2py_matlab = co2py_matlab.abs().max()
mad_co2pyo_matlab = co2pyo_matlab.abs().max()

# Max. abs. diff. as a percentage
pmad_co2py_matlab = 100 * mad_co2py_matlab / co2matlab.mean()
pmad_co2pyo_matlab = 100 * mad_co2pyo_matlab / co2matlab.mean()


def test_co2pyo_matlab():
    assert pmad_co2pyo_matlab.max() < 1e-9


def test_co2py_matlab():
    checkcols = [
        col
        for col in pmad_co2py_matlab.index
        if col not in ["RFin", "RFout", "PO4", "SAL", "SI"]
    ]
    assert np.all(
        (pmad_co2py_matlab[checkcols] < 1e-5).values
        | np.isnan(pmad_co2py_matlab[checkcols].values)
    )


# Run tests
test_co2pyo_matlab()
test_co2py_matlab()
