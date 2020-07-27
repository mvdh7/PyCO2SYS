from time import time
import numpy as np
import pandas as pd
import PyCO2SYS as pyco2

# Import input conditions: "compare_MATLAB_extd.csv" was generated in MATLAB
# using "compare_MATLAB_extd.m".
co2matlab = pd.read_csv("validate/results/compare_MATLAB_extd.csv")

# Convert constants options
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
        "NH3",
        "H2S",
        "pHSCALEIN",
        "K1K2CONSTANTS",
        "KSO4CONSTANT",
        "KFCONSTANT",
        "BORON",
    ]
]
go = time()
# co2py = pyco2.CO2SYS(*co2inputs, buffers_mode="auto", WhichR=3)
co2py = pyco2.api.CO2SYS_MATLABv3(*co2inputs)
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
        if col
        not in ["RFin", "RFout", "PO4", "SAL", "SI", "H2S", "NH3", "KSO4CONSTANTS"]
    ]
    assert np.all(
        (pmad_co2py_matlab[checkcols] < 1e-3).values
        | np.isnan(pmad_co2py_matlab[checkcols].values)
    )


test_co2py_matlab()


# Compare new n-d approach
co2nd = pd.DataFrame(
    pyco2.CO2SYS_nd(
        co2inputs[0],
        co2inputs[1],
        co2inputs[2],
        co2inputs[3],
        salinity=co2inputs[4],
        temperature=co2inputs[5],
        temperature_out=co2inputs[6],
        pressure=co2inputs[7],
        pressure_out=co2inputs[8],
        total_silicate=co2inputs[9],
        total_phosphate=co2inputs[10],
        total_ammonia=co2inputs[11],
        total_sulfide=co2inputs[12],
        pH_scale_opt=co2inputs[13],
        carbonic_opt=co2inputs[14],
        bisulfate_opt=co2inputs[15],
        fluoride_opt=co2inputs[16],
        borate_opt=co2inputs[17],
        gas_constant_opt=3,
    )
)


def test_nd():
    assert np.all(co2nd.isocapnic_quotient_out.values == co2py.isoQout.values)
    assert np.all(co2nd.pH_sws.values == co2py.pHinSWS.values)


test_nd()
