from time import time
import pandas as pd
import PyCO2SYS as pyco2

# Import input conditions: "compare_MATLAB_extd.csv" was generated in MATLAB
# using "compare_MATLAB_extd.m".
co2matlab = pd.read_csv("validate/results/compare_MATLAB_extd.csv")

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
    ]
]
go = time()
co2py, allgrads = pyco2.CO2SYS(*co2inputs, buffers_mode="auto")
print("PyCO2SYS runtime = {:.6f} s".format(time() - go))
co2py = pd.DataFrame(co2py)

# Compare the results
cvars = list(co2matlab.keys())
co2py_matlab = co2py.subtract(co2matlab)  # PyCO2SYS.CO2SYS vs MATLAB

# Get maximum absolute differences in each variable
mad_co2py_matlab = co2py_matlab.abs().max()

# Max. abs. diff. as a percentage
pmad_co2py_matlab = 100 * mad_co2py_matlab / co2matlab.mean()

# # Try OO version
# go = time()
# whoseKSO4, whoseTB = pyco2.convert.options_old2new(co2inputs[13])
# co2oo = pyco2.api.oo.MCS(co2inputs[0], co2inputs[1], co2inputs[2], co2inputs[3],
#     psal=co2inputs[4],
#     temp_in=co2inputs[5],
#     temp_out=co2inputs[6],
#     pres_in=co2inputs[7],
#     pres_out=co2inputs[8],
#     tSi=co2inputs[9],
#     tPO4=co2inputs[10],
#     tNH3=co2inputs[14],
#     tH2S=co2inputs[15],
#     ph_scale=co2inputs[11],
#     whichKs=co2inputs[12],
#     whoseKSO4=whoseKSO4,
#     whoseKF=1,
#     whoseTB=whoseTB,
#     buffers_mode="auto",)
# print("PyCO2SYS.api.oo.MCS runtime = {:.6f} s".format(time() - go))
# co2oo = pd.DataFrame(co2oo.mcs)
# goodcols = [col for col in co2oo.columns if col != "buffers_mode"]
# co2oo_co2py = co2oo[goodcols].subtract(co2py)
