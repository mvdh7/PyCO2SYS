# This is testing PyCO2SYS with all the settings adjusted to match the CO2SYS-MATLAB
# original v2.0.5, to prove that we understand the reasons for differences in the
# results with PyCO2SYS's updated default settings.

from time import time
import pandas as pd, numpy as np, PyCO2SYS as pyco2

# Switch to original CO2SYS-MATLAB v2.0.5 conditions
pyco2.solve.get.initial_pH_guess = 8.0  # don't use the more sophisticated pH guess
pyco2.solve.get.pH_tolerance = 0.0001  # use a looser tolerance for pH solvers
pyco2.solve.get.update_all_pH = False  # True keeps updating all pH's until all solved
pyco2.solve.get.halve_big_jumps = True  # different way to prevent too-big pH jumps
pyco2.solve.get.assume_pH_total = True  # replicate pH-Total assumption bug
pyco2.solve.delta.use_approximate_slopes = True  # don't use Autograd for solver slopes

# Import input conditions: "compare_MATLABv2_0_5[_loop].csv" generated in MATLAB
# using "compare_MATLABv2_0_5.m".
co2matlab_inone = pd.read_csv("manuscript/results/compare_MATLABv2_0_5.csv")
co2matlab = pd.read_csv("manuscript/results/compare_MATLABv2_0_5_loop.csv")

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
co2py = pyco2.CO2SYS(*co2inputs, buffers_mode=0)
print("PyCO2SYS runtime = {:.6f} s".format(time() - go))
co2py = pd.DataFrame(co2py)

# short = ((co2inputs[2] == 1) & (co2inputs[3] == 4)) | (
#     (co2inputs[2] == 4) & (co2inputs[3] == 1)
# )
# short_ix = co2py.index[short]
# co2py_short = pd.DataFrame(
#     pyco2.CO2SYS(*[i[short] for i in co2inputs], buffers_mode=0)
# )
# short_diff = co2py.loc[short_ix, "BAlkin"].values - co2py_short["BAlkin"].values

# Also test the original CO2SYS clone
go = time()
DATA, HEADERS, _ = pyco2.original.CO2SYS(*co2inputs)
print("PyCO2SYS.original runtime = {:.6f} s".format(time() - go))
co2pyo = pd.DataFrame({header: DATA[:, h] for h, header in enumerate(HEADERS)})

# Compare the results
cvars = list(co2matlab.keys())
co2py_pyo = co2py.subtract(co2pyo)  # PyCO2SYS.CO2SYS vs PyCO2SYS.original.CO2SYS
co2py_matlab = co2py.subtract(co2matlab)  # PyCO2SYS.CO2SYS vs MATLAB
co2pyo_matlab = co2pyo.subtract(co2matlab_inone)  # PyCO2SYS.original.CO2SYS vs MATLAB

# # Having fixed the pH scale conversion in AlkParts, can now only compare where input
# # pH scale is Total (which worked correctly before) - as of v1.6.0.
# # Unless pyco2.solve.get.assume_pH_total = True, in which case the bug is replicated.
# which_scale = 1
# co2py_matlab = co2py_matlab[co2py["pHSCALEIN"] == which_scale]
# co2py_cut = co2py[co2py["pHSCALEIN"] == which_scale]

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
    # ^ Revelle Factor differences are due to understood CO2SYS-MATLAB bugs.
    # Nutrients/salinity are set to zero for freshwater case in PyCO2SYS but not in
    # CO2SYS-MATLAB.
    assert np.all(
        (pmad_co2py_matlab[checkcols] < 1e-10).values
        | np.isnan(pmad_co2py_matlab[checkcols].values)
    )


# # Run tests
# test_co2pyo_matlab()
# test_co2py_matlab()


# Reset to PyCO2SYS conditions
pyco2.solve.get.initial_pH_guess = None
pyco2.solve.get.pH_tolerance = 1e-8
pyco2.solve.get.update_all_pH = False
pyco2.solve.get.halve_big_jumps = False
pyco2.solve.get.assume_pH_total = False
pyco2.solve.delta.use_approximate_slopes = False
