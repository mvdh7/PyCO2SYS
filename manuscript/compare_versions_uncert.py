import pandas as pd 
import PyCO2SYS as pyco2

# Switch to CO2SYS-MATLAB v3.2.0 conditions (note: not identical to v3.1.1)
pyco2.solve.get.initial_pH_guess = None  # use the more sophisticated pH guess
pyco2.solve.get.pH_tolerance = 0.0001  # use a looser tolerance for pH solvers
pyco2.solve.get.update_all_pH = False  # True keeps updating all pH's until all solved
pyco2.solve.get.halve_big_jumps = True  # different way to prevent too-big pH jumps
pyco2.solve.get.assume_pH_total = False  # replicate pH-Total assumption bug
pyco2.solve.delta.use_approximate_slopes = True  # don't use Autograd for solver slopes

co2ml = pd.read_csv("manuscript/results/compare_versions_co2s_v3.csv")
co2ml_u = pd.read_csv("manuscript/results/compare_versions_uncert.csv", na_values=-999)

# Convert constants options
co2ml["KSO4CONSTANTS"] = pyco2.convert.options_new2old(
    co2ml["KSO4CONSTANT"].values, co2ml["BORON"].values
)

# Run PyCO2SYS.CO2SYS under the same conditions
co2inputs = [
    co2ml[var].values
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
        "TNH4",
        "TH2S",
        "pHSCALEIN",
        "K1K2CONSTANTS",
        "KSO4CONSTANT",
        "KFCONSTANT",
        "BORON",
    ]
]
co2py = pd.DataFrame(pyco2.api.CO2SYS_MATLABv3(*co2inputs))

# Compute uncertainties
uncertainties_into = [c.split("_")[1] for c in co2ml_u.columns]
uncertainties_into = [c for c in uncertainties_into if c not in ["Hin", "Hout"]]
uncertainties_from = pyco2.uncertainty.pKs_OEDG18_ml
uncertainties_from["PAR1"] = co2ml.UPAR1
uncertainties_from["PAR2"] = co2ml.UPAR2
# uncertainties_from = {"pK1input": 0.001}
uncertainties = pd.DataFrame(pyco2.uncertainty.propagate(
    co2py, uncertainties_into, uncertainties_from
)[0])

# Compare uncertainties
co2py_u = {}
for k in uncertainties_into:
    co2py_u["u_{}_".format(k)] = uncertainties[k]
co2py_u = pd.DataFrame(co2py_u)
co2u = co2ml_u - co2py_u
co2u_pct = (100 * co2u / co2py_u).abs()

# Reset to PyCO2SYS conditions
pyco2.solve.get.initial_pH_guess = None
pyco2.solve.get.pH_tolerance = 1e-8
pyco2.solve.get.update_all_pH = False
pyco2.solve.get.halve_big_jumps = False
pyco2.solve.get.assume_pH_total = False
pyco2.solve.delta.use_approximate_slopes = False
