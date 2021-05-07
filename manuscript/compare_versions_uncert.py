import pandas as pd, numpy as np
import PyCO2SYS as pyco2

# Switch to CO2SYS-MATLAB v3.2.0 conditions (note: not identical to v3.1.1)
pyco2.solve.get.initial_pH_guess = None  # use the more sophisticated pH guess
pyco2.solve.get.pH_tolerance = 0.0001  # use a looser tolerance for pH solvers
pyco2.solve.get.update_all_pH = False  # True keeps updating all pH's until all solved
pyco2.solve.get.halve_big_jumps = True  # different way to prevent too-big pH jumps
pyco2.solve.get.assume_pH_total = False  # replicate pH-Total assumption bug
pyco2.solve.delta.use_approximate_slopes = True  # don't use Autograd for solver slopes

# Import files generated with compare_versions_uncert.m
co2ml = pd.read_csv("manuscript/results/compare_versions_co2s_v3.csv")
co2ml_u = pd.read_csv("manuscript/results/compare_versions_uncert.csv", na_values=-999)

# Import file created from .mat file provided by JD Sharp on 15 Apr 2021
co2ml_u_jds = pd.read_csv(
    "manuscript/results/compare_versions_uncert_JDS.csv", na_values=-999
)

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
uncertainties = pd.DataFrame(
    pyco2.uncertainty.propagate(co2py, uncertainties_into, uncertainties_from)[0]
)

# Compare uncertainties --- doesn't really work with old MATLAB syntax, can't
# propagate both input/output Ks simultaneously
co2py_u = {}
for k in uncertainties_into:
    co2py_u["u_{}_".format(k)] = uncertainties[k]
co2py_u = pd.DataFrame(co2py_u)
co2u = co2ml_u - co2py_u
co2u_pct = (100 * co2u / co2py_u).abs()

# Compare with JD Sharp file
max_diff_pyco2 = (co2py_u - co2ml_u_jds).abs().max()
max_diff_jds = (co2ml_u - co2ml_u_jds).abs().max().loc[max_diff_pyco2.index]
# ^ shows that differences between Python and MATLAB are not due to errors in my MATLAB
#   because max_diff_pyco2 is everywhere much greater than max_diff_jds.

#%% Do properly with pyco2.sys v1.7
kwargs = {
    "par1": co2ml["PAR1"],
    "par2": co2ml["PAR2"],
    "par1_type": co2ml["PAR1TYPE"],
    "par2_type": co2ml["PAR2TYPE"],
    "salinity": co2ml["SAL"],
    "temperature": co2ml["TEMPIN"],
    "temperature_out": co2ml["TEMPOUT"],
    "pressure": co2ml["PRESIN"],
    "pressure_out": co2ml["PRESOUT"],
    "total_silicate": co2ml["SI"],
    "total_phosphate": co2ml["PO4"],
    "total_ammonia": co2ml["TNH4"],
    "total_sulfide": co2ml["TH2S"],
    "opt_pH_scale": co2ml["pHSCALEIN"],
    "opt_k_carbonic": co2ml["K1K2CONSTANTS"],
    "opt_k_bisulfate": co2ml["KSO4CONSTANT"],
    "opt_k_fluoride": co2ml["KFCONSTANT"],
    "opt_total_borate": co2ml["BORON"],
}
uncert_into = [
    "alkalinity",
    "dic",
    "pCO2",
    "fCO2",
    "bicarbonate",
    "carbonate",
    "aqueous_CO2",
    "saturation_calcite",
    "saturation_aragonite",
    "xCO2",
    "pCO2_out",
    "fCO2_out",
    "bicarbonate_out",
    "carbonate_out",
    "aqueous_CO2_out",
    "saturation_calcite_out",
    "saturation_aragonite_out",
    "xCO2_out",
]
uncert_from = {"{}_both".format(k): v for k, v in pyco2.uncertainty.pKs_OEDG18.items()}
# uncert_from = {"{}_out".format(k): v for k, v in pyco2.uncertainty.pKs_OEDG18.items()}
# uncert_from.update(pyco2.uncertainty.pKs_OEDG18)
uncert_from["total_borate__f"] = pyco2.uncertainty_OEDG18["total_borate__f"]
uncert_from.update({"par1": co2ml.UPAR1.values, "par2": co2ml.UPAR2.values})
results = pd.DataFrame(
    pyco2.sys(**kwargs, uncertainty_into=uncert_into, uncertainty_from=uncert_from)
)
pyco2sys_u = pd.DataFrame(
    {
        "u_TAlk_": results.u_alkalinity,
        "u_TCO2_": results.u_dic,
        "u_pCO2in_": results.u_pCO2,
        "u_fCO2in_": results.u_fCO2,
        "u_HCO3in_": results.u_bicarbonate,
        "u_CO3in_": results.u_carbonate,
        "u_CO2in_": results.u_aqueous_CO2,
        "u_OmegaCAin_": results.u_saturation_calcite,
        "u_OmegaARin_": results.u_saturation_aragonite,
        "u_xCO2in_": results.u_xCO2,
        "u_pCO2out_": results.u_pCO2_out,
        "u_fCO2out_": results.u_fCO2_out,
        "u_HCO3out_": results.u_bicarbonate_out,
        "u_CO3out_": results.u_carbonate_out,
        "u_CO2out_": results.u_aqueous_CO2_out,
        "u_OmegaCAout_": results.u_saturation_calcite_out,
        "u_OmegaARout_": results.u_saturation_aragonite_out,
        "u_xCO2out_": results.u_xCO2_out,
    }
)
final_diff_pct = 100 * (co2ml_u[pyco2sys_u.columns] - pyco2sys_u) / pyco2sys_u
final_diffs = pd.DataFrame(
    {
        "min_pct": final_diff_pct.min(),
        "mean_pct": final_diff_pct.mean(),
        "max_pct": final_diff_pct.max(),
    }
)


def test_uncertainty_comparison_input_v3_2_0():
    """Do MATLAB v3.2.0 and PyCO2SYS uncertainties agree?"""
    for _, r in final_diffs.iterrows():
        if r.name in [
            "u_TAlk_",
            "u_TCO2_",
            "u_pCO2in_",
            "u_fCO2in_",
            "u_HCO3in_",
            "u_CO3in_",
            "u_CO2in_",
            "u_OmegaCAin_",
            "u_OmegaARin_",
            "u_xCO2in_",
        ]:
            assert np.abs(r.mean_pct) < 0.5, "Failed on {}".format(r.name)
        elif r.name in [
            # "u_pCO2out_",
            # "u_fCO2out_",
            "u_HCO3out_",
            "u_CO3out_",
            # "u_CO2out_",
            # "u_OmegaCAout_",
            # "u_OmegaARout_",
            # "u_xCO2out_",
        ]:
            assert np.abs(r.mean_pct) < 10, "Failed on {}".format(r.name)


# test_uncertainty_comparison_input_v3_2_0()

#%% Reset to PyCO2SYS conditions
pyco2.solve.get.initial_pH_guess = None
pyco2.solve.get.pH_tolerance = 1e-8
pyco2.solve.get.update_all_pH = False
pyco2.solve.get.halve_big_jumps = False
pyco2.solve.get.assume_pH_total = False
pyco2.solve.delta.use_approximate_slopes = False
