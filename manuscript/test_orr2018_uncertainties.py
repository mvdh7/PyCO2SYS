import pandas as pd, numpy as np
import PyCO2SYS as pyco2

# Comparison with Orr et al. (2018) Table 2
orr2 = pd.read_csv("manuscript/data/orr2018-table2.csv")
orr = dict(
    par1=2300,
    par2=2000,
    par1_type=1,
    par2_type=2,
    temperature=18,
    salinity=35,
    total_phosphate=0,
    total_silicate=0,
    opt_k_carbonic=10,
    opt_total_borate=2,
)
grads_of = [c for c in orr2.columns if c not in ["wrt", "program"]]
grads_of.append("pH_total")
grads_wrt = ["par1", "par2", "temperature", "salinity"]
results = pyco2.sys(**orr, grads_of=grads_of, grads_wrt=grads_wrt)
for wrt in grads_wrt:
    results["d_Hfree__d_{}".format(wrt)] = (
        -np.log(10)
        * 10 ** -results["pH_total"]
        * results["d_pH_total__d_{}".format(wrt)]
        * 1e6
    )
for wrt in grads_wrt:
    if wrt == "par1":
        wrt_label = "alkalinity"
    elif wrt == "par2":
        wrt_label = "dic"
    else:
        wrt_label = wrt
    nrow = pd.Series({"wrt": wrt_label, "program": "PyCO2SYS"})
    for of in grads_of:
        nrow[of] = results["d_{}__d_{}".format(of, wrt)]
        if of == "Hfree":
            nrow[of] *= 1e3
    orr2 = orr2.append(nrow, ignore_index=True)
orr2_groups = orr2.groupby("wrt").mean()
orr2.set_index(["wrt", "program"], inplace=True)

# Comparison with Orr et al. (2018) Table 3
orr3 = pd.read_csv("manuscript/data/orr2018-table3.csv")
orr = dict(
    par1=2300,
    par2=2000,
    par1_type=1,
    par2_type=2,
    temperature=18,
    salinity=35,
    total_phosphate=2,
    total_silicate=60,
    opt_k_carbonic=10,
    opt_total_borate=2,
)
grads_wrt = ["total_phosphate", "total_silicate"]
results = pyco2.sys(**orr, grads_of=grads_of, grads_wrt=grads_wrt)
for wrt in grads_wrt:
    results["d_Hfree__d_{}".format(wrt)] = (
        -np.log(10)
        * 10 ** -results["pH_total"]
        * results["d_pH_total__d_{}".format(wrt)]
        * 1e6
    )
for wrt in grads_wrt:
    nrow = pd.Series({"wrt": wrt, "program": "PyCO2SYS"})
    for of in grads_of:
        nrow[of] = results["d_{}__d_{}".format(of, wrt)]
        if of == "Hfree":
            nrow[of] *= 1e3
    orr3 = orr3.append(nrow, ignore_index=True)
orr3_groups = orr3.groupby("wrt").mean()
orr3.set_index(["wrt", "program"], inplace=True)

# Comparison with Orr et al. (2018) Table 4
orr4 = pd.read_csv("manuscript/data/orr2018-table4.csv")
orr = dict(
    par1=2300,
    par2=2000,
    par1_type=1,
    par2_type=2,
    temperature=18,
    salinity=35,
    total_phosphate=2,
    total_silicate=60,
    opt_k_carbonic=10,
    opt_total_borate=1,  # note this is different from Tables 2 and 3!
)
uncertainty_into = [
    c for c in orr4.columns if c not in ["wrt", "program", "with_k_uncertainties"]
]
uncertainty_into.append("pH_total")
uncertainty_from = {"par1": 2, "par2": 2, "total_phosphate": 0.1, "total_silicate": 4}
results = pyco2.sys(
    **orr, uncertainty_into=uncertainty_into, uncertainty_from=uncertainty_from
)
results["u_Hfree"] = (
    np.log(10) * 10 ** -results["pH_total"] * results["u_pH_total"] * 1e6
)
nrow = pd.Series(
    {"wrt": "dic_alkalinity", "program": "PyCO2SYS", "with_k_uncertainties": "no"}
)
for into in uncertainty_into:
    nrow[into] = results["u_{}".format(into)]
    if into == "Hfree":
        nrow[into] *= 1e3
orr4 = orr4.append(nrow, ignore_index=True)
uncertainty_from.update(pyco2.uncertainty_OEDG18)
results = pyco2.sys(
    **orr, uncertainty_into=uncertainty_into, uncertainty_from=uncertainty_from
)
results["u_Hfree"] = (
    np.log(10) * 10 ** -results["pH_total"] * results["u_pH_total"] * 1e6
)
nrow = pd.Series(
    {"wrt": "dic_alkalinity", "program": "PyCO2SYS", "with_k_uncertainties": "yes"}
)
for into in uncertainty_into:
    nrow[into] = results["u_{}".format(into)]
    if into == "Hfree":
        nrow[into] *= 1e3
orr4 = orr4.append(nrow, ignore_index=True)
orr4_groups = orr4.groupby("with_k_uncertainties").mean()
orr4.set_index(["wrt", "program", "with_k_uncertainties"], inplace=True)


def test_table2_OEDG18():
    """Does PyCO2SYS agree with OEDG18's Table 2?"""
    for wrt in orr2_groups.index:
        for of in orr2_groups.columns:
            # if of != "Hfree":
            v_orr = orr2_groups.loc[wrt][of]
            v_pyco2 = orr2.loc[wrt].loc["PyCO2SYS"][of]
            assert np.isclose(
                v_orr, v_pyco2, rtol=1e-3, atol=0
            ), "Failed on {} / {}".format(of, wrt)


def test_table3_OEDG18():
    """Does PyCO2SYS agree with OEDG18's Table 3?"""
    for wrt in orr3_groups.index:
        for of in orr3_groups.columns:
            # if of != "Hfree":
            v_orr = orr3_groups.loc[wrt][of]
            v_pyco2 = orr3.loc[wrt].loc["PyCO2SYS"][of]
            assert np.isclose(
                v_orr, v_pyco2, rtol=1e-3, atol=0
            ), "Failed on {} / {}".format(of, wrt)


def test_table4_OEDG18():
    """Does PyCO2SYS agree with OEDG18's Table 4?"""
    for with_k in orr4_groups.index:
        for of in orr4_groups.columns:
            # if of != "Hfree":
            v_orr = orr4_groups.loc[with_k][of]
            v_pyco2 = orr4.loc["dic_alkalinity"].loc["PyCO2SYS"].loc[with_k][of]
            assert np.isclose(
                v_orr, v_pyco2, rtol=1e-4, atol=0
            ), "Failed on {} / {}".format(of, wrt)


test_table2_OEDG18()
test_table3_OEDG18()
test_table4_OEDG18()
