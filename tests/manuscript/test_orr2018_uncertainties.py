import numpy as np
import pandas as pd

import PyCO2SYS as pyco2
from PyCO2SYS import CO2System

renamer = {
    "Hfree": "H_free",
    "aqueous_CO2": "CO2",
    "bicarbonate": "HCO3",
    "carbonate": "CO3",
}

# Prepare for comparison with Orr et al. (2018) Table 2
orr2 = pd.read_csv("tests/manuscript/data/orr2018-table2.csv").rename(columns=renamer)
grads_of = [c for c in orr2.columns if c not in ["wrt", "program"]]
grads_of.append("pH")
grads_wrt = ["alkalinity", "dic", "temperature", "salinity"]
values_orr2 = dict(
    alkalinity=2300,
    dic=2000,
    temperature=18,
    salinity=35,
    total_phosphate=0,
    total_silicate=0,
)
opts_orr2 = dict(
    opt_k_carbonic=10,
    opt_total_borate=2,
)
sys2 = CO2System(**values_orr2, **opts_orr2)
sys2.get_grads(grads_of, grads_wrt)
# Get grads w.r.t. H_free manually from pH grad, because that's how Orr did it and the
# results are not consistent otherwise
grads_Hfree_manual_orr2 = {}
for wrt in grads_wrt:
    grads_Hfree_manual_orr2[wrt] = (
        -np.log(10) * 10 ** -sys2["pH"] * sys2.grads["pH"][wrt] * 1e6
    )
# Merge PyCO2SYS calculations with the Orr table
for wrt in grads_wrt:
    nrow = pd.DataFrame({"wrt": [wrt], "program": "PyCO2SYS"})
    for of in grads_of:
        nrow[of] = sys2.grads[of][wrt]
        if of == "H_free":
            nrow[of] = grads_Hfree_manual_orr2[wrt] * 1e3
    orr2 = pd.concat((orr2, nrow), ignore_index=True)
orr2_groups = orr2.drop(columns="program")
orr2_groups = orr2_groups.groupby("wrt").mean()
orr2.set_index(["wrt", "program"], inplace=True)

# Prepare for comparison with Orr et al. (2018) Table 3
orr3 = pd.read_csv("tests/manuscript/data/orr2018-table3.csv").rename(columns=renamer)
values_orr3 = dict(
    alkalinity=2300,
    dic=2000,
    temperature=18,
    salinity=35,
    total_phosphate=2,
    total_silicate=60,
)
opts_orr3 = dict(
    opt_k_carbonic=10,
    opt_total_borate=2,
)
grads_wrt = ["total_phosphate", "total_silicate"]
sys3 = CO2System(**values_orr3, **opts_orr3)
sys3.get_grads(grads_of, grads_wrt)
grads_Hfree_manual_orr3 = {}
for wrt in grads_wrt:
    grads_Hfree_manual_orr3[wrt] = (
        -np.log(10) * 10 ** -sys3["pH"] * sys3.grads["pH"][wrt] * 1e6
    )
for wrt in grads_wrt:
    nrow = pd.DataFrame({"wrt": [wrt], "program": "PyCO2SYS"})
    for of in grads_of:
        nrow[of] = sys3.grads[of][wrt]
        if of == "H_free":
            nrow[of] = grads_Hfree_manual_orr3[wrt] * 1e3
    orr3 = pd.concat((orr3, nrow), ignore_index=True)
orr3_groups = orr3.drop(columns="program")
orr3_groups = orr3_groups.groupby("wrt").mean()
orr3.set_index(["wrt", "program"], inplace=True)

# Prepare for comparison with Orr et al. (2018) Table 4
orr4 = pd.read_csv("tests/manuscript/data/orr2018-table4.csv").rename(columns=renamer)
values_orr4 = dict(
    alkalinity=2300,
    dic=2000,
    temperature=18,
    salinity=35,
    total_phosphate=2,
    total_silicate=60,
)
opts_orr4 = dict(
    opt_k_carbonic=10,
    opt_total_borate=1,  # note this is different from Tables 2 and 3!
)
uncertainty_into = [
    c for c in orr4.columns if c not in ["wrt", "program", "with_k_uncertainties"]
]
uncertainty_into.append("pH")
uncertainty_from = {
    "alkalinity": 2,
    "dic": 2,
    "total_phosphate": 0.1,
    "total_silicate": 4,
}
sys4 = CO2System(**values_orr4, **opts_orr4)
sys4.propagate(uncertainty_into, uncertainty_from)

u_Hfree_manual = np.log(10) * 10 ** -sys4["pH"] * sys4.uncertainty["pH"]["total"] * 1e6
nrow = pd.DataFrame(
    {"wrt": ["dic_alkalinity"], "program": "PyCO2SYS", "with_k_uncertainties": "no"}
)
for into in uncertainty_into:
    nrow[into] = sys4.uncertainty[into]["total"]
    if into == "H_free":
        nrow[into] = u_Hfree_manual * 1e3
orr4 = pd.concat((orr4, nrow), ignore_index=True)
# Now also include the pKs etc.
uncertainty_from.update(pyco2.uncertainty_OEDG18)
sys4.propagate(uncertainty_into, uncertainty_from)

u_Hfree_manual_pks = (
    np.log(10) * 10 ** -sys4["pH"] * sys4.uncertainty["pH"]["total"] * 1e6
)
nrow = pd.DataFrame(
    {"wrt": ["dic_alkalinity"], "program": "PyCO2SYS", "with_k_uncertainties": "yes"}
)
for into in uncertainty_into:
    nrow[into] = sys4.uncertainty[into]["total"]
    if into == "H_free":
        nrow[into] = u_Hfree_manual_pks * 1e3
orr4 = pd.concat((orr4, nrow), ignore_index=True)
orr4_groups = orr4.drop(columns=["wrt", "program"])
orr4_groups = orr4_groups.groupby("with_k_uncertainties").mean()
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


# test_table2_OEDG18()
# test_table3_OEDG18()
# test_table4_OEDG18()
