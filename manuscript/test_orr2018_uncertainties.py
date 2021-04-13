import copy
import pandas as pd
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
)
grads_of = [c for c in orr2.columns if c not in ["wrt", "program"]]
grads_wrt = ["par1", "par2", "temperature", "salinity"]
results = pyco2.sys(**orr, grads_of=grads_of, grads_wrt=grads_wrt)
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
)
grads_wrt = ["total_phosphate", "total_silicate"]
results = pyco2.sys(**orr, grads_of=grads_of, grads_wrt=grads_wrt)
for wrt in grads_wrt:
    nrow = pd.Series({"wrt": wrt, "program": "PyCO2SYS"})
    for of in grads_of:
        nrow[of] = results["d_{}__d_{}".format(of, wrt)]
        if of == "Hfree":
            nrow[of] *= 1e3
    orr3 = orr3.append(nrow, ignore_index=True)

# Comparison with Orr et al. (2018) Table 4
orr4 = pd.read_csv("manuscript/data/orr2018-table4.csv")
uncertainty_into = [
    c for c in orr4.columns if c not in ["wrt", "program", "with_k_uncertainties"]
]
uncertainty_from = {"par1": 2, "par2": 2, "total_phosphate": 0.1, "total_silicate": 4}
results = pyco2.sys(
    **orr, uncertainty_into=uncertainty_into, uncertainty_from=uncertainty_from
)
nrow = pd.Series(
    {"wrt": "dic_alkalinity", "program": "PyCO2SYS", "with_k_uncertainties": False}
)
for into in uncertainty_into:
    nrow[into] = results["u_{}".format(into)]
    if into == "Hfree":
        nrow[into] *= 1e3
orr4 = orr4.append(nrow, ignore_index=True)
uncertainty_from.update(pyco2.u_pKs_OEDG18)
results = pyco2.sys(
    **orr, uncertainty_into=uncertainty_into, uncertainty_from=uncertainty_from
)
nrow = pd.Series(
    {"wrt": "dic_alkalinity", "program": "PyCO2SYS", "with_k_uncertainties": True}
)
for into in uncertainty_into:
    nrow[into] = results["u_{}".format(into)]
    if into == "Hfree":
        nrow[into] *= 1e3
orr4 = orr4.append(nrow, ignore_index=True)
