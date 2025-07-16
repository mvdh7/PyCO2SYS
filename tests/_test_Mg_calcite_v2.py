# %%
import PyCO2SYS as pyco2

co2s = pyco2.sys(
    ta=2300,
    dic=2100,
    t=25,
    opt_Mg_calcite_type=3,
    opt_Mg_calcite_kt_Tdep=3,
    mg_fraction=0.18,
).solve(["pkt_Mg_calcite_25C_1atm", "saturation_Mg_calcite"], store_steps=2)
# BUG if explicitly ask to solve a non-standard stored param then it
# should actually be stored even with store_steps != 2
# print(co2s.pkt_Mg_calcite_25C_1atm)
print(co2s["saturation_mg_calcite"])
print(co2s["oc"])
print(co2s["oa"])
print("calc", co2s["pk_calcite"])
print("mgca", co2s["pk_mg_calcite"])
print("arag", co2s["pk_aragonite"])
