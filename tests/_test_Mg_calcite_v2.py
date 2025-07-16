# %%
import PyCO2SYS as pyco2

co2s = pyco2.sys(
    co3=300,
    opt_Mg_calcite_type=3,
    opt_Mg_calcite_kt_Tdep=3,
).solve(["pkt_Mg_calcite_25C_1atm", "saturation_Mg_calcite"], store_steps=2)
# BUG if explicitly ask to solve a non-standard stored param then it
# should actually be stored even with store_steps != 2
print(co2s.pkt_Mg_calcite_25C_1atm)
