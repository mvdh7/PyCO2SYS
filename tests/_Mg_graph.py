# %%
import PyCO2SYS as pyco2

# co2s = pyco2.sys(pH=8.1, dic=2100)
# co2s.solve("saturation_calcite")
# co2s.plot_graph(show_unknown=False, show_isolated=False)
# # TODO why does this ^ calculate TF and TSO4?

co2s = pyco2.sys(
    dic=2100,
    alkalinity=2250,
    opt_Mg_calcite_type=3,
    Mg_percent=15,
    opt_Mg_calcite_kt_Tdep=2,
)
co2s.solve(["saturation_Mg_calcite"], store_steps=2)
co2s.plot_graph(show_unknown=False, show_isolated=False)
print(co2s.saturation_Mg_calcite)
