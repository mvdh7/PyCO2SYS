# %%
from sys import path
path.append('/Users/ben/Documents/GitHub/PyCO2SYS')
import PyCO2SYS as pyco2
import numpy as np

# co2s = pyco2.sys(pH=8.1, dic=2100)
# co2s.solve("saturation_calcite")
# co2s.plot_graph(show_unknown=False, show_isolated=False)
# # TODO why does this ^ calculate TF and TSO4?

co2s = pyco2.sys(
    dic=2300,
    alkalinity=2350,
    opt_Mg_calcite_type=1,
    Mg_percent=10,
    opt_Mg_calcite_kt_Tdep=2,
    temperature=25,
    pressure=1000,
)

# IAP is fine
# TODO vant Hoff temperature dependence is not right
# BP82 is fine
# acf seem fine

co2s.solve(["kt_Mg_calcite_25C_1atm_minprep",
            "kt_Mg_calcite_25C_1atm_biogenic",
            "kt_Mg_calcite_25C_1atm_synthetic",
            "kt_Mg_calcite_1atm_vantHoff",
            "kt_Mg_calcite_1atm_PB82",
            "saturation_calcite",
            "saturation_aragonite",
            "saturation_Mg_calcite"], store_steps=2)
co2s.plot_graph(show_unknown=False, show_isolated=False)
print('C1',np.log10(co2s.kt_Mg_calcite_25C_1atm_minprep))
print('C2',np.log10(co2s.kt_Mg_calcite_25C_1atm_biogenic))
print('C3',np.log10(co2s.kt_Mg_calcite_25C_1atm_synthetic))
print('IAP',co2s.kt_Mg_calcite_25C_1atm_synthetic)
print('IAP',np.log10(co2s.kt_Mg_calcite_25C_1atm))
print()
print('vant Hoff',np.log10(co2s.kt_Mg_calcite_1atm_vantHoff))
print('PB82',np.log10(co2s.kt_Mg_calcite_1atm_PB82))
print()
print('gamma Ca', co2s.acf_Ca)
print('gamma Mg', co2s.acf_Mg)
print('gamma CO3', co2s.acf_CO3)
print()
print("[Ca]", co2s.Ca)
print("[Mg]", co2s.Mg)
print("[CO3]", co2s.CO3)
print()
print('Mg calcite')
print(np.log10(co2s.k_Mg_calcite_1atm))
print(np.log10(co2s.k_Mg_calcite))
print(co2s.saturation_Mg_calcite)
print()
print('calcite')
print(np.log10(co2s.k_calcite))
print(co2s.saturation_calcite)
print('aragonite')
print(np.log10(co2s.k_aragonite))
print(co2s.saturation_aragonite)


# %%
