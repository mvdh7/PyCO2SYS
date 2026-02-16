# %%
import PyCO2SYS as pyco2

co2s = pyco2.sys(co2=10, pH=[7.8, 8, 8.2]).solve(["dic", "ta"])
