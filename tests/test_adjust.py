# %%
from PyCO2SYS import CO2System

sys = CO2System(values=dict(pCO2=100, temperature=10))
sysa = sys.adjust(temperature=11, method_fCO2=5)
print(sys.values["pCO2"], sysa.values["pCO2"])
