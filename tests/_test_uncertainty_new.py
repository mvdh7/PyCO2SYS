# %%
import PyCO2SYS as pyco2
from PyCO2SYS import CO2System

# from PyCO2SYS.engine import CO2System

co2s = CO2System(dic=2300, alkalinity=2400)
# co2s = CO2System(values=dict(dic=2300, alkalinity=2400))
co2s.propagate("pH", pyco2.uncertainty_OEDG18)
