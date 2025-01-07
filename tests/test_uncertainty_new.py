# %%
import PyCO2SYS as pyco2
from PyCO2SYS import CO2System

sys = CO2System(dic=2300, alkalinity=2400)
sys.propagate("pH", pyco2.uncertainty_OEDG18)
