# %%
from PyCO2SYS import CO2System

sys = CO2System(values=dict(dic=2100, alkalinity=2300))
sys.solve("pH", verbose=False)
sys.solve("fCO2", verbose=True)
# TOPOLOGICAL SORT!
