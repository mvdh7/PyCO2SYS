# %%
import PyCO2SYS as pyco2

co2s = pyco2.sys(dic=2100, ta=2250)  # .adjust(t=15, p=1000).solve("pH")
# BUG ^ warn bug with __pre terms
