# %%
import glodap
import numpy as np
import xarray as xr

import PyCO2SYS as pyco2

co2s = pyco2.sys(
    dic=[2100, 2150, 2200],
    ta=[2300, 2325, 2350],
    t=13,
    s=20,
    p=50,
    tsi=10,
).solve(["pH", "fco2"])

# Solve with no params
co2s = pyco2.sys().solve()

# Some calculations with one parameter
co2s = pyco2.sys(fco2=[400, 450])

# Adjust to different temperature/pressure
co2s_insitu = (
    pyco2.sys(
        dic=2100,
        fco2=400,
        t=25,
        # p=0,
        # s=30,
        # tsi=20,
        # tp=1,
    )
    .adjust(t=5)
    .solve("fco2")
)

# Adjust fCO2 etc to different temperature
co2s = pyco2.sys(fco2=400, t=25).adjust(t=5).solve("fco2")

# pyco2.sys(dic=2100, ta=2250).adjust(t=15, p=1000).solve("pH")  # warn bug with __pre terms
pyco2.sys(dic=2100, ta=2250, t=15, p=1000).solve("pH")

# Propagate uncertainties
co2s = (
    pyco2.sys(
        dic=2100,
        ph=8.1,
        salinity=500,
        tsi=5,
        pk1=8,
        total_borate=300,
    )
    .set_u(dic=2, ph=0.005, **pyco2.uncertainty_OEDG18)
    # .solve("pco2")
    .prop("ta")
)
# co2s.plot_graph(prog_graphviz="dot", show_unknown=False, mode="valid")
# pKs coming up orange in state graph?


gatl = glodap.atlantic()
gatl = gatl[["talk", "tco2", "salinity", "temperature", "pressure", "silicate"]]
# %%
# gatl["wtf"] = gatl.tco2.copy()
# gatl = gatl.drop(columns="tco2")
co2s = pyco2.sys(temperature=gatl.temperature)
# ^ should reject Series / DataArrays at this point

# %%
t = xr.DataArray([5, 10, 15], dims="t")
s = xr.DataArray([30, 35, 40], dims="s")
# shortcuts not implemented for xr datasets!!!
ds = xr.Dataset({"temperature": t, "salinity": s})

co2s = pyco2.sys(data=ds).solve("pk1")
