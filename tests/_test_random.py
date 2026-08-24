# %%
import glodap

import PyCO2SYS as pyco2


glo = glodap.world(version="v2.2023")
L = (
    glo.tco2.notnull()
    & glo.talk.notnull()
    & glo.temperature.notnull()
    & glo.salinity.notnull()
)
co2s = pyco2.sys(data=glo[L].drop(columns="fco2"), nitrite=0)
