import PyCO2SYS as pyco2
import numpy as np
import pandas as pd

par1 = 2300
par1type = 1
par2 = 2100
par2type = 2
sal = 34
temp = 12.3
pres = 35
si = 23
phos = 3
pHscale = 1
k1k2 = 10
kso4 = 3
nh3 = 10
h2s = 12

rr, dd = pyco2.test.roundrobin(par1, par2, par1type, par2type, sal, temp, pres,
                               si, phos, pHscale, k1k2, kso4, NH3=nh3, H2S=h2s)

rr = pd.DataFrame(rr)
dd = pd.DataFrame(dd)
