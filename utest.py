import PyCO2SYS as pyco2
import numpy as np

par1 = np.array([2150, 2150, 2020, 2000])
par2 = np.array([2300, 2200, 2100, 2400])
par1type = np.array([2, 2, 2, 2])
par2type = 1
sal = 32
tempin = 10
tempout = 20
presin = 0
presout = 1000
si = 10
phos = 3
nh3 = 1
h2s = 0.5
phscale = 1
k1k2c = 10
kso4c = 3
kfc = 1

co2dict = pyco2.CO2SYS(
    par1,
    par2,
    par1type,
    par2type,
    sal,
    tempin,
    tempout,
    presin,
    presout,
    si,
    phos,
    phscale,
    k1k2c,
    kso4c,
    KFCONSTANT=kfc,
    totals={"TB": 100e-6},
    Kis={"K1": 10.0 ** -6},
)

uncertainties, components = pyco2.uncertainty.propagate(
    co2dict,
    {"PAR1": np.array([1.0, 1.0, 2.0, 3.0]), "TEMPIN": 1.0},
    ["CO2out", "OmegaARin", "RFout"],
)
