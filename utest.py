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
totals = {"TB": 100}
equilibria_in = {"K1": 10.0 ** -6}

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
    totals=totals,
    equilibria_in=equilibria_in,
)

grads_of = ["CO2out", "OmegaARin", "RFout", "TB", "TF", "K1input", "K1output"]
grads_wrt = {
    "PAR1": np.array([1.0, 1.0, 2.0, 3.0]),
    "TEMPIN": 1.0,
    "TB": 1.0,
    "TF": 1.0,
    "SAL": 1.0,
    "K1input": 3.0,
}

co2derivs = pyco2.uncertainty.derivatives(
    co2dict,
    grads_of,
    grads_wrt,
    dx=1e-8,
    totals=totals,
    equilibria_in=equilibria_in,
    equilibria_out=None,
    use_explicit=True,
    verbose=True,
)

uncertainties, components = pyco2.uncertainty.propagate(
    co2dict,
    grads_of,
    grads_wrt,
    totals=totals,
    equilibria_in=equilibria_in,
    equilibria_out=None,
)
