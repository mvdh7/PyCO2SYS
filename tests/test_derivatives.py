# There aren't actually any tests in here right now

import PyCO2SYS as pyco2
import numpy as np

# Initialise a co2dict
# - generate all combinations of marine carbonate system parameters
pars = np.array([2300, 2150, 8.1, 400, 400, 350, 1700, 10])
partypes = np.arange(1, 9, 1)
par1type, par2type = np.meshgrid(partypes, partypes)
par1type = par1type.ravel()
par2type = par2type.ravel()
icases = pyco2.solve.getIcase(par1type, par2type, checks=False)
goodpars = ~np.isin(icases, [45, 48, 58]) & ~(par1type == par2type)
par1type = par1type[goodpars]
par2type = par2type[goodpars]
par1 = pars[par1type - 1]
par2 = pars[par2type - 1]
# - set other conditions
sal = 31.4
tempin = 12.2
tempout = 23.1
presin = 848.1
presout = 1509.2
si = 13
phos = 3
h2s = 0.12
nh3 = 0.5
k1k2c = 12
kso4c = 3
phscale = 3
# - get the co2dict
co2dict = pyco2.CO2SYS(
    np.array([par1[0]]),
    par2[0],
    par1type[0],
    par2type[0],
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
    H2S=h2s,
    NH3=nh3,
)
# - propagate the uncertainties
grads_of = "all"
grads_wrt = ["PAR1", "PAR2", "TB", "K1input"]
co2derivs, dxs = pyco2.uncertainty.forward(
    co2dict, grads_of, grads_wrt, totals=None, equilibria_in=None, equilibria_out=None,
)
