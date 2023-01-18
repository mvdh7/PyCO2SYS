# Check that the outputs of PyCO2SYS.uncertainty.forward are all dicts of floats
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
goodpars = ~np.isin(icases, [405, 408, 508]) & ~(par1type == par2type)
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
k1k2c = 16
kso4c = 3
phscale = 3
# - get the co2dict
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
    H2S=h2s,
    NH3=nh3,
)
# - propagate the uncertainties
grads_of = "all"
grads_wrt = "all"
co2derivs, dxs = pyco2.uncertainty.forward(
    co2dict,
    grads_of,
    grads_wrt,
    totals=None,
    equilibria_in=None,
    equilibria_out=None,
)


def test_derivs_are_floats():
    assert isinstance(co2derivs, dict)
    for of in co2derivs.values():
        assert isinstance(of, dict)
        for wrt in of.values():
            assert isinstance(wrt, np.ndarray)
            assert isinstance(wrt[0], float)


def test_dxs_are_floats():
    assert isinstance(dxs, dict)
    for dx in dxs.values():
        assert isinstance(dx, float)


# test_derivs_are_floats()
# test_dxs_are_floats()
