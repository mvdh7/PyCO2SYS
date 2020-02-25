from PyCO2SYS.pyversion import CO2SYS
from scipy.io import loadmat

# Import input conditions: CO2SYStest.mat was generated in MATLAB using the
# script CO2SYStest.m.
matfile = loadmat('testing/CO2SYStest.mat')['co2s']
PARSin = matfile['PARSin'][0][0]
PAR1 = PARSin[:, 0]
PAR2 = PARSin[:, 1]
PAR12combos = matfile['PAR12combos'][0][0]
sal = matfile['SAL'][0][0]
tempin = matfile['TEMPIN'][0][0]
tempout = matfile['TEMPOUT'][0][0]
presin = matfile['PRESIN'][0][0]
presout = matfile['PRESOUT'][0][0]
phos = matfile['PO4'][0][0]
si = matfile['SI'][0][0]
pHscales = matfile['pHSCALEIN'][0][0]
K1K2 = matfile['K1K2CONSTANTS'][0][0]
KSO4 = matfile['KSO4CONSTANTS'][0][0]

co2args = (PAR1, PAR2, PAR12combos[:, 0], PAR12combos[:, 1],
           sal, tempin, tempout, presin, presout, phos, si,
           pHscales, K1K2, KSO4)
test = CO2SYS(*co2args)[0]
# testd = grad(CO2SYS)(*co2args)