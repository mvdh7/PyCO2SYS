import numpy as np
from scipy.io import loadmat
from PyCO2SYS import CO2SYS

# Import input conditions: CO2SYStest.mat was generated in MATLAB using the
# script CO2SYStest.m.
matfile = loadmat('testing/CO2SYStest.mat')['co2s']
PARSin = matfile['PARSin'][0][0]
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

# Run CO2SYS in Python
co2py = CO2SYS(PARSin[:, 0], PARSin[:, 1], PAR12combos[:, 0],
               PAR12combos[:, 1], sal, tempin, tempout, presin, presout,
               si, phos, pHscales, K1K2, KSO4)[0]

# Compare with MATLAB
co2mat = {var: matfile[var][0][0].ravel() for var in co2py.keys()}
co2diff = {var: co2py[var] - co2mat[var] for var in co2py.keys()}
co2maxdiff = {var: np.max(np.abs(co2diff[var])) for var in co2py.keys()}
