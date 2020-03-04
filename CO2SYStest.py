import numpy as np
from scipy.io import loadmat
from PyCO2SYS import CO2SYS
from time import time

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

P1 = PARSin[:, 0]
P2 = PARSin[:, 1]
P1type = PAR12combos[:, 0]
P2type = PAR12combos[:, 1]

# Try subbing in a carbonate ion value - works!
carb = 200.0
replacevar = 1
P1[P1type == replacevar] = carb
P2[P2type == replacevar] = carb
P1type[P1type == replacevar] = 6
P2type[P2type == replacevar] = 6
    
# Run CO2SYS in Python
go = time()
co2py = CO2SYS(P1, P2, P1type, P2type, sal, tempin, tempout, presin, presout,
               si, phos, 0, 0, pHscales, K1K2, KSO4)[0]
print('PyCO2SYS runtime = {} s'.format(time() - go))

# Compare with MATLAB - see results in co2maxdiff
nomatvars = ['NH3Alkin', 'H2SAlkin', 'NH3Alkout', 'H2SAlkout',
             'KNH3input', 'KH2Sinput', 'KNH3output', 'KH2Soutput']
co2mat = {var: matfile[var][0][0].ravel() for var in co2py.keys()
          if var not in nomatvars}
co2diff = {var: co2py[var] - co2mat[var] for var in co2py.keys()
           if var not in nomatvars}
co2maxdiff = {var: np.max(np.abs(co2diff[var])) for var in co2py.keys()
              if var not in nomatvars}
