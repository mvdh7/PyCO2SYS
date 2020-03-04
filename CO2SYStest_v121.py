import numpy as np
from scipy.io import loadmat
from PyCO2SYS import CO2SYS
from time import time

# Import input conditions: CO2SYStest.mat was generated in MATLAB using the
# script CO2SYStest.m.
matfile = loadmat('testing/CO2SYStest_v121.mat')['co2s']
P1 = matfile['P1'][0][0]
P2 = matfile['P2'][0][0]
P1type = matfile['P1type'][0][0]
P2type = matfile['P2type'][0][0]
sal = matfile['SAL'][0][0]
tempin = matfile['TEMPIN'][0][0]
tempout = matfile['TEMPOUT'][0][0]
presin = matfile['PRESIN'][0][0]
presout = matfile['PRESOUT'][0][0]
phos = matfile['PO4'][0][0]
si = matfile['SI'][0][0]
nh3 = matfile['NH3'][0][0]
h2s = matfile['H2S'][0][0]
pHscales = matfile['pHSCALEIN'][0][0]
K1K2c = matfile['K1K2CONSTANTS'][0][0]
KSO4c = matfile['KSO4CONSTANT'][0][0]
# KSO4_only = matfile['KSO4_only'][0][0]
KFc = matfile['KFCONSTANT'][0][0]
BSal = matfile['BORON'][0][0]
    
# Run CO2SYS in Python
go = time()
co2py = CO2SYS(P1, P2, P1type, P2type, sal, tempin, tempout, presin, presout,
                si, phos, nh3, h2s, pHscales, K1K2c, KSO4c, KFc, BSal)
print('PyCO2SYS runtime = {} s'.format(time() - go))

# Compare with MATLAB - see results in co2maxdiff
nomat = ['KSO4CONSTANTS']
co2mat = {var: matfile[var][0][0].ravel() for var in co2py.keys()
          if var not in nomat}
co2diff = {var: co2py[var] - co2mat[var] for var in co2py.keys()
           if var not in nomat}
co2maxdiff = {var: np.max(np.abs(co2diff[var])) for var in co2py.keys()
              if var not in nomat}
