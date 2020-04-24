import numpy as np
from scipy.io import loadmat
from PyCO2SYS import CO2SYS
from time import time

# Import input conditions: "MATLAB_CO2SYSv1_21.mat" was generated in MATLAB
# using the script "compare_MATLABv1_21.m".
matfile = loadmat('compare/data/MATLAB_CO2SYSv1_21.mat')['co2s']
(P1, P2, P1type, P2type, sal, tempin, tempout, presin, presout, phos, si,
    pHscales, K1K2, KSO4, KF, nh3, h2s) = [matfile[var][0][0] for var in
    ['PAR1', 'PAR2', 'PAR1TYPE', 'PAR2TYPE', 'SAL', 'TEMPIN', 'TEMPOUT',
     'PRESIN', 'PRESOUT','PO4', 'SI', 'pHSCALEIN', 'K1K2CONSTANTS',
     'KSO4CONSTANTS', 'KFCONSTANT', 'NH3', 'H2S']]
co2inputs = [P1, P2, P1type, P2type, sal, tempin, tempout, presin, presout,
             si, phos, pHscales, K1K2, KSO4]
# xrow = 210 # just do one row, or...
xrow = range(len(P1)) # ... do all rows
co2inputs = [inp[xrow] for inp in co2inputs]

# Run CO2SYS in Python
go = time()
co2py = CO2SYS(*co2inputs, NH3=nh3[xrow], H2S=h2s[xrow], KFCONSTANT=KF[xrow])
print('PyCO2SYS runtime = {:.6f} s'.format(time() - go))
# Extract dict output if PyCO2SYS.original was used
if np.shape(co2py) == (4,):
    co2py = co2py[0]

# Prepare MATLAB results to compare
pyvars = ['NH3Alkin', 'NH3Alkout', 'H2SAlkin', 'H2SAlkout', 'KSO4CONSTANT',
          'KFCONSTANT', 'BORON', 'NH3', 'H2S', 'KNH3input', 'KNH3output',
          'KH2Sinput', 'KH2Soutput', 'gammaTCin', 'betaTCin', 'omegaTCin',
          'gammaTAin', 'betaTAin', 'omegaTAin', 'isoQin', 'isoQapprox_in',
          'gammaTCout', 'betaTCout', 'omegaTCout', 'isoQout', 'psi_in',
          'gammaTAout', 'betaTAout', 'omegaTAout', 'isoQapprox_out', 'psi_out']
co2mat = {var: matfile[var][0][0].ravel()[xrow] for var in co2py.keys()
          if var not in pyvars}
# Differences between PyCO2SYS and MATLAB v1.21
co2diff = {var: co2py[var] - co2mat[var] for var in co2mat.keys()}
co2maxdiff = {var: np.max(np.abs(co2diff[var])) for var in co2mat.keys()}
