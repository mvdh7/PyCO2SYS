from scipy.io import loadmat
from autograd import elementwise_grad as grad
from numpy import size, unique, full

def CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
        PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS):
    
    # Input conditioning.
    args = [PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN,
            PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS]

    # Determine lengths of input vectors.
    veclengths = [size(arg) for arg in args]
    if size(unique(veclengths)) > 2:
        print('*** INPUT ERROR: Input vectors must all be of same length, ' +
              'or of length 1. ***')
        return

    # Make row vectors of all inputs.
    ntps = max(veclengths)
    args = [full(ntps, arg) if size(arg)==1 else arg.ravel()
            for arg in args]
    (PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT,
        SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS) = args
    SAL = SAL.astype('float64')
    TEMPIN = TEMPIN.astype('float64')
    TEMPOUT = TEMPOUT.astype('float64')
    PRESIN = PRESIN.astype('float64')
    PRESOUT = PRESOUT.astype('float64')
    SI = SI.astype('float64')
    PO4 = PO4.astype('float64')
    
    # EXPERIMENTAL STUFF
    out = [p for p in KSO4CONSTANTS]
    
    return out

# Import input conditions: CO2SYStest.mat was generated in MATLAB using the
# script CO2SYStest.m.
matfile = loadmat('../testing/CO2SYStest.mat')['co2s']
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

test = CO2SYS(PARSin[:, 0], PARSin[:, 1], PAR12combos[:, 0], PAR12combos[:, 1],
              sal, tempin, tempout, presin, presout, phos, si,
              pHscales, K1K2, KSO4)
