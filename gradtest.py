from autograd import elementwise_grad as egrad
from autograd import numpy as anp
import numpy as onp
from jax import numpy as np
from jax import grad, vmap
import PyCO2SYS as pyco2
from PyCO2SYS import CO2SYS
from scipy.misc import derivative

def whoop(varin):
    varsum = np.sum(np.sqrt(varin))
    i = 0
    while i < 10:
        varsum = varsum*varin[1] + varin[0] + 0.5
        i += 1
    return varsum

varin = np.arange(10.0)
varout = whoop(varin)
# gradout = egrad(whoop)(varin)
jaxout = grad(whoop)(np.array([2.0, 2.5]))

# def sum_logistic(x):
#   return np.sum(1.0 / (1.0 + np.exp(-x)))

# x_small = np.arange(3.)
# derivative_fn = grad(sum_logistic)
# print(derivative_fn(x_small))

npts = 10000
Sal = onp.full(npts, 35.)
WhichKs = onp.full(npts, 10)
WhoseTB = onp.full(npts, 2)
TempC = onp.full(npts, 25.)
Pdbar = onp.full(npts, 1.)
pHScale = onp.full(npts, 3)
WhoseKSO4 = onp.full(npts, 1)
WhoseKF = onp.full(npts,  1)
TP = onp.full(npts, 0.5e-6)
TSi = onp.full(npts, 5e-6)
TNH3 = onp.full(npts, 0.1e-6)
TH2S = onp.full(npts, 0.01e-6)
TA = onp.full(npts, 2300e-6)
TC = onp.full(npts, 2000e-6)

TB, TF, TS = pyco2.assemble.concentrations(Sal, WhichKs, WhoseTB)
K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, fH = \
    pyco2.assemble.equilibria(TempC, Pdbar, pHScale, WhichKs, WhoseKSO4,
                              WhoseKF, TP, TSi, Sal, TF, TS)

phargs = (TA, TC, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
          TB, TF, TS, TP, TSi, TNH3, TH2S)
ph = pyco2.solve.pHfromTATC(*phargs)

tc = pyco2.solve.TCfromTApH(TA, ph, *phargs[2:])
tcg = egrad(pyco2.solve.TCfromTApH)(TA, ph, *phargs[2:])

phag = egrad(pyco2.solve.pHfromTATC, argnum=0)(*phargs)
phdg = derivative(lambda TA: pyco2.solve.pHfromTATC(TA, TC,
    K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
    TB, TF, TS, TP, TSi, TNH3, TH2S), TA, dx=1e-9)

co2d = CO2SYS(TA*1e6, TC*1e6, 1, 2, Sal, TempC, TempC, Pdbar, Pdbar, TSi*1e6,
              TP*1e6, 3, 7, 3, NH3=TNH3*1e6, H2S=TH2S*1e6, KFCONSTANT=1)
# print(co2d['OmegaCAin'][0])
# print(co2d['OmegaARin'][0])
clc = pyco2.solubility.aragonite(Sal, TempC, Pdbar, TC, ph, WhichKs, K1, K2)
clcg = egrad(pyco2.solubility.aragonite)(Sal, TempC, Pdbar, TC, ph, WhichKs, K1, K2)

# # From 2 to 6
# TA, TC, PH, PC, FC, CARB = pyco2.solve.from2to6(p1, p2, K0, Ks, Ts, TA, TC, PH,
#     PC, FC, CARB, PengCorrection, FugFac)

# Assemble concentrations
conc = lambda Sal: pyco2.assemble.concs_TB(Sal, WhichKs, WhoseTB)
conco = lambda Sal: pyco2.assemble.concs_TB_original(Sal, WhichKs, WhoseTB)
print(pyco2.assemble.concs_TB_original(Sal, WhichKs, WhoseTB)[0]*1e6)
print(derivative(conco, Sal, dx=1e-5)[0])
print(conc(Sal)[0]*1e6)
print(egrad(conc)(Sal)[0])

# Assemble equilibria
# eq = lambda TempC: pyco2.assemble.equilibria(TempC, Pdbar, pHScale, WhichKs,
#     WhoseKSO4, WhoseKF, TP, TSi, Sal, TF, TS)[1]
# print(egrad(eq)(TempC)[0])
