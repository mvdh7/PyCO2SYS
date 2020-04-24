from autograd import elementwise_grad as egrad
from autograd import numpy as np
anp = np
import numpy as onp
# from jax import numpy as np
# from jax import grad, vmap
import PyCO2SYS as pyco2
from PyCO2SYS import CO2SYS
# from scipy.misc import derivative

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
# jaxout = grad(whoop)(varin)

# def sum_logistic(x):
#   return np.sum(1.0 / (1.0 + np.exp(-x)))

# x_small = np.arange(3.)
# derivative_fn = grad(sum_logistic)
# print(derivative_fn(x_small))

npts = 1000
Sal = onp.full(npts, 32.3)
WhichKs = onp.full(npts, 8)
WhoseTB = onp.full(npts, 2)
TempC = onp.full(npts, 25.)
Pdbar = onp.full(npts, 5000.)
pHScale = onp.full(npts, 3)
WhoseKSO4 = onp.full(npts, 1)
WhoseKF = onp.full(npts,  1)
TP = onp.full(npts, 0.5e-6)
TSi = onp.full(npts, 5e-6)
TNH3 = onp.full(npts, 1e-6)
TH2S = onp.full(npts, 1e-6)
TA = onp.full(npts, 2300e-6)
TC = onp.full(npts, 2000e-6)

totals = pyco2.salts.assemble(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB)[-1]
_, FugFac, _, Ks = pyco2.equilibria.assemble(TempC, Pdbar, Sal, totals, pHScale,
                          WhichKs, WhoseKSO4, WhoseKF)


caniph = np.array([8.0])
canidic = np.array([2100e-6])
FREEtoTOT = pyco2.convert.free2tot(totals["TSO4"], Ks["KSO4"])
cani = pyco2.solve.get.TAfromTCpH(canidic, caniph, Ks, totals)

alkparts = pyco2.solve.get.AlkParts(caniph, canidic, FREEtoTOT, **Ks, **totals)
HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = alkparts

#%%
# TB, TF, TS = pyco2.assemble.concentrations(Sal, WhichKs, WhoseTB)
# K0, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S, fH = \
#     pyco2.assemble.equilibria(TempC, Pdbar, pHScale, WhichKs, WhoseKSO4,
#                               WhoseKF, TP, TSi, Sal, TF, TS)

# phargs = (TA, TC, K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
#           TB, TF, TS, TP, TSi, TNH3, TH2S)
# ph = pyco2.solve.pHfromTATC(*phargs)

# tc = pyco2.solve.TCfromTApH(TA, ph, *phargs[2:])
# tcg = egrad(pyco2.solve.TCfromTApH)(TA, ph, *phargs[2:])

# phag = egrad(pyco2.solve.pHfromTATC, argnum=0)(*phargs)
# phdg = derivative(lambda TA: pyco2.solve.pHfromTATC(TA, TC,
#     K1, K2, KW, KB, KF, KS, KP1, KP2, KP3, KSi, KNH3, KH2S,
#     TB, TF, TS, TP, TSi, TNH3, TH2S), TA, dx=1e-9)

# from PyCO2SYS.original import CO2SYS
co2d = CO2SYS(TA*1e6, TC*1e6, 1, 2, Sal, 15, 26, 0, Pdbar, TSi*1e6,
              TP*1e6, 3, 10, 3, NH3=TNH3*1e6, H2S=TH2S*1e6, KFCONSTANT=1)
print(co2d['pHinSWS'][0])
print(co2d['pHoutSWS'][0])
print(co2d['RFin'][0])
print(co2d['RFout'][0])
co2e = CO2SYS(co2d['fCO2in'][0], co2d['CO3in'][0], 5, 6, Sal, 15, 26, 0, Pdbar, TSi*1e6,
              TP*1e6, 2, 10, 3, NH3=TNH3*1e6, H2S=TH2S*1e6, KFCONSTANT=1)
print(co2e['TCO2'][0])
# clc = pyco2.solubility.aragonite(Sal, TempC, Pdbar, TC, ph, WhichKs, K1, K2)
# clcg = egrad(pyco2.solubility.aragonite)(Sal, TempC, Pdbar, TC, ph, WhichKs, K1, K2)
#%%
co2co2 = CO2SYS(co2e['HCO3in'][0], co2e['TAlk'][0], 7, 1, Sal, 15, 26, 0, Pdbar, TSi*1e6,
              TP*1e6, 2, 10, 3, NH3=TNH3*1e6, H2S=TH2S*1e6, KFCONSTANT=1)
print(co2co2['TCO2'][0])

#%%
tmpo = 26.0

def gtest(tmpo):
    return CO2SYS(TA*1e6, TC*1e6, 1, 2, Sal, 15.0, tmpo, 0.0, Pdbar, TSi*1e6,
              TP*1e6, 3, 10, 3)['pHoutSWS'][0]

omg = egrad(gtest)(26.0)
print(omg)

# # From 2 to 6
# TA, TC, PH, PC, FC, CARB = pyco2.solve.from2to6(p1, p2, K0, Ks, Ts, TA, TC, PH,
#     PC, FC, CARB, PengCorrection, FugFac)

# # Assemble concentrations
# conc = lambda Sal: pyco2.assemble.concs_TB(Sal, WhichKs, WhoseTB)
# # print(conc(Sal)[0]*1e6)
# # print(egrad(conc)(Sal)[0])

# # Assemble equilibria
# eq = lambda TS: pyco2.assemble.equilibria(TempC, Pdbar, pHScale, WhichKs,
#     WhoseKSO4, WhoseKF, TP, TSi, Sal, TF, TS)[1]
# print(egrad(eq)(TS)[0])

# why nan?
aa = anp.array
test = egrad(lambda *args: pyco2.equilibria.pressured.KC(*args)[0], argnum=0)(
    aa([298.15]), aa([35.0]), aa([10.0]), aa([6]), aa([1.0]), aa([1.0]))
print(' ')
print(test)

pcxargs = (aa([298.15]), aa([10.0]), aa([6]))
test3 = egrad(pyco2.equilibria.pcx.K1fac)(*pcxargs)
test4 = pyco2.equilibria.pcx.K1fac(*pcxargs)
print(test4)
print(test3)

#%% Munhoven
ta = co2d['TAlk'][:3]*1e-6
tc = co2d['TCO2'][:3]*1e-6
tb = co2d['TB'][:3]*1e-6
k1 = co2d['K1input'][:3]
k2 = co2d['K2input'][:3]
kB = co2d['KBinput'][:3]
ph = co2d['pHinFREE'][:3]

test = pyco2.solve.initialise.fromTC(ta, tc, tb, k1, k2, kB)
print(' ')
print(test)

#%% Myhoven
from autograd.numpy import sqrt, log10, where

bicarb = co2d['HCO3in'][:3]*1e-6

myh = pyco2.solve.initialise.fromHCO3(ta, bicarb, tb, k1, k2, kB)

print(myh)
