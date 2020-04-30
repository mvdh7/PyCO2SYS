import PyCO2SYS as pyco2
import numpy as np
from PyCO2SYS import CO2SYS

test = pyco2.api.CO2SYS_wrap(co2aq=np.linspace(1500, 2500, 11), dic=2000)
TempC = np.array([25.0])
Pdbar = np.array([0.0])
Sal = np.array([35.0])
TSi = np.array([0.0])
TP = np.array([0.0])
TNH3 = np.array([0.0])
TH2S = np.array([0.0])
WhichKs = np.array([10])
pHScale = np.array([3])
WhoseTB = np.array([2])
WhoseKSO4 = np.array([1])
WhoseKSO4s = np.array([3])  # depends on WhoseTB and WhoseKSO4
WhoseKF = np.array([1])
co2d = CO2SYS(2000, 2100, 1, 2, Sal, TempC, TempC, Pdbar, Pdbar, TSi, TP, pHScale,
              WhichKs, WhoseKSO4s)

#%%
totals = pyco2.salts.assemble(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB)
Ks = pyco2.equilibria.assemble(TempC, Pdbar, Sal, totals, pHScale, WhichKs, WhoseKSO4, 
                               WhoseKF)
# TC = np.array([0.0])
# pH = np.array([14.0])
# FREEtoTOT = pyco2.convert.free2tot(totals['TSO4'], Ks['KSO4'])
# alkparts = pyco2.solve.get.AlkParts(TC, pH, FREEtoTOT, Ks, totals)


TA = co2d['TAlk']*1e-6
TC = co2d['TCO2']*1e-6
PH = co2d['pHinFREE']  # -np.log10(co2d['Hfreein']*1e-6)
FC = co2d['fCO2in']*1e-6
CO2 = co2d['CO2in']*1e-6
CO3 = co2d['CO3in']*1e-6
HCO3 = co2d['HCO3in']*1e-6
OH = co2d['OHin']*1e-6
BAlk = co2d['BAlkin']*1e-6
KB = co2d['KBinput']

TempK = pyco2.convert.TempC2K(TempC)
Pbar = pyco2.convert.Pdbar2bar(Pdbar)

eggX = pyco2.buffers.explicit.all_ESM10(TC, TA, CO2, HCO3, CO3, PH, OH, BAlk, KB)
eggA = pyco2.buffers.all_ESM10(TA, TC, PH, CO3, Sal, TempK, Pbar, WhichKs, Ks, totals)
dvar = 'omegaTA'
gX = eggX[dvar]
gA = eggA[dvar]
g0 = pyco2.buffers.omegaTA(TC, PH, CO3, Sal, TempK, Pbar, WhichKs, Ks, totals)
g1 = pyco2.buffers.omegaTA(TC, PH, CO3, Sal, TempK, Pbar, WhichKs, Ks, totals)
print(gX, gA, g0, g1, gA)

#%%
TA = np.array([6.0])
TC = np.array([2300e-6])
phi = pyco2.solve.initialise.fromTC(
        TA, TC, totals["TB"], Ks["K1"], Ks["K2"], Ks["KB"]
    )
ph = pyco2.solve.get.pHfromTATC(TA, TC, Ks, totals)
