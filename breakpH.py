import PyCO2SYS as pyco2
import numpy as np

test = pyco2.api.CO2SYS_wrap(pH=np.linspace(8.0, 14.0, 13), alk=2300)

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
WhoseKF = np.array([1])

#%%
totals = pyco2.salts.assemble(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB)
Ks = pyco2.equilibria.assemble(TempC, Pdbar, Sal, totals, pHScale, WhichKs, WhoseKSO4, 
                               WhoseKF)

TC = np.array([0.0])
pH = np.array([14.0])
FREEtoTOT = pyco2.convert.free2tot(totals['TSO4'], Ks['KSO4'])

alkparts = pyco2.solve.get.AlkParts(TC, pH, FREEtoTOT, Ks, totals)
