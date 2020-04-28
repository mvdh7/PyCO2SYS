import PyCO2SYS as pyco2
import numpy as np
from PyCO2SYS import CO2SYS

test = pyco2.api.CO2SYS_wrap(pH=np.linspace(8.0, 14.0, 13), dic=2300)
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
co2d = CO2SYS(2300, 2100, 1, 2, 35, 20, 5, 0, 0, 0, 0, 3, 1, 1)

#%%
totals = pyco2.salts.assemble(Sal, TSi, TP, TNH3, TH2S, WhichKs, WhoseTB)
Ks = pyco2.equilibria.assemble(TempC, Pdbar, Sal, totals, pHScale, WhichKs, WhoseKSO4, 
                               WhoseKF)
TC = np.array([0.0])
pH = np.array([14.0])
FREEtoTOT = pyco2.convert.free2tot(totals['TSO4'], Ks['KSO4'])
alkparts = pyco2.solve.get.AlkParts(TC, pH, FREEtoTOT, Ks, totals)

#%%
TA = np.array([6.0])
TC = np.array([2300e-6])
phi = pyco2.solve.initialise.fromTC(
        TA, TC, totals["TB"], Ks["K1"], Ks["K2"], Ks["KB"]
    )
ph = pyco2.solve.get.pHfromTATC(TA, TC, Ks, totals)
