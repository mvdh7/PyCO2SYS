import PyCO2SYS as pyco2
from autograd import numpy as np
from autograd import elementwise_grad as egrad

inputs = dict(
    TA = np.array([2300]),
    TC = np.array([2150, np.nan, np.nan, 2150, np.nan, np.nan]),
    PH = np.array([np.nan, 7.82, np.nan, np.nan, 7.82, np.nan]),
    PC = np.array([np.nan, np.nan, 970, np.nan, np.nan, 970]),
    FC = np.nan,
    CARB = np.nan,
    HCO3 = np.nan,
    CO2 = np.nan,
    TempC = 25,
    Pdbar = 10,
    SAL = 35,
    WhichKs = 1,
    WhoseKSO4 = 1,
    WhoseKF = 1,
    TSi = 0,
    TPO4 = 0,
    TNH3 = 0,
    TH2S = 0,
    WhoseTB = 2,
    pHScale = 3,
    Icase = np.array([12, 13, 14, 12, 13, 14]),
)
inputs = pyco2.engine.inputs(inputs)[0]
(
    TA, TC, PH, PC, FC, CARB, HCO3, CO2, TempC, Pdbar, SAL, WhichKs, WhoseKSO4, WhoseKF,
    TSi, TPO4, TNH3, TH2S, WhoseTB, pHScale, Icase,
) = (inputs[k] for k in inputs.keys())
TA = TA*1e-6
TC = TC*1e-6
PC = PC*1e-6
FC = FC*1e-6
CARB = CARB*1e-6
HCO3 = HCO3*1e-6
FugFac = pyco2.gas.fugacityfactor(TempC, WhichKs)
totals = pyco2.salts.assemble(SAL, TSi, TPO4, TNH3, TH2S, WhichKs, WhoseTB)
Ks = pyco2.equilibria.assemble(TempC, Pdbar, SAL, totals, pHScale, 
                               WhichKs, WhoseKSO4, WhoseKF)

coreargs = (Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFac, Ks, totals)
TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pyco2.solve.core(*coreargs)

TAgrad = egrad(lambda PH: pyco2.solve.core(
    Icase, TA, TC, PH, PC, FC, CARB, HCO3, CO2, FugFac, Ks, totals)[0])(PH)
print(TAgrad)
# this works fine!
