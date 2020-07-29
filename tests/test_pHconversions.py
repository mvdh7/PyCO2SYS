import numpy as np
import PyCO2SYS as pyco2

# Prepare test conditions
npts = 100
pHScale = np.full(npts, 1)  # start on Total scale
pH_T_i = np.random.normal(size=npts, loc=8, scale=1)
H_T_i = 10.0 ** -pH_T_i

# Set and get total molinities
Sal = np.full(npts, 31.0)
TSi = np.full(npts, 12.0)
TPO4 = np.full(npts, 1.5)
TNH3 = np.full(npts, 2.0)
TH2S = np.full(npts, 0.5)
WhichKs = np.full(npts, 10)
WhoseTB = np.full(npts, 2)
totals = pyco2.salts.assemble(Sal, TSi, TPO4, TNH3, TH2S, WhichKs, WhoseTB)
# Set and get equilibrium constants
TempC = np.full(npts, 22.3)
Pdbar = np.full(npts, 100.0)
WhoseKSO4 = np.full(npts, 1)
WhoseKF = np.full(npts, 1)
WhichR = np.full(npts, 1)
equilibria = pyco2.equilibria.assemble(
    TempC, Pdbar, totals, pHScale, WhichKs, WhoseKSO4, WhoseKF, WhichR
)

# Do pH scale conversions in a loop forwards: Total => Seawater => NBS => Free => Total
H_S_f = H_T_i * pyco2.convert.tot2sws(totals, equilibria)
H_N_f = H_S_f * pyco2.convert.sws2nbs(totals, equilibria)
H_F_f = H_N_f * pyco2.convert.nbs2free(totals, equilibria)
H_T_f = H_F_f * pyco2.convert.free2tot(totals, equilibria)

# Do pH scale conversions in a loop backwards: Total => Free => NBS => Seawater => Total
H_F_b = H_T_i * pyco2.convert.tot2free(totals, equilibria)
H_N_b = H_F_b * pyco2.convert.free2nbs(totals, equilibria)
H_S_b = H_N_b * pyco2.convert.nbs2sws(totals, equilibria)
H_T_b = H_S_b * pyco2.convert.sws2tot(totals, equilibria)

# Do the missing combinations
H_N_m = H_T_i * pyco2.convert.tot2nbs(totals, equilibria)
H_T_m = H_N_m * pyco2.convert.nbs2tot(totals, equilibria)
H_S_m = H_F_b * pyco2.convert.free2sws(totals, equilibria)
H_F_m = H_S_f * pyco2.convert.sws2free(totals, equilibria)


def close_enough(a, b):
    return np.all(np.abs(a - b) < 1e-20)


def test_pHconversions():
    assert close_enough(H_T_i, H_T_f)
    assert close_enough(H_T_i, H_T_b)
    assert close_enough(H_T_i, H_T_m)
    assert close_enough(H_S_f, H_S_b)
    assert close_enough(H_S_f, H_S_m)
    assert close_enough(H_N_f, H_N_b)
    assert close_enough(H_N_f, H_N_m)
    assert close_enough(H_F_f, H_F_b)
    assert close_enough(H_F_f, H_F_m)


test_pHconversions()
