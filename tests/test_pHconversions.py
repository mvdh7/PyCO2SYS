import numpy as np, PyCO2SYS as pyco2

# Prepare test conditions
npts = 100
pHScale = np.full(npts, 1)  # start on Total scale
pH_T_i = np.random.normal(size=npts, loc=8, scale=1)
H_T_i = 10.0**-pH_T_i

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
H_S_f = H_T_i * pyco2.convert.pH_total_to_sws(totals, equilibria)
H_N_f = H_S_f * pyco2.convert.pH_sws_to_nbs(totals, equilibria)
H_F_f = H_N_f * pyco2.convert.pH_nbs_to_free(totals, equilibria)
H_T_f = H_F_f * pyco2.convert.pH_free_to_total(totals, equilibria)

# Do pH scale conversions in a loop backwards: Total => Free => NBS => Seawater => Total
H_F_b = H_T_i * pyco2.convert.pH_total_to_free(totals, equilibria)
H_N_b = H_F_b * pyco2.convert.pH_free_to_nbs(totals, equilibria)
H_S_b = H_N_b * pyco2.convert.pH_nbs_to_sws(totals, equilibria)
H_T_b = H_S_b * pyco2.convert.pH_sws_to_total(totals, equilibria)

# Do the missing combinations
H_N_m = H_T_i * pyco2.convert.pH_total_to_nbs(totals, equilibria)
H_T_m = H_N_m * pyco2.convert.pH_nbs_to_total(totals, equilibria)
H_S_m = H_F_b * pyco2.convert.pH_free_to_sws(totals, equilibria)
H_F_m = H_S_f * pyco2.convert.pH_sws_to_free(totals, equilibria)


def test_pH_conversions():
    assert np.all(np.isclose(H_T_i, H_T_f, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_T_i, H_T_b, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_T_i, H_T_m, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_S_f, H_S_b, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_S_f, H_S_m, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_N_f, H_N_b, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_N_f, H_N_m, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_F_f, H_F_b, rtol=0, atol=1e-20))
    assert np.all(np.isclose(H_F_f, H_F_m, rtol=0, atol=1e-20))


def test_pH_conversions_sys():
    r1 = pyco2.sys(par1=8.1, par1_type=3, opt_pH_scale=1)
    r2 = pyco2.sys(par1=r1["pH_sws"], par1_type=3, opt_pH_scale=2)
    r3 = pyco2.sys(par1=r1["pH_free"], par1_type=3, opt_pH_scale=3)
    r4 = pyco2.sys(par1=r1["pH_nbs"], par1_type=3, opt_pH_scale=4)
    for scale in ["pH_total", "pH_sws", "pH_free", "pH_nbs"]:
        assert np.isclose(r1[scale], r2[scale], rtol=0, atol=1e-12)
        assert np.isclose(r2[scale], r3[scale], rtol=0, atol=1e-12)
        assert np.isclose(r3[scale], r4[scale], rtol=0, atol=1e-12)


test_pH_conversions()
test_pH_conversions_sys()
