import numpy as np

import PyCO2SYS as pyco2
from PyCO2SYS import CO2System

# Prepare test conditions
npts = 100
rng = np.random.default_rng(7)
pH_T_i = rng.normal(size=npts, loc=8, scale=1)
H_T_i = 10.0**-pH_T_i

# Set and get total molinities
temperature = 22.3
salinity = 31.0
total_sulfate = pyco2.salts.total_sulfate_MR66(salinity)
total_fluoride = pyco2.salts.total_fluoride_R65(salinity)
k_HF_free = pyco2.equilibria.p1atm.k_HF_free_PF87(temperature, salinity)
k_HSO4_free = pyco2.equilibria.p1atm.k_HSO4_free_WM13(temperature, salinity)
fH = pyco2.convert.fH_TWB82(temperature, salinity)

# Do pH scale conversions in a loop forwards: Total => Seawater => NBS => Free => Total
H_S_f = H_T_i * pyco2.convert.pH_tot_to_sws(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free
)
H_N_f = H_S_f * pyco2.convert.pH_sws_to_nbs(fH)
H_F_f = H_N_f * pyco2.convert.pH_nbs_to_free(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH
)
H_T_f = H_F_f * pyco2.convert.pH_free_to_tot(total_sulfate, k_HSO4_free)

# Do pH scale conversions in a loop backwards: Total => Free => NBS => Seawater => Total
H_F_b = H_T_i * pyco2.convert.pH_tot_to_free(total_sulfate, k_HSO4_free)
H_N_b = H_F_b * pyco2.convert.pH_free_to_nbs(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH
)
H_S_b = H_N_b * pyco2.convert.pH_nbs_to_sws(fH)
H_T_b = H_S_b * pyco2.convert.pH_sws_to_tot(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free
)

# Do the missing combinations
H_N_m = H_T_i * pyco2.convert.pH_tot_to_nbs(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH
)
H_T_m = H_N_m * pyco2.convert.pH_nbs_to_tot(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free, fH
)
H_S_m = H_F_b * pyco2.convert.pH_free_to_sws(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free
)
H_F_m = H_S_f * pyco2.convert.pH_sws_to_free(
    total_fluoride, total_sulfate, k_HF_free, k_HSO4_free
)


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
    scales = ["pH_total", "pH_sws", "pH_free", "pH_nbs"]
    sys1 = CO2System(pH=8.1, opt_pH_scale=1)
    sys2 = CO2System(pH=sys1["pH_sws"], opt_pH_scale=2)
    sys3 = CO2System(pH=sys1["pH_free"], opt_pH_scale=3)
    sys4 = CO2System(pH=sys1["pH_nbs"], opt_pH_scale=4)
    for sys in [sys1, sys2, sys3, sys4]:
        sys.solve(scales)
    for scale in scales:
        assert np.isclose(sys1[scale], sys2[scale], rtol=0, atol=1e-12)
        assert np.isclose(sys1[scale], sys3[scale], rtol=0, atol=1e-12)
        assert np.isclose(sys1[scale], sys4[scale], rtol=0, atol=1e-12)
        assert np.isclose(sys2[scale], sys3[scale], rtol=0, atol=1e-12)
        assert np.isclose(sys2[scale], sys4[scale], rtol=0, atol=1e-12)
        assert np.isclose(sys3[scale], sys4[scale], rtol=0, atol=1e-12)


test_pH_conversions()
test_pH_conversions_sys()
