import numpy as np

import PyCO2SYS as pyco2


def test_IAP_curves():
    co2s = pyco2.sys(Mg_fraction=[0, 0.1, 0.2])
    co2s.solve(
        [
            "kt_Mg_calcite_25C_1atm_minprep",
            "kt_Mg_calcite_25C_1atm_biogenic",
            "kt_Mg_calcite_25C_1atm_synthetic",
        ],
        store_steps=2,
    )
    #  category 1
    assert np.all(
        np.isclose(
            np.log10(co2s.kt_Mg_calcite_25C_1atm_minprep),
            [-8.51219119, -8.05063159, -7.27890792],
        )
    )
    # category 2
    assert np.all(
        np.isclose(
            np.log10(co2s.kt_Mg_calcite_25C_1atm_biogenic),
            [-8.37659138, -8.30087635, -8.15566151],
        )
    )
    # category 3
    assert np.all(
        np.isclose(
            np.log10(co2s.kt_Mg_calcite_25C_1atm_synthetic),
            [-8.50230163, -8.43921642, -8.27062648],
        )
    )


def test_temperature_correction():
    # TODO add ideal_mix tests
    # which temperature correction leads to higher solubility at same T?
    co2s = pyco2.sys(Mg_fraction=0.15, temperature=[20, 30], opt_Mg_calcite_type=3)
    co2s.solve(
        ["kt_Mg_calcite_1atm_vantHoff", "kt_Mg_calcite_1atm_PB82"], store_steps=2
    )
    assert np.log10(co2s.kt_Mg_calcite_1atm_vantHoff[0]) > np.log10(
        co2s.kt_Mg_calcite_1atm_PB82[0]
    )
    assert np.log10(co2s.kt_Mg_calcite_1atm_vantHoff[1]) < np.log10(
        co2s.kt_Mg_calcite_1atm_PB82[1]
    )
    # how does van't Hoff compare to the PB82 calcite line?
    co2s = pyco2.sys(Mg_fraction=0.03, temperature=[17, 21], opt_Mg_calcite_type=3)
    co2s.solve(
        ["kt_Mg_calcite_1atm_vantHoff", "kt_Mg_calcite_1atm_PB82"], store_steps=2
    )
    TempK = 17 + 273.15
    assert np.log10(
        co2s.kt_Mg_calcite_1atm_vantHoff[0]
    ) > -171.9065 - 0.077993 * TempK + 2839.319 / TempK + 71.595 * np.log10(TempK)
    TempK = 21 + 273.15
    assert np.log10(
        co2s.kt_Mg_calcite_1atm_vantHoff[1]
    ) < -171.9065 - 0.077993 * TempK + 2839.319 / TempK + 71.595 * np.log10(TempK)
    # same value at 298.15K?
    co2s = pyco2.sys(
        Mg_fraction=np.arange(0, 0.2, 0.02), temperature=[25], opt_Mg_calcite_type=2
    )
    co2s.solve(
        ["kt_Mg_calcite_1atm_vantHoff", "kt_Mg_calcite_1atm_PB82"], store_steps=2
    )
    assert np.all(
        np.isclose(co2s.kt_Mg_calcite_1atm_vantHoff, co2s.kt_Mg_calcite_1atm_PB82)
    )


def test_activity_coefficients():
    # activity coefficients for the tests were calculated with Pytzer v0.6.0
    co2s = pyco2.sys(
        dic=2000,
        alkalinity=2200,
        opt_Mg_calcite_type=2,
        Mg_fraction=0.2,
        opt_Mg_calcite_kt_Tdep=2,
        temperature=[0, 30, 0, 30],
        salinity=[20, 20, 40, 40],
    )
    co2s.solve(["saturation_Mg_calcite", "saturation_calcite"], store_steps=2)
    # activity coefficients
    assert np.all(np.isclose(co2s.acf_Ca, [0.2248, 0.2154, 0.1853, 0.1811], atol=5e-04))
    assert np.all(np.isclose(co2s.acf_Mg, [0.2458, 0.2260, 0.2189, 0.1956], atol=5e-04))
    assert np.all(
        np.isclose(co2s.acf_CO3, [0.0781, 0.0622, 0.0462, 0.0374], atol=5e-04)
    )
    # do I get similarly good/bad results as [C23]?
    TempK = 273.15
    k_t = 10 ** (
        -171.9065 - 0.077993 * TempK + 2839.319 / TempK + 71.595 * np.log10(TempK)
    )
    assert np.isclose(
        np.log(k_t / (co2s.acf_Ca[0] * co2s.acf_CO3[0])),
        np.log(co2s.k_calcite[0]),
        atol=0.03,
    )
    TempK = 303.15
    k_t = 10 ** (
        -171.9065 - 0.077993 * TempK + 2839.319 / TempK + 71.595 * np.log10(TempK)
    )
    assert np.isclose(
        np.log(k_t / (co2s.acf_Ca[3] * co2s.acf_CO3[3])),
        np.log(co2s.k_calcite[3]),
        atol=0.11,
    )


def test_pressure_correction():
    co2s = pyco2.sys(
        dic=2100,
        alkalinity=2200,
        opt_Mg_calcite_type=3,
        Mg_fraction=0,
        opt_Mg_calcite_kt_Tdep=2,
        temperature=5,
        salinity=35,
        pressure=[0, 1000, 2000, 3000, 4000, 5000, 6000],
    )
    co2s.solve(["saturation_Mg_calcite", "saturation_calcite"], store_steps=2)
    # saturation state is decreasing with pressure
    assert np.all(
        co2s.saturation_Mg_calcite == sorted(co2s.saturation_Mg_calcite, reverse=True)
    )
    # similar to calcite?
    assert np.all(
        np.isclose(co2s.saturation_Mg_calcite, co2s.saturation_calcite, rtol=0.1)
    )


test_IAP_curves()
test_temperature_correction()
test_activity_coefficients()
test_pressure_correction()
