import pandas as pd, numpy as np
import PyCO2SYS as pyco2

# Import MATLAB results and recalculate with PyCO2SYS
matlab = pd.read_csv("manuscript/results/compare_equilibrium_constants_v3_2_0.csv")
python = pd.DataFrame(
    pyco2.sys(
        matlab.PAR1.values,
        matlab.PAR2.values,
        matlab.PAR1TYPE.values,
        matlab.PAR2TYPE.values,
        salinity=matlab.SAL.values,
        temperature=matlab.TEMPIN.values,
        temperature_out=matlab.TEMPOUT.values,
        pressure=matlab.PRESIN.values,
        pressure_out=matlab.PRESOUT.values,
        opt_pH_scale=matlab.pHSCALEIN.values,
        opt_gas_constant=3,
        opt_k_carbonic=matlab.K1K2CONSTANTS.values,
        opt_total_borate=matlab.BORON.values,
        opt_k_bisulfate=matlab.KSO4CONSTANT.values,
        opt_k_fluoride=matlab.KFCONSTANT.values,
        total_phosphate=matlab.PO4.values,
        total_silicate=matlab.SI.values,
        total_ammonia=matlab.TNH4.values,
        total_sulfide=matlab.TH2S.values,
        opt_buffers_mode=0,
    )
)


def test_equilibrium_constants():
    for m, p in (
        ("K0input", "k_CO2"),
        ("K0output", "k_CO2_out"),
        ("K1input", "k_carbonic_1"),
        ("K1output", "k_carbonic_1_out"),
        ("K2input", "k_carbonic_2"),
        ("K2output", "k_carbonic_2_out"),
        ("KWinput", "k_water"),
        ("KWoutput", "k_water_out"),
        ("KP1input", "k_phosphoric_1"),
        ("KP1output", "k_phosphoric_1_out"),
        ("KP2input", "k_phosphoric_2"),
        ("KP2output", "k_phosphoric_2_out"),
        ("KP3input", "k_phosphoric_3"),
        ("KP3output", "k_phosphoric_3_out"),
        ("KFinput", "k_fluoride"),
        ("KFoutput", "k_fluoride_out"),
        ("KSinput", "k_bisulfate"),
        ("KSoutput", "k_bisulfate_out"),
        ("KSiinput", "k_silicate"),
        ("KSioutput", "k_silicate_out"),
        ("KBinput", "k_borate"),
        ("KBoutput", "k_borate_out"),
        ("KNH4input", "k_ammonia"),
        ("KNH4output", "k_ammonia_out"),
        ("KH2Sinput", "k_sulfide"),
        ("KH2Soutput", "k_sulfide_out"),
    ):
        pk_matlab = np.where(matlab[m].values == 0, -999.9, -np.log10(matlab[m].values))
        pk_python = np.where(python[p].values == 0, -999.9, -np.log10(python[p].values))
        assert np.all(np.isclose(pk_matlab, pk_python, rtol=1e-12, atol=1e-16))


def test_total_salts():
    for m, p in (
        ("TS", "total_sulfate"),
        ("TF", "total_fluoride"),
        ("TB", "total_borate"),
    ):
        assert np.all(
            np.isclose(matlab[m].values, python[p].values, rtol=1e-12, atol=1e-16)
        )


# test_equilibrium_constants()
# test_total_salts()
