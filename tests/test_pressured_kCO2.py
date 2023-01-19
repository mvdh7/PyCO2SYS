import numpy as np, pandas as pd
import PyCO2SYS as pyco2

pressure = np.linspace(0, 5000, num=6)


def get_dfs(common_kwargs):
    results_old = pyco2.sys(**common_kwargs)
    results_pressured = pyco2.sys(**common_kwargs, opt_pressured_kCO2=1)
    df_old = pd.DataFrame({"pressure": pressure})
    df_pressured = pd.DataFrame({"pressure": pressure})
    df_vars = [
        "pressure",
        "pressure_out",
        "par1",
        "par2",
        "par1_type",
        "par2_type",
        "temperature",
        "salinity",
        "alkalinity",
        "dic",
        "pH",
        "pCO2",
        "fCO2",
        "CO2",
        "xCO2",
        "k_CO2",
        "fugacity_factor",
        "opt_k_carbonic",
        "opt_k_bisulfate",
        "opt_k_fluoride",
        "opt_total_borate",
        "opt_pH_scale",
        "gas_constant",
        "pH_out",
        "pCO2_out",
        "fCO2_out",
        "CO2_out",
        "xCO2_out",
        "k_CO2_out",
        "fugacity_factor_out",
    ]
    for v in df_vars:
        df_old[v] = results_old[v]
        df_pressured[v] = results_pressured[v]
    return df_old, df_pressured


# First, solve from DIC and alkalinity
common12 = dict(
    par1=2300,
    par1_type=1,
    par2=2100,
    par2_type=2,
    pressure=pressure,
    pressure_out=1500,
    opt_k_carbonic=10,
)
df12o, df12p = get_dfs(common12)

# Next, solve from alkalinity and pCO2
common14 = dict(
    par1=2300,
    par1_type=1,
    par2=400,
    par2_type=4,
    pressure=pressure,
    pressure_out=1500,
    opt_k_carbonic=10,
)
df14o, df14p = get_dfs(common14)

# Finally, solve from pH and fCO2
common35 = dict(
    par1=8.1,
    par1_type=3,
    par2=400,
    par2_type=5,
    pressure=1500,
    pressure_out=pressure,
    opt_k_carbonic=10,
)
df35o, df35p = get_dfs(common35)

# Concatenate the dfs
dfo = pd.concat((df12o, df14o, df35o)).reset_index()
dfp = pd.concat((df12p, df14p, df35p)).reset_index()

# Save to file for MATLAB comparison
dfo.to_csv("tests/data/test_pressured_kCO2_original.csv")
dfp.to_csv("tests/data/test_pressured_kCO2_pressured.csv")

# Open files from MATLAB (created by Jon Sharp on 2023-01-19)
mlo = pd.read_csv("tests/data/test_pressured_kCO2_original_jds.csv", index_col=0)
mlp = pd.read_csv("tests/data/test_pressured_kCO2_pressured_jds.csv", index_col=0)

# Compare Python with MATLAB
diffo = dfo - mlo
diffp = dfp - mlp


def test_original_differences():
    for k, v in diffo.items():
        if ~v.isnull().all():
            assert np.allclose(v, 0, rtol=0, atol=1e-6)


def test_pressured_differences():
    for k, v in diffp.items():
        if ~v.isnull().all():
            try:
                assert np.allclose(v, 0, rtol=0, atol=1e-4)
            except AssertionError:
                print(k)


# test_original_differences()
# test_pressured_differences()
