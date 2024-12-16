import pandas as pd, numpy as np
import PyCO2SYS as pyco2

# Use the EZIO function
filestem = "tests/data/EZIO_input_template"
df = pyco2.ezio(filestem + ".xlsx")
df_cols = [c for c in df.columns if c != "-"]
df_numpy = df[df_cols].to_numpy(dtype=float)
df_saved = pd.read_csv(filestem + "_processed.csv")


def test_ezio_df():
    """Does the EZIO function return a pandas DataFrame with non-NaN values?"""
    assert isinstance(df, pd.DataFrame)
    assert ~np.any(np.isnan(df_numpy))


def test_ezio_output():
    """Does the EZIO function save its results to a file?"""
    assert isinstance(df_saved, pd.DataFrame)
    for col in df_cols:
        assert np.all(np.isclose(df[col].to_numpy(dtype=float), df_saved[col]))


# test_ezio_df()
# test_ezio_output()
