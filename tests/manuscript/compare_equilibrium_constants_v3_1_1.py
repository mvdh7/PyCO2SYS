# %%
import warnings

import numpy as np
import pandas as pd

from PyCO2SYS import CO2System

# Import MATLAB results and recalculate with PyCO2SYS
matlab = pd.read_csv(
    "tests/manuscript/results/compare_equilibrium_constants_v3_1_1.csv"
)


def test_equilibrium_constants():
    m_to_p = (
        ("K0", "pk_CO2_1atm"),
        ("K1", "pk_H2CO3"),
        ("K2", "pk_HCO3"),
        ("KW", "pk_H2O"),
        ("KP1", "pk_H3PO4"),
        ("KP2", "pk_H2PO4"),
        ("KP3", "pk_HPO4"),
        ("KF", "pk_HF_free"),
        ("KS", "pk_HSO4_free"),
        ("KSi", "pk_Si"),
        ("KB", "pk_BOH3"),
        ("KNH4", "pk_NH3"),
        ("KH2S", "pk_H2S"),
    )
    svars = [p for m, p in m_to_p]
    for g, group in matlab.groupby(
        ["K1K2CONSTANTS", "pHSCALEIN", "KSO4CONSTANT", "BORON", "KFCONSTANT"]
    ):
        # Set up arguments dicts
        values_in = dict(
            temperature=group.TEMPIN.values,
            pressure=group.PRESIN.values,
            salinity=group.SAL.values,
        )
        values_out = dict(
            temperature=group.TEMPOUT.values,
            pressure=group.PRESOUT.values,
            salinity=group.SAL.values,
        )
        opts = dict(
            opt_k_carbonic=g[0],
            opt_pH_scale=g[1],
            opt_k_HSO4=g[2],
            opt_total_borate=g[3],
            opt_k_HF=g[4],
            opt_gas_constant=3,
        )
        # Deal with GEOSECS and freshwater weirdness
        if g[0] == 6:
            opts.update(
                dict(
                    opt_k_BOH3=2,
                    opt_factor_k_BOH3=2,
                    opt_factor_k_H2CO3=2,
                    opt_factor_k_HCO3=2,
                )
            )
        elif g[0] == 7:
            opts.update(
                dict(
                    opt_fH=2,
                    opt_k_BOH3=2,
                    opt_k_H2O=2,
                    opt_k_phosphate=2,
                    opt_k_Si=2,
                    opt_factor_k_BOH3=2,
                    opt_factor_k_H2CO3=2,
                    opt_factor_k_HCO3=2,
                )
            )
        elif g[0] == 8:
            values_in.update(dict(salinity=0.0))
            values_out.update(dict(salinity=0.0))
            opts.update(
                dict(
                    opt_fH=3,
                    opt_k_H2O=3,
                    opt_factor_k_H2O=2,
                    opt_factor_k_H2CO3=3,
                    opt_factor_k_HCO3=3,
                )
            )
        # Solve under input and output conditions
        sys_in = CO2System(**values_in, **opts)
        sys_in.solve(svars, store_steps=2)
        sys_out = CO2System(**values_out, **opts)
        sys_out.solve(svars, store_steps=2)
        # Compare MATLAB with Python
        for m, p in m_to_p:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pk_matlab_in = np.where(
                    group[m + "input"].values == 0,
                    -999.9,
                    -np.log10(group[m + "input"].values),
                )
                pk_python_in = np.where(sys_in[p] == 0, -999.9, sys_in[p])
                pk_matlab_out = np.where(
                    group[m + "output"].values == 0,
                    -999.9,
                    -np.log10(group[m + "output"].values),
                )
                pk_python_out = np.where(sys_out[p] == 0, -999.9, sys_out[p])
                # These terms are not included when opt_k_carbonic == 6
                if g[0] == 6 and p in [
                    "pk_H2O",
                    "pk_H3PO4",
                    "pk_H2PO4",
                    "pk_HPO4",
                    "pk_Si",
                    "pk_NH3",
                    "pk_H2S",
                ]:
                    pk_python_in[:] = -999.9
                    pk_python_out[:] = -999.9
                # These terms are not included when opt_k_carbonic == 7
                if g[0] == 7 and p in [
                    "pk_NH3",
                    "pk_H2S",
                ]:
                    pk_python_in[:] = -999.9
                    pk_python_out[:] = -999.9
                # These terms are not included when opt_k_carbonic == 8
                if g[0] == 8 and p in [
                    "pk_H3PO4",
                    "pk_H2PO4",
                    "pk_HPO4",
                    "pk_Si",
                    "pk_BOH3",
                    "pk_NH3",
                    "pk_H2S",
                ]:
                    pk_python_in[:] = -999.9
                    pk_python_out[:] = -999.9
            assert np.all(
                np.isclose(
                    pk_matlab_in,
                    pk_python_in,
                    rtol=1e-12,
                    atol=1e-16,
                )
            )
            assert np.all(
                np.isclose(
                    pk_matlab_out,
                    pk_python_out,
                    rtol=1e-12,
                    atol=1e-16,
                )
            )


def test_total_salts():
    m_to_p = (
        ("TS", "total_sulfate"),
        ("TF", "total_fluoride"),
        ("TB", "total_borate"),
    )
    svars = [p for m, p in m_to_p]
    for g, group in matlab.groupby(["K1K2CONSTANTS", "BORON"]):
        # Set up arguments dicts
        values = dict(salinity=group.SAL.values)
        opts = dict(
            opt_k_carbonic=g[0],
            opt_total_borate=g[1],
        )
        # Deal with GEOSECS and freshwater weirdness
        if g[0] == 6:
            opts.update(dict(opt_total_borate=4))
        elif g[0] == 7:
            opts.update(dict(opt_total_borate=4))
        # Solve
        sys = CO2System(**values, **opts)
        sys.solve(svars)
        # Compare MATLAB with Python
        for m, p in m_to_p:
            python = sys[p]
            # These terms are not included when opt_k_carbonic == 8
            if g[0] == 8 and p in ["total_sulfate", "total_fluoride", "total_borate"]:
                python[:] = 0.0
            assert np.all(np.isclose(group[m].values, python, rtol=1e-12, atol=1e-16))


# test_equilibrium_constants()
# test_total_salts()
