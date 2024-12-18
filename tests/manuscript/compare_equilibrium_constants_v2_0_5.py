import warnings

import numpy as np
import pandas as pd

import PyCO2SYS as pyco2
from PyCO2SYS import CO2System

# Import MATLAB results and recalculate with PyCO2SYS
matlab = pd.read_csv(
    "tests/manuscript/results/compare_equilibrium_constants_v2_0_5.csv"
)
matlab["opt_k_bisulfate"], matlab["opt_total_borate"] = pyco2.convert.options_old2new(
    matlab.KSO4CONSTANTS.values
)


def test_equilibrium_constants():
    m_to_p = (
        ("K0", "k_CO2_1atm"),
        ("K1", "k_H2CO3"),
        ("K2", "k_HCO3"),
        ("KW", "k_H2O"),
        ("KP1", "k_H3PO4"),
        ("KP2", "k_H2PO4"),
        ("KP3", "k_HPO4"),
        ("KF", "k_HF_free"),
        ("KS", "k_HSO4_free"),
        ("KSi", "k_Si"),
        ("KB", "k_BOH3"),
    )
    svars = [p for m, p in m_to_p]
    for g, group in matlab.groupby(
        ["K1K2CONSTANTS", "pHSCALEIN", "opt_k_bisulfate", "opt_total_borate"]
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
            opt_gas_constant=1,
        )
        # Deal with GEOSECS and freshwater weirdness
        if g[0] == 6:
            opts.update(
                dict(
                    opt_k_BOH3=2,
                    opt_factor_k_BOH3=2,
                    opt_factor_k_H2CO3=3,
                    opt_factor_k_HCO3=3,
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
                    opt_factor_k_H2CO3=3,
                    opt_factor_k_HCO3=3,
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
                    opt_factor_k_H2CO3=2,
                    opt_factor_k_HCO3=2,
                )
            )
        # Solve under input and output conditions
        sys_in = CO2System(values=values_in, opts=opts)
        sys_in.solve(svars)
        sys_out = CO2System(values=values_out, opts=opts)
        sys_out.solve(svars)
        # Compare MATLAB with Python
        for m, p in m_to_p:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                pk_matlab_in = np.where(
                    group[m + "input"].values == 0,
                    -999.9,
                    -np.log10(group[m + "input"].values),
                )
                pk_python_in = np.where(
                    sys_in.values[p] == 0, -999.9, -np.log10(sys_in.values[p])
                )
                pk_matlab_out = np.where(
                    group[m + "output"].values == 0,
                    -999.9,
                    -np.log10(group[m + "output"].values),
                )
                pk_python_out = np.where(
                    sys_out.values[p] == 0, -999.9, -np.log10(sys_out.values[p])
                )
                # These terms are not included when opt_k_carbonic == 6
                if g[0] == 6 and p in [
                    "k_H2O",
                    "k_H3PO4",
                    "k_H2PO4",
                    "k_HPO4",
                    "k_Si",
                ]:
                    pk_python_in[:] = -999.9
                    pk_python_out[:] = -999.9
                # These terms are not included when opt_k_carbonic == 8
                if g[0] == 8 and p in [
                    "k_H3PO4",
                    "k_H2PO4",
                    "k_HPO4",
                    "k_Si",
                    "k_BOH3",
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
    for g, group in matlab.groupby(["K1K2CONSTANTS", "opt_total_borate"]):
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
        sys = CO2System(values=values, opts=opts)
        sys.solve(svars)
        # Compare MATLAB with Python
        for m, p in m_to_p:
            python = sys.values[p]
            # These terms are not included when opt_k_carbonic == 8
            if g[0] == 8 and p in ["total_sulfate", "total_fluoride", "total_borate"]:
                python[:] = 0.0
            assert np.all(np.isclose(group[m].values, python, rtol=1e-12, atol=1e-16))


# test_equilibrium_constants()
# test_total_salts()
