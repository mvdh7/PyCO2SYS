import numpy as np
import pandas as pd

from PyCO2SYS import CO2System


def test_pressured_differences():
    # Open file from MATLAB (created by Jon Sharp on 2023-01-19)
    mlp = pd.read_csv("tests/data/test_pressured_kCO2_pressured_jds.csv", index_col=0)
    partypes = {1: "alkalinity", 2: "dic", 3: "pH", 4: "pCO2", 5: "fCO2"}
    opts = dict(opt_k_carbonic=10)
    solve_out = [
        "pH",
        "pCO2",
        "fCO2",
        "CO2",
        "xCO2",
        "k_CO2",
        "fugacity_factor",
    ]
    solve_in = ["alkalinity", "dic", *solve_out]
    for pars, group in mlp.groupby(["par1_type", "par2_type"]):
        print(pars)
        values_pars = {
            partypes[pars[0]]: group.par1.values,
            partypes[pars[1]]: group.par2.values,
        }
        sys_in = CO2System(
            values={
                **values_pars,
                "pressure": group.pressure.values,
                "temperature": group.temperature.values,
                "salinity": group.salinity.values,
            },
            opts=opts,
        )
        sys_in.solve(solve_in)
        sys_out = CO2System(
            values={
                "alkalinity": sys_in.values["alkalinity"],
                "dic": sys_in.values["dic"],
                "pressure": group.pressure_out.values,
                "temperature": group.temperature.values,
                "salinity": group.salinity.values,
            },
            opts=opts,
        )
        sys_out.solve(solve_out)
        for v in solve_in:
            assert np.all(np.abs(group[v].values - sys_in.values[v]) < 1e-6)
        for v in solve_out:
            assert np.all(np.abs(group[v + "_out"].values - sys_out.values[v]) < 1e-4)


# test_pressured_differences()
