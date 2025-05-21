# %%
import itertools

import numpy as np

from PyCO2SYS import CO2System

# Define round-robin test conditions
values = dict(
    salinity=33,
    temperature=22,
    pressure=1234,
    total_silicate=10,
    total_phosphate=1,
    total_ammonia=2,
    total_sulfide=3,
    total_nitrite=4,
    # total_alpha=5,
    # k_alpha=1e-4,
    # total_beta=6,
    # k_beta=1e-8,
)
values_init = dict(alkalinity=2300, dic=2100, **values)
opts = dict(
    opt_k_carbonic=10,
    opt_pH_scale=1,
    opt_total_borate=1,
)
sys_init = CO2System(**values_init, **opts)

# Define parameter types and names and get initial values
partypes = {
    1: "alkalinity",
    2: "dic",
    3: "pH",
    4: "pCO2",
    5: "fCO2",
    6: "CO3",
    7: "HCO3",
    8: "CO2",
    9: "xCO2",
    10: "saturation_calcite",
    11: "saturation_aragonite",
}
sys_init.solve(partypes.values())


def test_round_robin():
    icases = (
        (partypes[k], partypes[j])
        for k, j in itertools.product(partypes, partypes)
        if k != j
        and k < j
        and 100 * k + j not in [405, 408, 508, 409, 509, 610, 611, 809, 1011]
    )
    for par1, par2 in icases:
        # print(" ")
        # print(par1, par2)
        sys = CO2System(
            **{
                par1: sys_init[par1],
                par2: sys_init[par2],
                **values,
            },
            **opts,
        )
        sys.solve(partypes.values())
        for par in partypes.values():
            # print(par, sys_init.values[par], sys.values[par])
            assert np.isclose(
                sys_init[par],
                sys[par],
                rtol=1e-12,
                atol=1e-12,
            ), f"{par1} & {par2} => {par}"


# test_round_robin()
