import PyCO2SYS as pyco2
from PyCO2SYS import CO2System, system
import networkx as nx
import jax
import numpy as np

# # With old Autograd approach, below code (solve pH from TA and DIC, 3 x 100000)
# # takes ~2.5 s --- new jax approach takes ~25 ms --- 100 x speedup!
# results = pyco2.sys(
#     par1=np.linspace(2001, 2100, 100000),
#     par2=np.linspace(2201, 2300, 100000),
#     par1_type=2,
#     par2_type=1,
#     salinity=np.vstack([30, 35, 40]),
# )['pH']

sys = CO2System(
    dict(
        salinity=np.vstack([30, 35, 40]),
        pressure=1000,
        # dic=np.linspace(2001, 2100, 100),
        alkalinity=np.linspace(2201, 2300, 100),
        fCO2=np.linspace(500, 1000, 100),
        # pH=8.1,
        total_silicate=100,
        total_phosphate=10,
    ),
    opts=dict(
        opt_k_HF=1,
        opt_pH_scale=1,
        opt_gas_constant=3,
        opt_k_BOH3=1,
        opt_factor_k_BOH3=1,
        opt_k_H2O=1,
        opt_factor_k_H2O=1,
        opt_k_phosphate=1,
        opt_k_Si=1,
        opt_k_NH3=1,
        opt_k_carbonic=10,
        opt_factor_k_HCO3=1,
    ),
    # use_default_values=False,
)
sys.get(
    [
        "alkalinity",
        "fCO2",
        "pCO2",
        "xCO2",
        "dic",
        "pH",
    ]
)
sys.plot_graph(
    show_unknown=False,
    show_isolated=False,
    prog_graphviz="neato",
    # exclude_nodes='gas_constant',
    # skip_nodes=['pressure_atmosphere'], #, 'total_to_sws_1atm'],
)

# sys.get()
# print(sys.values)
