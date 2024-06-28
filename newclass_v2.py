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
    values=dict(
        salinity=np.vstack([30, 35, 40]),
        pressure=1000,
        # dic=np.linspace(2001, 2100, 10),
        alkalinity=np.linspace(2201, 2300, 10),
        # xCO2=np.linspace(500, 1000, 10),
        # CO3=np.linspace(100, 200, 10),
        # HCO3=np.linspace(1700, 1800, 10),
        pH=8.1,
        # saturation_calcite=1.5,
        # saturation_aragonite=1.5,
        total_silicate=100,
        total_phosphate=10,
    ),
    # values_out=dict(
    #     # temperature=10,
    # ),
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
        opt_HCO3_root=2,
    ),
    # use_default_values=False,
)
# %%
sys.get(
    parameters=[
        # "d_dic__d_pH__alkalinity",
        # "d_alkalinity__d_pH__dic",
        # "d_lnCO2__d_pH__alkalinity",
        # "gamma_dic",
        # "gamma_alkalinity",
        # "beta_dic",
        # "beta_alkalinity",
        # "d_lnOmega__d_CO3",
        # "d_CO3__d_pH__alkalinity",
        # "d_CO3__d_pH__dic",
        # "omega_dic",
        # "omega_alkalinity",
        # "psi",
        "revelle_factor",
        # "alkalinity",
        # "fCO2",
        # "pCO2",
        # "xCO2",
        # "dic",
        # "pH",
        # "HCO3",
        # "CO3",
        # "k_aragonite",
        # "k_calcite",
        # "saturation_calcite",
        # "saturation_aragonite",
        # "pH_free",
        # "substrate_inhibitor_ratio",
    ],
    # parameters_out=["substrate_inhibitor_ratio"],
    # save_steps=False,
)
# %%
sys.plot_graph(
    show_unknown=False,
    show_isolated=False,
    prog_graphviz="neato",
    # exclude_nodes='gas_constant',
    # skip_nodes=['pressure_atmosphere'], #, 'total_to_sws_1atm'],
    conditions="input",
)
if hasattr(sys, "graph_out"):
    sys.plot_graph(
        show_unknown=False,
        show_isolated=False,
        prog_graphviz="neato",
        # exclude_nodes='gas_constant',
        # skip_nodes=['pressure_atmosphere'], #, 'total_to_sws_1atm'],
        conditions="output",
    )

# sys.get()
# print(sys.values)
