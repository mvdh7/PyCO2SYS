import PyCO2SYS as pyco2
from PyCO2SYS import CO2System, system
import networkx as nx
import numpy as np

sys = CO2System(
    dict(
        salinity=np.vstack([30, 35, 40]),
        pressure=1000,
        dic=np.linspace(2001, 2100, 1000),
        pH=8.1,
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
        # "total_sulfate",
        # "k_HSO4_free_1atm",
        # "ionic_strength",
        # "k_HF_free_1atm",
        # "total_to_sws_1atm",
        # "nbs_to_sws",
        # "k_H2S_sws_1atm",
        # 'gas_constant',
        # "factor_k_H2S",
        # "k_H2S_sws",
        # "sws_to_opt",
        # "k_H2O",
        # "k_H3PO4",
        # "k_H2PO4",
        # "k_HPO4",
        # "k_NH3",
        # "k_HCO3_sws_1atm",
        # "k_H2CO3",
        # "k_HCO3",
        # "fCO2",
        # "HCO3",
        # "CO3",
        # "OH", 'H_free',
        # "H4SiO4", "H3SiO4",
        # "HSO4", "SO4",
        # "HF", "F",
        # "NH3", "NH4",
        # "H2S", "HS",
        # "alkalinity",
        "fugacity_factor",
        "pCO2",
        "CO2",
        "xCO2",
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
print(sys.values)
