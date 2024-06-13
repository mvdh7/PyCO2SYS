import PyCO2SYS as pyco2
from PyCO2SYS import CO2System, system
import networkx as nx

sys = CO2System(
    dict(salinity=32, pressure=1000),
    opts=dict(
        opt_k_HF=2,
        opt_pH_scale=1,
        opt_gas_constant=3,
        opt_k_BOH3=1,
        opt_factor_k_BOH3=1,
        opt_k_H2O=1,
        opt_factor_k_H2O=1,
        opt_k_phosphate=1,
        opt_k_Si=1,
        opt_k_NH3=1,
        opt_k_carbonic=18,
        opt_factor_k_HCO3=3,
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
        "k_CO2",
    ]
)
sys.plot_graph(
    show_unknown=False,
    show_isolated=False,
    prog_graphviz="neato",
)

sys.get()
print(sys.values)
