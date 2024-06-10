import PyCO2SYS as pyco2
from PyCO2SYS import CO2System, system
import networkx as nx

sys = CO2System(
    # dict(salinity=32),
    opts=dict(opt_k_HF=2),
    # use_default_values=False,
)
sys.get(
    (
        # "total_sulfate",
        # "k_HSO4_free_1atm",
        # "ionic_strength",
        # "k_HF_free_1atm",
        # "total_to_sws_1atm",
        # "nbs_to_sws",
        "k_H2S_sws_1atm",
        # 'gas_constant',
        "factor_k_H2S",
    )
)
sys.plot_graph(
    # show_unknown=False,
    # show_isolated=False,
    # prog_graphviz='neato',
)

print(sys.values)
