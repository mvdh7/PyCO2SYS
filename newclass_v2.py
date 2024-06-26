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
        dic=np.linspace(2001, 2100, 100000),
        alkalinity=np.linspace(2201, 2300, 100000),
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
        # "fugacity_factor",
        # "pCO2",
        # "CO2",
        # "xCO2",
        # "dic",
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
print(sys.values)

#%%
def egrad(g):
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        x_bar, = g_vjp(np.ones_like(y))
        return x_bar
    return wrapped



def test(a):
    return 3 * a**2

a = np.array([[1, 2, 3], [4, 2., 4]])
da = egrad(test)(a).__array__()
