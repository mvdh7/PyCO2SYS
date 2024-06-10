from PyCO2SYS import CO2System
import networkx as nx

sys = CO2System(
    # dict(salinity=32),
    # opts=dict(opt_total_borate=2),
    # use_default_values=False,
)
sys.get(("total_borate", "total_sulfate"))
sys.plot_links(
    # show_missing=False,
    # show_isolated=False,
)

print(sys.values)
