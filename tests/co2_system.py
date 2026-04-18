# %%
# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
import networkx as nx
from jax import numpy as np

from PyCO2SYS.classes.function_graph import (
    FunctionGraph,
    ShortcutDotDict,
    ShortcutsDict,
)
from PyCO2SYS.engine import (
    get_funcs,
    get_funcs_core,
    get_funcs_opts,
    opts_default,
    parameters_core,
    shortcuts,
    values_default,
)
from tests._plotting import plot_graph


class CO2System(FunctionGraph):
    def __init__(
        self,
        defaults: dict | None = None,
        graph: nx.DiGraph | None = None,
        funcs: dict | None = None,
        shortcuts: dict | None = None,
        icase: int = None,
        opts: dict = None,
    ):
        super().__init__(
            defaults=defaults, graph=graph, funcs=funcs, shortcuts=shortcuts
        )
        self.icase = icase
        self.opts = ShortcutDotDict(self.shortcuts)
        self.opts.update(opts)


kwargs = dict(
    dic=2300,
    ta=2400,
    # ph=8.1,
    p=10,
    t=12,
    s=35,
    opt_k_carbonic=19,
)
shortcuts = ShortcutsDict(**shortcuts)
# The below will be def sys(**kwargs)
opts = {k: v for k, v in kwargs.items() if k in opts_default}
opts = opts_default | opts
data = {shortcuts[k]: v for k, v in kwargs.items() if k not in opts_default}
# Get icase
core_known = np.array([v in data for v in parameters_core])
icase_all = np.arange(1, len(parameters_core) + 1)
icase = icase_all[core_known]
if len(icase) > 2:
    raise Exception("A maximum of 2 known core parameters can be provided.")
if len(icase) == 0:
    icase = np.array(0)
elif len(icase) == 2:
    icase = icase[0] * 100 + icase[1]
icase = icase.item()
# Assemble relevant functions
funcs = get_funcs | get_funcs_core[icase]
for opt, v in opts.items():
    # opt_HCO3_root is available only for icase == 207 (known DIC & HCO3)
    if not (opt == "opt_HCO3_root" and icase != 207):
        funcs.update(get_funcs_opts[opt][v])
# If pH is not accessible, we can't calculate it on different scales
if icase < 100 and icase not in [3]:
    pH_vars = ["pH", "pH_total", "pH_sws", "pH_free", "pH_nbs"]
    for v in pH_vars:
        if v in funcs:
            funcs.pop(v)
co2s = CO2System(
    funcs=funcs,
    shortcuts=shortcuts,
    defaults=values_default,
    icase=icase,
    opts=opts,
)
co2s.set_data(**data)
plot_graph(co2s)
