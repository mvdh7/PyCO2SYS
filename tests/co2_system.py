# %%
# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
from PyCO2SYS.classes.function_graph import FunctionGraph


class CO2System(FunctionGraph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


co2s = CO2System()
# TODO (18 April): build up the dict of funcs!
