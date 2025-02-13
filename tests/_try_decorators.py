# %%
import warnings
from functools import wraps
from inspect import signature

import jax
from jax import numpy as np

import PyCO2SYS as pyco2
from PyCO2SYS.meta import valid


@valid(test=3)
def get_kval(t, s):
    return 50 + 3 * t - 0.1 * s**2


# get_lval.temperature = [5, 25]


args = (25.0, 35.0)
kval = get_kval(*args)
kval_jit = jax.jit(get_kval)(*args)
kval_grad_0 = jax.grad(get_kval)(*args)
kval_grad_1 = jax.grad(get_kval, argnums=1)(*args)
print(kval)
print(kval_jit)
print(kval_grad_0)
print(kval_grad_1)
print(get_kval.__name__)
print(signature(get_kval).parameters.keys())
print(get_kval.valid)
print("-----")

co2s = pyco2.sys()
co2s.solve("k_CO2")
co2s.plot_graph(
    show_isolated=False,
    show_unknown=False,
    prog_graphviz="dot",
)
