# %%
import functools
from inspect import signature

import jax
from jax import numpy as np

import PyCO2SYS as pyco2
from decorators import decorator

validity = {}


@decorator
def set_validity(func, temperature=None, *args, **kwargs):
    if temperature is not None:
        validity[func.__name__] = temperature
    return func(*args, **kwargs)


def _get_kval(t, s):
    return 50 + 3 * t - 0.1 * s**2


# get_kval = tweak_args(_get_kval)


# @set_validity(temperature=[5, 25])
def get_kval(t, s):
    return 50 + 3 * t - 0.1 * s**2


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
print(get_kval.__code__.co_varnames)
# print(get_kval.__signature__.parameters.keys())
print(signature(get_kval).parameters.keys())
print(validity)
print("-----")

co2s = pyco2.sys()
co2s.solve()
