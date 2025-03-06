# %%
import jax
from jax import grad
from jax import numpy as np

import PyCO2SYS as pyco2

jax.config.update("jax_enable_x64", True)


def dic_from_ph_alkalinity(ph, alkalinity, temperature):
    return alkalinity * ph / temperature


def pco2_from_dic_alkalinity(dic, alkalinity, temperature):
    return temperature * (dic + alkalinity) / 200


def pco2_from_ph_alkalinity(ph, alkalinity, temperature_0, temperature_1):
    # Can't use this function in final solution - just for testing!
    dic = dic_from_ph_alkalinity(ph, alkalinity, temperature_0)
    return pco2_from_dic_alkalinity(dic, alkalinity, temperature_1)


# Set values
ph = 8.1
alkalinity = 2300.5
temperature_0 = 10.5
temperature_1 = 25.0

# Set uncertainties
u_ph = 0.01
u_alkalinity = 2.0
u_temperature_0 = 0.005
u_temperature_1 = 0.005

# Initial calculations
dic = dic_from_ph_alkalinity(ph, alkalinity, temperature_0)
pco2 = pco2_from_dic_alkalinity(dic, alkalinity, temperature_1)
pco2_direct = pco2_from_ph_alkalinity(ph, alkalinity, temperature_0, temperature_1)
print(dic, pco2, pco2_direct)

# Get gradients
d_dic = grad(dic_from_ph_alkalinity, argnums=(0, 1, 2))(ph, alkalinity, temperature_0)
print([d.item() for d in d_dic])

#
co2s = (
    pyco2.sys(pH=ph, alkalinity=alkalinity, temperature=temperature_0)
    .set_uncertainty(pH=0.01, alkalinity=2)
    .solve(["dic", "alkalinity"])
    .propagate()
)

# %%
testfunc = co2s._get_func_of("k_H2CO3")


def make_positional(get_value_of):
    assert hasattr(get_value_of, "args_list")

    def func_positional(*args):
        kwargs = {k: v for k, v in zip(get_value_of.args_list, args)}
        return get_value_of(**kwargs)

    func_positional.__doc__ = (
        get_value_of.__doc__.replace("kwargs", "args")
        .replace("dict", "tuple")
        .replace("Key-value pairs for", "Values of")
    )
    func_positional.args_list = get_value_of.args_list
    return func_positional


tf = make_positional(testfunc)

co2a = co2s.adjust(temperature=12)
