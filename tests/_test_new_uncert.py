# %%
import jax
from jax import numpy as np

import PyCO2SYS as pyco2
from PyCO2SYS.engine import make_positional
from PyCO2SYS.meta import egrad
from PyCO2SYS.salts import coeffs_total_sulfate_MR66

co2s = pyco2.sys(
    cf_total_sulfate=coeffs_total_sulfate_MR66(),
).solve("total_sulfate")
# co2s.plot_graph()

coeffs = np.array(coeffs_total_sulfate_MR66())
# salinity = np.array([[30.0, 35, 40], [31, 36, 39]])
salinity = np.array([35.1, 40, 30])

get_tso4 = co2s._get_func_of("total_sulfate")
get_tso4 = co2s._get_func_of_from_wrt(get_tso4, "cf_total_sulfate")
tso4 = get_tso4(coeffs, salinity=salinity)

get_tso4_grad = co2s.get_grad_func("total_sulfate", "cf_total_sulfate")

tso4_grad = egrad(get_tso4)(coeffs, salinity=salinity)
tso4_jac = jax.jacfwd(get_tso4)(coeffs, salinity=salinity)
# tso4_grad = co2s.get_grad("total_sulfate", "cf_total_sulfate")

u_cf_total_sulfate = np.array(
    [
        [0.02, -0.02],
        [-0.02, 1],
    ]
)
u_cf_total_sulfate = np.array(
    [
        [(1e6 * (0.00023 / 96.062) / 1.80655) ** 2, 0],
        [0, 0],
    ]
)
uncert_uncorr = np.sqrt(
    np.sum(
        tso4_jac**2 * np.diag(u_cf_total_sulfate),
        axis=-1,
    )
)  # THIS IS THE WAY!!!  Works for multidimensionals.  DOESN'T include covars.
print(tso4)
print(uncert_uncorr)
uncert = tso4_jac @ u_cf_total_sulfate @ tso4_jac.T
print(np.sqrt(np.diag(uncert)))
# uncert_manual = np.sum(
#     np.array(
#         [
#             np.sum(tso4_jac[0, :] * u_cf_total_sulfate[:, 0]),
#             np.sum(tso4_jac[0, :] * u_cf_total_sulfate[:, 1]),
#         ]
#     )
#     * tso4_jac[0, :]
# )
# uncert_simple = np.sum(
#     np.array(
#         [
#             np.sum(tso4_jac[0, :] * u_cf_total_sulfate[:, 0]),
#             np.sum(tso4_jac[0, :] * u_cf_total_sulfate[:, 1]),
#         ]
#     )
#     * tso4_jac[0, :]
# )
# uncert_flat = np.diag(u_cf_total_sulfate) * tso4_jac**2
# print(uncert)
# print(uncert_manual)
# print(uncert_simple)
# print(uncert_flat)
