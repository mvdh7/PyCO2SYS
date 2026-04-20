# %%
from warnings import warn

import networkx as nx
import numpy as np

import PyCO2SYS as pyco2
from PyCO2SYS.uncertainty import pKs_OEDG18


co2s = (
    pyco2.sys(dic=2100, ta=2300)
    .set_u_coeffs_from_single(**pKs_OEDG18)
    .prop(["ph", "pco2"])
)


# def get_u_coeffs_from_single(u_single: dict[str, float]) -> dict[str, float]:
#     """Convert a set of single uncertainty values for pKs (e.g., from OEDG18)
#     into the vectors needed for propagation in PyCO2SYS.

#     The lengths of these vectors might be different depending on which
#     parameterisation has been chosen for each pK.

#     Parameters
#     ----------
#     u_single : dict[str, float]
#         The single uncertainty values in the pKs, e.g.:
#         `dict(pk_H2O=0.01)`.


#     Returns
#     -------
#     dict[str, float]
#         The uncertainty values in the pK coefficients.
#     """
#     u_coeffs = {}
#     for k, v in u_single.items():
#         try:
#             u_coeffs["coeffs_" + k] = np.zeros_like(co2s["coeffs_" + k])
#             u_coeffs["coeffs_" + k][-1] = u_single[k]
#         except nx.NetworkXError:
#             warn(f'No coeffs available for "{k}"')
#     return u_coeffs


# u_single = pKs_OEDG18.copy()
# u_coeffs = get_u_coeffs_from_single(u_single)
