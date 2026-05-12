# %%
import PyCO2SYS as pyco2
import PyCO2SYS.equilibria.p1atm as eq
from PyCO2SYS.uncertainty import pks_OEDG18


# co2s = pyco2.sys(dic=2100, ta=2300).set_u_coeffs_from_single(**pks_OEDG18)
# co2s.set_u(coeffs_total_borate=(0.02 * co2s.coeffs_total_borate) ** 2)

co2s = pyco2.sys(dic=2100, ta=2300).set_u_OEDG18()
co2s.prop(["ph", "pco2"])

# co2s = pyco2.sys(s=[0, 5, 10, 15, 25, 30], opt_k_carbonic=16)

#
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
