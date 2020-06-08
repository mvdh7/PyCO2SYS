import PyCO2SYS as pyco2
import numpy as np

# Initialise a co2dict
# - generate all combinations of marine carbonate system parameters
pars = np.array([2300, 2150, 8.1, 400, 400, 350, 1700, 10])
partypes = np.arange(1, 9, 1)
par1type, par2type = np.meshgrid(partypes, partypes)
par1type = par1type.ravel()
par2type = par2type.ravel()
icases = pyco2.solve.getIcase(par1type, par2type, checks=False)
goodpars = ~np.isin(icases, [45, 48, 58]) & ~(par1type == par2type)
par1type = par1type[goodpars]
par2type = par2type[goodpars]
par1 = pars[par1type - 1]
par2 = pars[par2type - 1]
# - set other conditions
sal = 31.4
tempin = 12.2
tempout = 23.1
presin = 848.1
presout = 1509.2
si = 13
phos = 3
h2s = 0.12
nh3 = 0.5
k1k2c = 12
kso4c = 3
phscale = 3
# - get the co2dict
co2dict = pyco2.CO2SYS(
    np.array([par1[0]]),
    par2[0],
    par1type[0],
    par2type[0],
    sal,
    tempin,
    tempout,
    presin,
    presout,
    si,
    phos,
    phscale,
    k1k2c,
    kso4c,
    H2S=h2s,
    NH3=nh3,
)
# - propagate the uncertainties
grads_of = "all"
grads_wrt = ["PAR1", "PAR2", "TB", "K1input"]
co2derivs, dxs = pyco2.uncertainty.forward(
    co2dict, grads_of, grads_wrt, totals=None, equilibria_in=None, equilibria_out=None,
)


# # Get every derivative with the uncertainty module
# grads_of = "all" # ["OmegaARin", "OmegaARout"]
# grads_wrt = ["PAR1", "PAR2"]
# dx = 1e-4
# co2derivs = pyco2.uncertainty.derivatives(
#     co2dict,
#     grads_of,
#     grads_wrt,
#     totals=None,
#     equilibria_in=None,
#     equilibria_out=None,
#     dx=dx,
#     use_explicit=False,
#     verbose=True,
# )
# grads_of = list(co2derivs.keys())

# # Next get the same derivatives but by perturbing the inputs one by one
# co2dict_perturb = {}
# for par in grads_wrt:
#     if par == "PAR1":
#         par1perturb = par1 + dx
#     else:
#         par1perturb = par1
#     if par == "PAR2":
#         par2perturb = par2 + dx
#     else:
#         par2perturb = par2
#     co2dict_perturb.update(
#         {
#             par: pyco2.CO2SYS(
#                 par1perturb,
#                 par2perturb,
#                 par1type,
#                 par2type,
#                 sal,
#                 tempin,
#                 tempout,
#                 presin,
#                 presout,
#                 si,
#                 phos,
#                 phscale,
#                 k1k2c,
#                 kso4c,
#                 H2S=h2s,
#                 NH3=nh3,
#             )
#         }
#     )
# co2derivs_perturb = {
#     of: {par: (co2dict_perturb[par][of] - co2dict[of]) / dx for par in grads_wrt}
#     for of in grads_of
# }


# @np.errstate(invalid="ignore", divide="ignore")
# def aspercent(a, b):
#     """Calculate % difference between `a` and `b` relative to their mean, or return zero
#     if/where their mean is zero."""
#     abmean = np.mean(np.array([a, b]), axis=0)
#     return np.where(abmean == 0, 0.0, 100 * np.abs((a - b) / abmean))


# # Compare the results
# co2derivs_diffpcts = {
#     of: {
#         par: aspercent(co2derivs_perturb[of][par], co2derivs[of][par])
#         for par in grads_wrt
#     }
#     for of in grads_of
# }
# co2derivs_maxdiffpcts = {
#     of: {par: np.max(co2derivs_diffpcts[of][par]) for par in grads_wrt}
#     for of in grads_of
# }
