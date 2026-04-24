# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

import PyCO2SYS as pyco2


rng = np.random.default_rng()


temperature = np.linspace(0, 45, num=500)
salinity = np.linspace(0, 45, num=500)

cov_pkb = pyco2.uncertainty.utest

co2t = (
    pyco2.sys(opt_k_BOH3=3, temperature=temperature)
    .set_u(coeffs_pk_BOH3=np.pad(cov_pkb, ((0, 1), (0, 1))))
    # .set_u_coeffs_from_single(pk_BOH3=0.04**2)
    .prop("pk_BOH3")
)
co2s = (
    pyco2.sys(opt_k_BOH3=3, salinity=salinity)
    .set_u(coeffs_pk_BOH3=np.pad(cov_pkb, ((0, 1), (0, 1))))
    # .set_u_coeffs_from_single(pk_BOH3=0.04**2)
    .prop("pk_BOH3")
)

var_pkb = np.diag(cov_pkb)
std_pkb = np.sqrt(var_pkb)
coeffs_true = pyco2.equilibria.p1atm.coeffs_pk_BOH3_total_MMB26()[:-1]
coeffs_sim = rng.normal(
    loc=coeffs_true,
    scale=std_pkb,
    size=(10000, len(coeffs_true)),
)

mvn = multivariate_normal(
    mean=coeffs_true,
    cov=cov_pkb,
    allow_singular=True,
)
coeffs_mvn = mvn.rvs(size=10000)

nc = len(coeffs_true)
fig, axs = plt.subplots(nrows=nc, ncols=nc)
for i in range(nc):
    for j in range(nc):
        axs[i, j].scatter(coeffs_sim[:, i], coeffs_sim[:, j], s=5)
        axs[i, j].scatter(coeffs_mvn[:, i], coeffs_mvn[:, j], s=5)

# %%
coeffs_sim = rng.normal(
    loc=coeffs_true,
    scale=std_pkb,
)
mvn = multivariate_normal(
    mean=coeffs_true,
    cov=cov_pkb,
    allow_singular=True,
)
coeffs_mvn = mvn.rvs(size=1)
pkb_sim_t = pyco2.equilibria.p1atm.pk_BOH3_total_MMB26(
    np.array([*coeffs_mvn, 0]),
    # pyco2.equilibria.p1atm.coeffs_pk_BOH3_total_MMB26(),
    temperature,
    35,
)
pkb_sim_s = pyco2.equilibria.p1atm.pk_BOH3_total_MMB26(
    np.array([*coeffs_mvn, 0]),
    # pyco2.equilibria.p1atm.coeffs_pk_BOH3_total_MMB26(),
    25,
    salinity,
)
#
fig, axs = plt.subplots(nrows=2)
ax = axs[0]
ax.plot(temperature, co2t.pkb)
ax.plot(temperature, pkb_sim_t)
ax.fill_between(
    temperature,
    co2t.pkb - 2 * np.sqrt(np.diag(co2t.u.pkb)),
    co2t.pkb + 2 * np.sqrt(np.diag(co2t.u.pkb)),
)
ax.set_xlabel("Temperature / °C")
ax = axs[1]
ax.plot(salinity, co2s.pkb)
ax.plot(salinity, pkb_sim_s)
ax.fill_between(
    salinity,
    co2s.pkb - 2 * np.sqrt(np.diag(co2s.u.pkb)),
    co2s.pkb + 2 * np.sqrt(np.diag(co2s.u.pkb)),
)
ax.set_xlabel("Salinity")
for ax in axs:
    ax.set_ylabel(r"p$K_\mathrm{B}$")
fig.tight_layout()
