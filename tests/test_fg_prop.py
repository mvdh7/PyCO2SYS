# %%
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from scipy import stats

from tests.function_graph import FunctionGraph


funcs = {
    "c": lambda a, b, coeffs: coeffs[0] * a - b,
    "d": lambda a, b, coeffs: coeffs[1] * b - a,
    "e": lambda b, d, coeffs: coeffs[2] * b + coeffs[3] * d,
}
fu = FunctionGraph(funcs=funcs)

pos = nx.nx_agraph.graphviz_layout(fu.graph, prog="dot")
fig, ax = plt.subplots()
nx.draw_networkx(
    fu.graph,
    pos=pos,
    nodelist=fu.graph.nodes,
    node_color=[
        nx.get_node_attributes(fu.graph, "state", default=-1)[n]
        for n in fu.graph.nodes
    ],
    vmin=-1,
    vmax=3,
)


def diff_pct(x, y):
    return 200 * np.abs(x - y) / (x + y)


def test_ind_const_scalar():
    """Independent, constant uncertainties in scalar a and b."""
    pvars = ["c", "d", "e"]
    rng = default_rng(1)
    data = dict(
        a=5.2,
        b=8.1,
        coeffs=np.array([0.5, -3, 4, 2]),
    )
    uncert = dict(
        a=0.1**2,
        b=0.3**2,
    )
    fu_prop = (
        FunctionGraph(funcs=funcs).set_data(**data).set_u(**uncert).prop(pvars)
    )
    n_reps = 10_000_000
    data_sim = {
        k: rng.normal(loc=data[k], scale=np.sqrt(uncert[k]), size=n_reps)
        if k in uncert
        else data[k]
        for k in data
    }
    fu_sim = FunctionGraph(funcs=funcs).set_data(**data_sim).solve(pvars)
    u_sim = {k: np.var(fu_sim[k]) for k in pvars}
    for k in pvars:
        assert diff_pct(fu_prop.u[k], u_sim[k]) < 0.1


def test_ind_const_vector():
    """Independent, constant uncertainties in vector a and b."""
    pvars = ["c", "d", "e"]
    rng = default_rng(1)
    n_data = 10
    data = dict(
        a=rng.uniform(low=-10, high=10, size=n_data),
        b=rng.uniform(low=-10, high=10, size=n_data),
        coeffs=np.array([0.5, -3, 4, 2]),
    )
    uncert = dict(
        a=0.1**2,
        b=0.3**2,
    )
    fu_prop = (
        FunctionGraph(funcs=funcs).set_data(**data).set_u(**uncert).prop(pvars)
    )
    fu_prop_nocov = (
        FunctionGraph(funcs=funcs)
        .set_data(**data)
        .set_u(**uncert)
        .prop(pvars, keep_cov=False)
    )
    n_reps = 1_000_000
    data_sim = {
        k: rng.normal(
            loc=data[k], scale=np.sqrt(uncert[k]), size=(n_reps, n_data)
        )
        if k in uncert
        else data[k]
        for k in data
    }
    fu_sim = FunctionGraph(funcs=funcs).set_data(**data_sim).solve(pvars)
    u_sim_var = {k: np.var(fu_sim[k], axis=0) for k in pvars}
    u_sim_cov = {k: np.cov(fu_sim[k], rowvar=False) for k in pvars}
    for k in pvars:
        assert all(diff_pct(np.diag(fu_prop.u[k]), u_sim_var[k]) < 0.5)
        assert np.max(np.abs(u_sim_cov[k] - fu_prop.u[k])) < 0.01
        assert np.allclose(np.diag(fu_prop.u[k]), fu_prop_nocov.u[k])


def test_coeff_cov():
    pvars = ["c", "d", "e"]
    rng = default_rng(1)
    n_data = 10
    data = dict(
        a=rng.uniform(low=-10, high=10, size=n_data),
        b=rng.uniform(low=-10, high=10, size=n_data),
        coeffs=np.array([0.5, -3, 4, 2]),
    )
    uncert = dict(
        coeffs=stats.wishart(
            df=len(data["coeffs"]),
            scale=np.eye(len(data["coeffs"])) * 0.1,
            seed=rng,
        ).rvs()
    )
    fu_prop = (
        FunctionGraph(funcs=funcs).set_data(**data).set_u(**uncert).prop(pvars)
    )
    n_reps = 1_000_000
    coeffs_sim = (
        stats.multivariate_normal(
            mean=data["coeffs"],
            cov=uncert["coeffs"],
        )
        .rvs(size=n_reps)
        .T
    )
    data_sim = data.copy()
    data_sim["a"] = np.vstack(data["a"])
    data_sim["b"] = np.vstack(data["b"])
    data_sim["coeffs"] = coeffs_sim
    fu_sim = FunctionGraph(funcs=funcs).set_data(**data_sim).solve(pvars)
    u_sim_cov = {k: np.cov(fu_sim[k]) for k in pvars}
    for k in pvars:
        assert (diff_pct(fu_prop.u[k], u_sim_cov[k]) < 5).all()


# test_ind_const_scalar()
# test_ind_const_vector()
# test_coeff_cov()
