# %%
import numpy as onp
from jax import numpy as np
from scipy import stats

from PyCO2SYS.classes.function_graph import FunctionGraph


# alpha: standard input with default
# beta: computed by function with no inputs
# gamma, epsilon: computed from part of coeffs and other inputs/intermediates
# delta: standard intermediate
# phi: standard end product
# coeffs: set of coefficients used in parts by multiple other steps
rng = onp.random.default_rng(1)
funcs = {
    "beta": lambda: 1.5,
    "gamma": lambda coeffs, alpha, beta: (
        coeffs[0] + np.log(alpha) * coeffs[1] + beta * coeffs[2]
    ),
    "Delta": lambda beta, gamma: beta + np.sqrt(gamma),
    "epsilon": lambda coeffs, alpha, Delta: (
        coeffs[3] * alpha**2 + coeffs[4] * Delta * alpha
    ),
    "phi": lambda Delta, epsilon: 2 * Delta + epsilon,
    # Below is to get covariances between two (or more) parameters
    "combi": lambda gamma, phi: np.array([gamma, phi]),
}
defaults = dict(
    alpha=12.0,
    coeffs=np.array([0.0, 1, 1, 1, 1]),
)
shortcuts = dict(
    a="alpha",
    b="beta",
    c="gamma",
    delta="Delta",
    d="Delta",
    e="epsilon",
    g="gamma",
    f="phi",
)


def abs_diff_pct(a, b):
    return np.abs(200 * (a - b) / (a + b))


def test_make_fg():
    fg = FunctionGraph(
        defaults=defaults,
        funcs=funcs,
        shortcuts=shortcuts,
    )
    assert isinstance(fg, FunctionGraph)
    assert len(fg.data) == 0


def test_make_fg_all_defaults():
    fg = FunctionGraph(
        defaults=defaults,
        funcs=funcs,
        shortcuts=shortcuts,
    ).set_data()
    assert set(fg.data.keys()) == set(defaults.keys())
    fg.solve()
    for k in funcs:
        assert k in fg.data


def test_shortcuts():
    fg = FunctionGraph(
        defaults=defaults,
        funcs=funcs,
        shortcuts=shortcuts,
    ).set_data()
    assert (
        fg["d"] == fg["Delta"] == fg["delta"] == fg.d == fg.Delta == fg.delta
    )


def test_uncertainty_scalar():
    test_ua_std = 0.001
    fg = (
        FunctionGraph(
            defaults=defaults,
            funcs=funcs,
            shortcuts=shortcuts,
        )
        .set_data()
        .set_uncertainty(a=test_ua_std**2)  # this should be the variance
        .propagate("phi")
    )
    fd_diff = 1e-6
    fd_fg = (
        FunctionGraph(
            defaults=defaults,
            funcs=funcs,
            shortcuts=shortcuts,
        )
        .set_data(a=np.array([fg.a, fg.a + fd_diff]))
        .solve("phi")
    )
    fd_df_da = (fd_fg.phi[1] - fd_fg.phi[0]) / fd_diff
    fg.get_jacs("phi", "a")
    assert abs_diff_pct(fd_df_da, fg.jacs.phi.a) < 1e-5
    for _ in range(10):
        mc_nreps = 10_000_000
        mc_a = rng.normal(loc=fg.a, scale=test_ua_std, size=mc_nreps)
        mc_fg = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data(a=mc_a)
            .solve("phi")
        )
        assert abs_diff_pct(np.var(mc_fg.phi), fg.u.phi) < 0.3


def test_u_coeffs_independent():
    u_coeffs_vec = np.array([0.01, 0.02, 0.005, 0.001, 0.003])  # as std
    u_coeffs_mx = np.array(
        [
            [1.0e-04, 0, 0, 0, 0],
            [0, 4.0e-04, 0, 0, 0],
            [0, 0, 2.5e-05, 0, 0],
            [0, 0, 0, 1.0e-06, 0],
            [0, 0, 0, 0, 9.0e-06],
        ]
    )
    fg_vec = (
        FunctionGraph(
            defaults=defaults,
            funcs=funcs,
            shortcuts=shortcuts,
        )
        .set_data()
        .set_uncertainty(coeffs=u_coeffs_vec**2)
        .solve()
        .propagate()
    )
    fg_mx = (
        FunctionGraph(
            defaults=defaults,
            funcs=funcs,
            shortcuts=shortcuts,
        )
        .set_data()
        .set_uncertainty(coeffs=u_coeffs_mx)
        .solve()
        .propagate()
    )
    for _ in range(10):
        mc_nreps = 10_000_000
        mc_coeffs = rng.normal(
            loc=fg_mx.coeffs, scale=u_coeffs_vec, size=(mc_nreps, 5)
        ).T
        fg_mc = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data(coeffs=mc_coeffs)
            .solve()
        )
        for k in fg_vec:
            if k not in ["alpha", "beta", "coeffs", "combi"]:
                assert np.allclose(fg_vec.u[k], fg_mx.u[k])
                assert abs_diff_pct(fg_vec.u[k], np.var(fg_mc[k])) < 0.2


def test_u_coeffs_covar_scalar():
    u_coeffs_ind = np.array(
        [
            [1.0e-04, 0, 0, 0, 0],
            [0, 4.0e-04, 0, 0, 0],
            [0, 0, 2.5e-05, 0, 0],
            [0, 0, 0, 1.0e-06, 0],
            [0, 0, 0, 0, 9.0e-06],
        ]
    )
    for _ in range(10):
        u_coeffs = stats.wishart(
            len(u_coeffs_ind), u_coeffs_ind, seed=rng
        ).rvs()
        fg = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data()
            .set_uncertainty(coeffs=u_coeffs)
            .solve()
            .propagate()
        )
        mc_nreps = 10_000_000
        coeffs_mc = (
            stats.multivariate_normal(
                mean=fg.coeffs,
                cov=u_coeffs,
                seed=rng,
            )
            .rvs(size=mc_nreps)
            .T
        )
        fg_mc = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data(coeffs=coeffs_mc)
            .solve()
        )
        for k in fg:
            if k not in ["alpha", "beta", "coeffs", "combi"]:
                assert abs_diff_pct(fg.u[k], np.var(fg_mc[k])) < 0.3


def test_u_coeffs_covar_vec():
    u_coeffs_ind = np.array(
        [
            [1.0e-04, 0, 0, 0, 0],
            [0, 4.0e-04, 0, 0, 0],
            [0, 0, 2.5e-05, 0, 0],
            [0, 0, 0, 1.0e-06, 0],
            [0, 0, 0, 0, 9.0e-06],
        ]
    )
    for _ in range(10):
        alpha = rng.uniform(20, 50, size=1000)
        u_coeffs = stats.wishart(
            len(u_coeffs_ind), u_coeffs_ind, seed=rng
        ).rvs()
        fg = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data(alpha=alpha)
            .set_uncertainty(coeffs=u_coeffs)
            .solve()
            .propagate()
        )
        mc_nreps = 10_000
        coeffs_mc = (
            stats.multivariate_normal(
                mean=fg.coeffs,
                cov=u_coeffs,
                seed=rng,
            )
            .rvs(size=mc_nreps)
            .T
        )
        fg_mc = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data(alpha=np.vstack(alpha), coeffs=coeffs_mc)
            .solve()
        )
        for k in fg:
            if k not in ["alpha", "beta", "coeffs", "combi"]:
                assert (abs_diff_pct(np.cov(fg_mc[k]), fg.u[k]) < 5).all()


def test_u_covar_combi():
    u_coeffs_ind = np.array(
        [
            [1.0e-04, 0, 0, 0, 0],
            [0, 4.0e-04, 0, 0, 0],
            [0, 0, 2.5e-05, 0, 0],
            [0, 0, 0, 1.0e-06, 0],
            [0, 0, 0, 0, 9.0e-06],
        ]
    )
    for _ in range(10):
        alpha = rng.uniform(20, 50, size=1000)
        u_coeffs = stats.wishart(
            len(u_coeffs_ind), u_coeffs_ind, seed=rng
        ).rvs()
        fg = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data(alpha=alpha)
            .set_uncertainty(coeffs=u_coeffs)
            .solve()
            .propagate()
        )
        # Check combi variances match
        combi_vars = FunctionGraph.cut_cov(fg.u.combi)
        assert np.allclose(np.diag(fg.u.gamma), combi_vars[0])
        assert np.allclose(np.diag(fg.u.phi), combi_vars[1])
        mc_nreps = 10_000
        coeffs_mc = (
            stats.multivariate_normal(
                mean=fg.coeffs,
                cov=u_coeffs,
                seed=rng,
            )
            .rvs(size=mc_nreps)
            .T
        )
        fg_mc = (
            FunctionGraph(
                defaults=defaults,
                funcs=funcs,
                shortcuts=shortcuts,
            )
            .set_data(alpha=np.vstack(alpha), coeffs=coeffs_mc)
            .solve()
        )
        # Check combi covariances make sense
        assert (
            abs_diff_pct(fg.u.combi[0, :, 0, :], np.cov(fg_mc.g)) < 5
        ).all()
        assert (
            abs_diff_pct(fg.u.combi[1, :, 1, :], np.cov(fg_mc.f)) < 5
        ).all()


# test_make_fg()
# test_make_fg_all_defaults()
# test_shortcuts()
# test_uncertainty_scalar()
# test_u_coeffs_independent()
# test_u_coeffs_covar_scalar()
# test_u_coeffs_covar_vec()
# test_u_covar_combi()
