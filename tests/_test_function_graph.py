# %%
import jax
import networkx as nx
import numpy as onp
from jax import numpy as np
from matplotlib import pyplot as plt

from tests.function_graph import FunctionGraph


# alpha: standard input with default
# beta: computed by function with no inputs
# gamma, epsilon: computed from part of coeffs and other inputs/intermediates
# delta: standard intermediate
# phi: standard end product
# coeffs: set of coefficients used in parts by multiple other steps
rng = onp.random.default_rng()
funcs = {
    "beta": lambda: 1.5,
    "gamma": lambda coeffs, alpha, beta: (
        coeffs[0] + np.log(alpha) * coeffs[1] + beta * coeffs[2]
    ),
    # "gamma": lambda alpha, beta: alpha + beta,
    "Delta": lambda beta, gamma: beta + np.sqrt(gamma),
    "epsilon": lambda coeffs, alpha, Delta: (
        coeffs[3] * alpha**2 + coeffs[4] * Delta * alpha
    ),
    "phi": lambda Delta, epsilon: 2 * Delta + epsilon,
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
    # Computing phi should make everything else be calculated
    fg.solve("phi")
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


# def test_uncertainty_scalar():
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
print(fd_df_da, fg.jacs.phi.a)  # NOTE Jacobian looks good
mc_nreps = 1_000_000
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
print(np.var(mc_fg.phi), fg.u.phi)  # NOTE uncertainty looks good too!


# test_make_fg()
# test_make_fg_all_defaults()
# test_shortcuts()
# test_uncertainty_scalar()
