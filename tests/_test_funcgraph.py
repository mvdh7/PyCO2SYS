# %%
import jax
import networkx as nx
import numpy as onp
from jax import numpy as np
from matplotlib import pyplot as plt

from tests.function_graph import FunctionGraph

funcs = {
    "beta": lambda: 1.5,
    "gamma": lambda coeffs_gamma, alpha, beta: (
        coeffs_gamma[0] + alpha * coeffs_gamma[1] + beta * coeffs_gamma[2]
    ),
    # "gamma": lambda alpha, beta: alpha + beta,
    "Delta": lambda beta, gamma: beta + gamma,
    "epsilon": lambda coeffs_epsilon, alpha, Delta: (
        coeffs_epsilon[0] * alpha**2 + coeffs_epsilon[1] * Delta
    ),
    "phi": lambda Delta, epsilon: 2 * Delta + epsilon,
}
defaults = dict(
    alpha=0.0,
    coeffs_gamma=np.array([0.0, 1.0, 1.0]),
    coeffs_epsilon=np.array([1.0, 1.0]),
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

fu = FunctionGraph(
    defaults=defaults,
    funcs=funcs,
    shortcuts=shortcuts,
)

data = dict(
    # alpha=1.0,
    # alpha=np.array([1.0]),
    alpha=np.array([[1, 2.0]]),
    beta=np.vstack([1, 2, 3.0]),
    # alpha=np.array([[1.0, 2.0], [3.0, 4.0], [5, 6]]),
    # d=3,
    # b=2,
    # f=3,
    # g=4.0,
    # h=2,
)
fu.set_data(**data).set_u(a=0.1).solve("e")
# print(fu.data)
# result_f = fu.f
# print(fu.data)
# results = fu[["f", "g"]]

kwargs = {k: fu[k] for k in fu.nodes_original}
get_e = fu.get_func_of("e")
get_e_from_cg = fu.get_func_of_from_wrt(get_e, "coeffs_gamma")

e_from_cg = get_e_from_cg(
    kwargs["coeffs_gamma"],
    **{k: v for k, v in kwargs.items() if k != "coeffs_gamma"},
)

testgrad = jax.jacfwd(get_e_from_cg)(
    kwargs["coeffs_gamma"],
    **{k: v for k, v in kwargs.items() if k != "coeffs_gamma"},
)
# print(testgrad)

# get_dd_dg = fu.get_grad_func("d", "g")
# TODO make a more convenient way to get the args and kwargs for get_func_of_from_wrt
# de_dcg = get_dd_dg(kwargs["gamma"], **{k: v for k, v in kwargs.items() if k != "gamma"})
# get_dd_da = fu.get_grad_func("d", "a")
# dd_da = get_dd_da(kwargs["alpha"], **{k: v for k, v in kwargs.items() if k != "alpha"})

fu.get_grads("e", "a")
# jac = fu.grads.e.a
# jshape = np.shape(jac)


def parse_jac_from_scalar(jac):
    """Collapse a Jacobian matrix to remove superfluous zeroes and make the
    shape match the 'of' parameter when the 'wrt' parameter is 'scalar',
    i.e., it isn't a set a coefficients.
    """
    # NOTE Is this actually necessary for uncertainty propagation?
    #      Quite possibly not...
    # NOTE it might in fact be *incorrect* to do this, but I do still need
    #      to use the einsum notation for propagation.
    jshape = np.shape(jac)
    if jshape == ():
        return jac
    else:
        # `ixs` is "aa->a", "abab->ab", "abcabc->abc", ...
        ixs = "".join(chr(97 + i) for i in range(int(len(jshape) / 2)))
        return np.einsum(ixs + ixs + "->" + ixs, jac)


def printif(arg):
    return
    print(arg)


# NOTE Jacobian has shape: (*y.shape, *x.shape)
# Uncertainty matrix for x has shape (*x.shape, *x.shape)
# Uncertainty matrix for y has shape (*y.shape, *y.shape)
#
# for a vector (as if x was coeffs):
# x.shape = (2,)
# y.shape = (1,)
# ux.shape = (2, 2)
# jac.shape = (1, 2)
# jac @ ux @ jac.T is
# [[dy0/dx0, dy0/dx1]] @ [[v_x0, c_x0x1]  @ [[dy0/dx0]
#                         [c_x0x1, v_x1]]    [dy0/dx1]]
# uy is:
# (dy0/dx0)**2 * v_x0 + (dy0/dx1)**2 * v_x1 + 2*(dy0/dx0)*(dy0/dx1) * c_x0x1
# which is
# = jac[0,0] * ux[0,0] * jac[0,0]
# + jac[0,1] * ux[1,1] * jac[0,1]
# + jac[0,0] * ux[0,1] * jac[0,1]
# + jac[0,1] * ux[1,0] * jac[0,0]
#
# Let's generalise
# a = index(es) within x
# b = index(es) within y
# So we have:
# = jac[0,1] * ux[0,0] * jac[0,0]
# + jac[0,1] * ux[1,1] * jac[0,1]
# + jac[0,0] * ux[0,1] * jac[0,1]
# + jac[0,1] * ux[1,0] * jac[0,0]


# pj = parse_jac_from_scalar(fu.grads.e.a)
print(f"fu.e {fu.e.shape}:")
printif(fu.e)

print(f"fu.a {fu.a.shape}:")
printif(fu.a)

print(f"fu.coeffs_gamma {fu.coeffs_gamma.shape}:")
printif(fu.coeffs_gamma)

print(f"fu.grads.e.a {fu.grads.e.a.shape}:")
printif(fu.grads.e.a)

fu.get_grads("e", ["coeffs_gamma", "coeffs_epsilon"])
print(f"fu.grads.e.coeffs_gamma {fu.grads.e.coeffs_gamma.shape}:")
printif(fu.grads.e.coeffs_gamma)


# %%
def get_c(a, b):
    return a * b[0] + b[1]


a = np.array([1.0, 2.0, 3.0, 4.0])
aa = np.array(
    [
        [1.0, 2.0],
        [3.0, 4.0],
    ]
)
ua_vec = np.array([0.1, 0.1, 0.1, 0.1])
ua = np.diag(np.array([0.1, 0.1, 0.1, 0.1]))
uaa = np.array(
    [
        [0.1, 0.1],
        [0.1, 0.1],
    ]
)
b = np.array([2.5, 0.5])
ub = np.array(
    [
        [0.5, 0.1],
        [0.1, 0.3],
    ]
)
c = get_c(a, b)
jac_ca = jax.jacfwd(get_c)(a, b)
jac_cb = jax.jacfwd(get_c, argnums=1)(a, b)
uc_a = jac_ca @ ua @ jac_ca.T
uc_b = jac_cb @ ub @ jac_cb.T
uc_a_einsum = np.einsum("ab,bc,dc->ad", jac_ca, ua, jac_ca)
uc_b_einsum = np.einsum("ab,bc,dc->ad", jac_cb, ub, jac_cb)
uc_a_vec_einsum = np.einsum("ab,b,cb->ac", jac_ca, ua_vec, jac_ca)

cc = get_c(aa, b)
jac_ccaa = jax.jacfwd(get_c)(aa, b)
jac_ccb = jax.jacfwd(get_c, argnums=1)(aa, b)
# First two below are NOT correct...
ucc_aa_wrong = jac_ccaa @ uaa @ jac_ccaa.T
ucc_b_wrong = jac_ccb @ ub @ jac_ccb.T
# ... but these are good!
ucc_aa_vec_einsum = np.einsum("abcd,cd,efcd->abef", jac_ccaa, uaa, jac_ccaa)
ucc_b_einsum = np.einsum("abc,cd,efd->abef", jac_ccb, ub, jac_ccb)

# What's the general pattern?
# - Start with dims of jac, which is (*y.shape, *x.shape) [AB]
# - then dims of u: either x.shape or (*x.shape, *x.shape) [BC]
# --- the first set of dims gets the same labels as the last part of jac [B]
# --- the second set (if it's there) gets new labels [C]
# --- if no covariances (x.shape == ux.shape), then B = C
# - Then jac again including a 'transpose' [DC]
# --- the first set of dims gets new labels [D]
# --- the second set gets the labels of the second set of u [C]
# - Output is the first jac dims from its first and final appearance [AD]


# Below is IT!
def get_einsum_code(
    x: tuple[int, ...],
    y: tuple[int, ...],
    ux: tuple[int, ...],
) -> str:
    """Get the einsum code for uncertainty propagation of `ux` from `x` to `y`.

    Parameters
    ----------
    x : tuple[int, ...]
        The shape of the variable to propagate uncertainty from.
    y : tuple[int, ...]
        The shape of the variable to propagate uncertainty into.
    ux : tuple[int, ...]
        The shape of the uncertainties for `x`.  Should be either
          - the same as `x`, or
          - `(*x, *x)`.

    Returns
    -------
    einsum_code : str
        The code to use with `np.einsum`:
            `uy = np.einsum(einsum_code, jac_yx, ux, jac_yx)`
    """
    i0 = 97
    A = ""
    for i in range(len(y.shape)):
        A += chr(i0)
        i0 += 1
    B = ""
    for i in range(len(x.shape)):
        B += chr(i0)
        i0 += 1
    if x.shape == ux.shape:
        C = B
    else:
        C = ""
        for i in range(len(x.shape)):
            C += chr(i0)
            i0 += 1
    D = ""
    for i in range(len(y.shape)):
        D += chr(i0)
        i0 += 1
    if B == C:
        return f"{A}{B},{B},{D}{C}->{A}{D}"
    else:
        return f"{A}{B},{B}{C},{D}{C}->{A}{D}"


@jax.jit
def prop(arg0, val, jac, uncert):
    esc = get_einsum_code(arg0, val, uncert)
    return np.einsum(esc, jac, uncert, jac)


func = lambda b, a: get_c(a, b)  # noqa
args = (b, aa)
uncert = ub

val = func(*args)
jac = jax.jacfwd(func)(*args)

uprop = prop(args[0], val, jac, uncert)
print(uprop)

# %%
pos = nx.nx_agraph.graphviz_layout(fu.graph, prog="dot")
fig, ax = plt.subplots()
nx.draw_networkx(
    fu.graph,
    pos=pos,
    nodelist=fu.graph.nodes,
    node_color=[
        nx.get_node_attributes(fu.graph, "state", default=-1)[n] for n in fu.graph.nodes
    ],
    vmin=-1,
    vmax=3,
)

# %%
# https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
numbers = onp.array(
    [
        [1, 2, 3],
        [-1, -2, -3],
    ]
)
letters = onp.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
cast = onp.einsum("ab,bc->ac", numbers, letters)
print(cast)

part1 = onp.array(
    [
        [
            [1, 2, 3],
            [8, 10, 12],
            [21, 24, 27],
        ]
    ]
)  # if `numbers` has only the first row, this is ab,bc->abc
