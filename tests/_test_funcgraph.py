# %%
import jax
import networkx as nx
import numpy as onp
from jax import numpy as np
from matplotlib import pyplot as plt

from PyCO2SYS.meta import valid
from tests.function_graph import FunctionGraph, egrad


@valid(alpha=[0, 1.5], beta=[1, 2])
def get_gamma(coeffs, alpha, beta):
    return coeffs[0] + np.exp(-alpha) * coeffs[1] + beta * coeffs[2]


# TODO for uncertainty propagation, I need to be able to choose which
# parameters to propagate with Jacobians (e.g., sets of coefficients which
# have covarying uncertainties) and which not to (e.g., other parameters on
# the same set of dimensions as the first)
# Otherwise, the uncertainty is always size dimensions squared, but mostly
# full of zeroes!


# alpha: standard input with default
# beta: computed by function with no inputs
# gamma, epsilon: computed from part of coeffs and other inputs/intermediates
# delta: standard intermediate
# phi: standard end product
# coeffs: set of coefficients used in parts by multiple other steps
funcs = {
    "beta": lambda: 1.5,
    "gamma": get_gamma,
    "Delta": lambda beta, gamma: beta + np.sqrt(gamma),
    "epsilon": lambda coeffs, alpha, Delta: (
        coeffs[3] * alpha**2 + coeffs[4] * Delta * alpha
    ),
    "phi": lambda Delta, epsilon: 2 * Delta + epsilon,
}
defaults = dict(
    alpha=0.0,
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
fu = FunctionGraph(
    defaults=defaults,
    funcs=funcs,
    shortcuts=shortcuts,
)

data = dict(
    # alpha=1.0,
    # alpha=np.array([1.0]),
    # alpha=np.array(
    #     [
    #         [1, 2.0],
    #         [1, 2.0],
    #         [1, 2.0],
    #     ]
    # ),
    alpha=np.vstack([1.5, 2.5, 3.5]),
    # beta=np.vstack([1, 3.0, 2]),
    b=2.5,
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
get_e_from_cg = fu.get_func_of_from_wrt(get_e, "coeffs")

e_from_cg = get_e_from_cg(
    kwargs["coeffs"],
    **{k: v for k, v in kwargs.items() if k != "coeffs"},
)

testgrad = jax.jacfwd(get_e_from_cg)(
    kwargs["coeffs"],
    **{k: v for k, v in kwargs.items() if k != "coeffs"},
)
# print(testgrad)

# get_dd_dg = fu.get_grad_func("d", "g")
# TODO make a more convenient way to get the args and kwargs for get_func_of_from_wrt
# de_dcg = get_dd_dg(kwargs["gamma"], **{k: v for k, v in kwargs.items() if k != "gamma"})
# get_dd_da = fu.get_grad_func("d", "a")
# dd_da = get_dd_da(kwargs["alpha"], **{k: v for k, v in kwargs.items() if k != "alpha"})

# For uncertainty propagation, decide whether to use grad or jac based on
# shape of uncertainty relative to parameter OR always use jac if it's coeffs
# (check `fu.graph.nodes['coeffs']`)
# And don't (ever?) store jacs (too big?)
fu.get_grads(["e", "f"], ["a", "b"])
fu.get_jacs(["e", "f"], ["a", "b"])
# jac = fu.grads.e.a
# jshape = np.shape(jac)


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


print(f"fu.e {fu.e.shape}:")
printif(fu.e)

print(f"fu.a {fu.a.shape}:")
printif(fu.a)

print(f"fu.coeffs {fu.coeffs.shape}:")
printif(fu.coeffs)

print(f"fu.jacs.e.a {fu.jacs.e.a.shape}:")
printif(fu.jacs.e.a)

fu.get_jacs("d", "coeffs")
print(f"fu.jacs.d.coeffs {fu.jacs.d.coeffs.shape}:")
printif(fu.jacs.d.coeffs)

fu.propagate("phi")
fu.get_valid("f")

# %%
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


# @jax.jit
func = lambda a, b: get_c(a, b)  # noqa
args = (np.array([1.5, 2.5]), b)
uncert = np.array(
    [
        [0.2, 0.3],
        [0.3, 0.5],
    ]
)

val = func(*args)
x_ndims = len(np.shape(args[0]))
y_ndims = len(np.shape(val))
ux_ndims = len(np.shape(uncert))
print(x_ndims, y_ndims, ux_ndims)
subscripts = FunctionGraph.get_einsum_code(
    len(np.shape(args[0])),
    len(np.shape(val)),
    len(np.shape(uncert)),
)
jac = jax.jacfwd(func)(*args)

uprop = FunctionGraph._propagate(args[0], val, jac, uncert)
print(uprop)

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
