# %%
import jax
import networkx as nx
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
    "epsilon": lambda alpha, Delta: alpha**2 + Delta,
    "phi": lambda Delta, epsilon: 2 * Delta + epsilon,
}
defaults = dict(
    alpha=0.0,
    coeffs_gamma=np.array([0.0, 1.0, 1.0]),
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
    # alpha=np.array([1, 2.0]),
    alpha=np.array([[1.0, 2.0], [3.0, 4.0], [5, 6]]),
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
    jshape = np.shape(jac)
    if jshape == ():
        return jac
    else:
        ixs = "".join(chr(97 + i) for i in range(int(len(jshape) / 2)))
        return np.einsum(ixs + ixs + "->" + ixs, jac)


pj = parse_jac_from_scalar(fu.grads.e.a)
print(pj)

# print("de/da =", np.einsum("ijij->ij", fu.grads.e.a))

fu.get_grads("e", "coeffs_gamma")
print(fu.grads.e.coeffs_gamma)

# 0,0,0,0 => 0,0
# 0,1,0,1 => 0,1
# 1,0,1,0 => 1,0
# 1,1,1,1 => 1,1
# 2,0,2,0 => 2,0
# 2,1,2,1 => 2,1

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
