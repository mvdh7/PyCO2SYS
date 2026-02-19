from collections import UserDict
from inspect import signature
from itertools import product
from warnings import warn

import jax
import jax.numpy as np
import networkx as nx

# NODE STATES
# ===========
# -1 = value unknown
#  0 = default value
#  1 = user-provided value
#  2 = calculated as intermediate
#  3 = calculated by explicit request


def egrad(g):
    # From https://github.com/google/jax/issues/3556#issuecomment-649779759
    # modified to allow kwargs for g
    def wrapped(x, *args, **kwargs):
        y, g_vjp = jax.vjp(lambda x: g(x, *args, **kwargs), x)
        (x_bar,) = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped


class ShortcutsDict(UserDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        try:
            return self.data[key.lower()]
        except KeyError:
            return key.lower()


class ShortcutDotDict(UserDict):
    def __init__(self, shortcuts):
        super().__init__()
        self._shortcuts = shortcuts

    def __getattr__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            try:
                return self.data[self._shortcuts[attr]]
            except KeyError:
                raise AttributeError(attr)


class Uncertainties(ShortcutDotDict):
    def __init__(self, shortcuts):
        super().__init__(shortcuts)
        self.assigned = ShortcutDotDict(shortcuts)
        self.parts = ShortcutDotDict(shortcuts)

    def assign(self, **uncertainties):
        for k, v in uncertainties.items():
            self.assigned[self._shortcuts[k]] = v


class FunctionGraph(UserDict):
    def __init__(
        self,
        defaults: dict | None = None,
        graph: nx.DiGraph | None = None,
        funcs: dict | None = None,
        shortcuts: dict | None = None,
    ):
        super().__init__()
        if defaults is not None:
            self.defaults = defaults.copy()
        else:
            self.defaults = {
                n: None for n, attrs in self.graph.nodes.items() if "func" not in attrs
            }
        if graph is not None:
            self.graph = graph.copy()
        else:
            if not isinstance(funcs, dict):
                raise Exception("Either `graph` or `funcs` must be provided")
            self.graph = self.get_graph(funcs)
        if shortcuts is not None:
            self.shortcuts = ShortcutsDict(**shortcuts)
        else:
            self.shortcuts = ShortcutsDict()
        self.ignored = set()
        self.requested = set()
        self.nodes_original = set()
        self.grads = ShortcutDotDict(self.shortcuts)
        self.uncertainty = Uncertainties(self.shortcuts)
        self.u = self.uncertainty

    def __getitem__(self, key):
        # When the user requests a dict key that hasn't been solved for yet,
        # then solve and provide the requested parameter
        self.solve(parameters=key)
        if isinstance(key, list):
            # If the user provides a list of keys to solve for, return all of
            # them as a dict
            return {k: self.data[self.shortcuts[k.lower()]] for k in key}
        else:
            # If a single key is requested, return the corresponding value(s)
            # directly
            return self.data[self.shortcuts[key.lower()]]

    def __getattr__(self, attr):
        # This allows parameter values to be accessed with dot notation, purely
        # for convenience.
        # So, when the user tries to access something with dot notation...
        try:
            # ... then if it's an attribute, return it (this is the standard
            # behaviour)...
            return object.__getattribute__(self, attr)
        except AttributeError:
            # ... but if it's not an attribute,  return the corresponding
            # parameter value, solving for it first if necessary.
            return self[self.shortcuts[attr.lower()]]

    def __setitem__(self, key, value):
        # Don't allow the user to assign new key-value pairs to the dict
        raise RuntimeError("Item assignment is not allowed.")

    def set_data(self, **data):
        ignored = []
        # First, remove anything from the defaults that is also in data
        # (more complicated than we expect because data could contain
        # shortcut keys)
        self_defaults = self.defaults.copy()
        for k in data:
            if self.shortcuts[k] in self_defaults:
                self_defaults.pop(self.shortcuts[k])
        # Then assign the data values and adjust the graph accordingly
        for ks, v in (self_defaults | data).items():
            if v is not None:
                k = self.shortcuts[ks]
                if k in self.graph.nodes:
                    # State 1 means that the parameter was given as an argument
                    # so we need to remove its parent edges if it has any
                    nx.set_node_attributes(
                        self.graph, {k: 1 if ks in data else 0}, name="state"
                    )
                    if "args" in self.graph.nodes[k]:
                        for arg in self.graph.nodes[k]["args"]:
                            self.graph.remove_edge(arg, k)
                        del self.graph.nodes[k]["args"]
                    if "func" in self.graph.nodes[k]:
                        del self.graph.nodes[k]["func"]
                    self.data[k] = v
                else:
                    ignored.append(k)
        if len(ignored) > 0:
            print(
                "Some parameters were not recognised or not valid for this"
                + " combination of known carbonate system parameters and are"
                + " being ignored (see `ignored` attribute)"
            )
        self.ignored |= set(ignored)
        self.nodes_original = set(
            self.shortcuts[k]
            for k, v in (self_defaults | data).items()
            if v is not None
        )
        return self

    def solve(
        self,
        parameters: list | str | None = None,
    ):
        if parameters is None:
            parameters = list(self.graph.nodes)
        elif isinstance(parameters, str):
            parameters = [parameters]
        parameters = [self.shortcuts[p] for p in parameters]
        parameters = set(parameters)
        self.requested |= parameters
        keys_known = list(self.data.keys())
        # Remove known nodes from a copy of self.graph, so that ancestors of
        # known nodes are not unnecessarily recomputed
        graph_unknown = self.graph.copy()
        graph_unknown.remove_nodes_from([k for k in keys_known if k not in parameters])
        # Add intermediate parameters that we need to know in order to
        # calculate the requested parameters
        parameters_all = parameters.copy()
        for p in parameters:
            parameters_all = parameters_all | nx.ancestors(graph_unknown, p)
        # Convert the set of parameters into a list, exclude already-known
        # ones, and organise the list into the order required for calculations
        parameters_all = [
            p
            for p in nx.topological_sort(self.graph)
            if p in parameters_all and p not in keys_known
        ]
        for p in parameters_all:
            attrs = self.graph.nodes[p]
            try:
                self.data[p] = attrs["func"](*[self.data[r] for r in attrs["args"]])
                if p in parameters:
                    nx.set_node_attributes(self.graph, {p: 3}, name="state")
                else:
                    nx.set_node_attributes(self.graph, {p: 2}, name="state")
            except KeyError:
                raise Exception(f"{p} has no associated function in the graph")
        self.remove_jax_overhead(self.data)
        return self

    def get_func_of(self, var_of: str):
        """Create a function to compute `var_of` directly from an input set
        of values.

        The created function has the signature

            value_of = get_value_of(**kwargs)

        where the `kwargs` are the originally user-defined and default values,
        obtained with

            kwargs = {k: fg[k] for k in fg.nodes_original}
        """
        # We get a sub-graph of the node of interest and all its ancestors,
        # excluding originally fixed / user-defined values
        var_of = self.shortcuts[var_of]
        nodes_vo_all = nx.ancestors(self.graph, var_of)
        nodes_vo_all.add(var_of)
        nodes_vo = [n for n in nodes_vo_all if n not in self.nodes_original]
        graph_vo = self.graph.subgraph(nodes_vo)

        def get_value_of(**kwargs):
            kwargs = kwargs.copy()
            # This loops through the functions in the correct order determined
            # above so we end up calculating the value of interest, which is
            # returned
            for n in nx.topological_sort(graph_vo):
                kwargs.update(
                    {
                        n: self.graph.nodes[n]["func"](
                            *[kwargs[v] for v in self.graph.nodes[n]["args"]]
                        )
                    }
                )
            return kwargs[var_of]

        # Generate docstring
        get_value_of.__doc__ = (
            f"Calculate `{var_of}`."
            + "\n\nParameters\n----------"
            + "\nkwargs : dict"
            + "\n    Key-value pairs for the following parameters:"
        )
        for p in self.nodes_original:
            if p in nodes_vo_all:
                get_value_of.__doc__ += f"\n        {p}"
        get_value_of.__doc__ += "\n\nReturns\n-------"
        get_value_of.__doc__ += f"\n{var_of}"
        get_value_of.args_list = [n for n in self.nodes_original if n in nodes_vo_all]
        return get_value_of

    def get_func_of_from_wrt(self, get_value_of, var_wrt):
        """Reorganise a function created with `_get_func_of` so that one of
        its kwargs is instead a positional arg (and which can thus be gradded).

        Parameters
        ----------
        get_value_of : func
            Function created with `get_func_of`.
        var_wrt : str
            Name of the value to use as a positional arg instead.

        Returns
        -------
        A function with the signature
            value_of = get_of_from_wrt(value_wrt, **other_values_original)
        """

        def get_value_of_from_wrt(value_wrt, **other_values_original):
            other_values_original = other_values_original.copy()
            other_values_original.update({self.shortcuts[var_wrt]: value_wrt})
            return get_value_of(**other_values_original)

        # TODO generate a docstring for `get_value_of_from_wrt`
        # NOTE probably better to redesign this so it takes var_of as an arg
        #      instead of the get_value_of function
        return get_value_of_from_wrt

    def get_grad_func(self, var_of: str, var_wrt: str):
        get_value_of = self.get_func_of(var_of)
        get_value_of_from_wrt = self.get_func_of_from_wrt(get_value_of, var_wrt)
        return egrad(get_value_of_from_wrt)

    def get_grad(self, var_of: str, var_wrt: str):
        """Compute the derivative of `var_of` with respect to `var_wrt` and
        store it in `sys.grads[var_of][var_wrt]`.  If there is already a value
        there, then that value is returned instead of recalculating.

        Parameters
        ----------
        var_of : str
            The name of the variable to get the derivative of.
        var_wrt : str
            The name of the variable to get the derivative with respect to.
            This must be one of the fixed values provided when creating the
            `CO2System`, i.e., listed in its `nodes_original` attribute.

        Returns
        -------
        float
            The gradient of `var_of` with respect to `var_wrt`.
        """
        var_of = self.shortcuts[var_of]
        var_wrt = self.shortcuts[var_wrt]
        assert var_wrt in self.nodes_original, (
            "`var_wrt` must be one of `sys.nodes_original!`"
        )
        try:  # see if we've already calculated this value
            d_of__d_wrt = self.grads[var_of][var_wrt]
        except KeyError:  # Do the calculations only if needed
            # We need to know the shape of the variable that we want the grad
            # of.  The easiest way to get this is just to solve for it (if that
            # hasn't already been done)
            if var_of not in self.data:
                self.solve(var_of)
            # Next, we extract the originally set values, which are fixed
            # during the differentiation
            other_values_original = {
                k: self.data[k] for k in self.nodes_original if k != var_wrt
            }
            # We have to make sure the value we are differentiating with
            # respect to has the same shape as the value we want the
            # derivative of
            value_wrt = self.data[var_wrt] * np.ones_like(self.data[var_of])
            # Here we compute the gradient
            grad_func = self.get_grad_func(var_of, var_wrt)
            d_of__d_wrt = grad_func(value_wrt, **other_values_original)
            # Put the final value into self.grads, first creating a new
            # sub-dict if necessary
            if var_of not in self.grads:
                self.grads[var_of] = ShortcutDotDict(self.shortcuts)
            self.grads[var_of][var_wrt] = d_of__d_wrt
            self.remove_jax_overhead(self.grads[var_of])
        return d_of__d_wrt

    def get_grads(
        self,
        vars_of: str | list,
        vars_wrt: str | list,
    ):
        """Compute the derivatives of `vars_of` with respect to `vars_wrt` and
        store them in `sys.grads[var_of][var_wrt]`.

        Parameters
        ----------
        vars_of : str | list
            The name(s) of the variable(s) to get the derivative(s) of.
        vars_wrt : str | list
            The name(s) of the variable(s) to get the derivative(s) with
            respect to.  These must all be one of the fixed parameters
            provided on initialisation, i.e., listed in `nodes_original`.

        Returns
        -------
        FunctionGraph
            The `FunctionGraph` with the additional gradients computed.
        """
        if isinstance(vars_of, str):
            vars_of = [vars_of]
        if isinstance(vars_wrt, str):
            vars_wrt = [vars_wrt]
        for var_of, var_wrt in product(vars_of, vars_wrt):
            self.get_grad(var_of, var_wrt)
        return self

    def set_uncertainty(self, **kwargs):
        """Assign independent uncertainties for parameters.

        The values should be the 1-sigma independent uncertainty in each
        parameter.  These can be single scalar values, or arrays of the same
        shape as the corresponding parameter.
        """
        uset = []
        for k, v in kwargs.items():
            if k.lower().endswith("__f"):
                skl = self.shortcuts[k[:-3]] + "__f"
            else:
                skl = self.shortcuts[k]
            if skl in uset:
                raise SyntaxError(
                    f"Keyword argument repeated, possibly with a different alias: {k}"
                )
            uset.append(skl)
            if skl not in self.nodes_original:
                raise Exception(
                    "Uncertainty can be assigned only for user-provided parameters"
                )
            self.uncertainty.assign(**{skl: v})
        # # Recalculate any uncertainties that have already been propagated
        # self.propagate([self.shortcuts[k] for k in self.uncertainty])
        return self

    set_u = set_uncertainty

    # def _propagate(self, uncertainty_into, uncertainty_from):
    #     for var_in in uncertainty_into:
    #         # This should always be reset to zero and all values wiped, even if
    #         # it already exists (so you don't end up with old uncertainty_from
    #         # components from a previous calculation which are no longer part of
    #         # the total)
    #         self.uncertainty[var_in] = np.zeros_like(self.data[var_in])
    #         u_total = self.uncertainty[var_in]
    #         for var_from, u_from in uncertainty_from.items():
    #             is_fractional = var_from.endswith("__f")
    #             if is_fractional:
    #                 # If the uncertainty is fractional, multiply through
    #                 var_from = var_from[:-3]
    #                 u_from = self.data[var_from] * u_from
    #             # Propagate uncertainties only from ancestor nodes
    #             if var_from in nx.ancestors(self.graph, var_in):
    #                 if var_from in self.nodes_original:
    #                     self.get_grad(var_in, var_from)
    #                     u_part = np.abs(self.grads[var_in][var_from] * u_from)
    #                 else:
    #                     # If the uncertainty is from some internally calculated value,
    #                     # then we need to make a second CO2System where that value
    #                     # is one of the known inputs, and get the grad from that
    #                     data = self.get_values_original()
    #                     data.update({var_from: self.data[var_from]})
    #                     sys = CO2System(**data, **self.opts)
    #                     sys.get_grad(var_in, var_from)
    #                     u_part = np.abs(sys.grads[var_in][var_from] * u_from)
    #                 if is_fractional:
    #                     var_from += "__f"
    #                 if var_in not in self.uncertainty.parts:
    #                     self.uncertainty.parts[var_in] = ShortcutDotDict()
    #                 self.uncertainty.parts[var_in][var_from] = u_part
    #                 u_total = u_total + u_part**2
    #         self.uncertainty[var_in] = np.sqrt(u_total)
    #     return self

    @staticmethod
    def get_graph(funcs: dict) -> nx.DiGraph:
        graph = nx.DiGraph()
        for k, func in funcs.items():
            for f in signature(func).parameters.keys():
                graph.add_edge(f, k)
        nx.set_node_attributes(graph, funcs, name="func")
        args = {}
        for node, attrs in graph.nodes.items():
            if "func" in attrs:
                args[node] = list(signature(attrs["func"]).parameters)
        nx.set_node_attributes(graph, args, name="args")
        return graph

    @staticmethod
    def remove_jax_overhead(data: dict):
        for k, v in data.items():
            try:
                data[k] = v.item()
            except (AttributeError, ValueError):
                pass
            try:
                data[k] = v.__array__()
            except AttributeError:
                pass
