from collections import UserDict
from inspect import signature
from itertools import product

import jax
import jax.numpy as np
import networkx as nx
from jax import jacfwd


jax.config.update("jax_enable_x64", True)

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


class Valids(ShortcutDotDict):
    def __init__(self, shortcuts):
        super().__init__(shortcuts)
        self.direct = ShortcutDotDict(shortcuts)
        self.indirect = ShortcutDotDict(shortcuts)


class FunctionGraph(UserDict):
    def __init__(
        self,
        defaults: dict | None = None,
        graph: nx.DiGraph | None = None,
        funcs: dict | None = None,
        shortcuts: dict | None = None,
    ):
        super().__init__()
        if graph is not None:
            self.graph = graph.copy()
        else:
            if not isinstance(funcs, dict):
                raise Exception("Either `graph` or `funcs` must be provided")
            self.graph = self.get_graph(funcs)
        if defaults is not None:
            self.defaults = defaults.copy()
        else:
            self.defaults = {
                n: None
                for n, attrs in self.graph.nodes.items()
                if "func" not in attrs
            }
        if shortcuts is not None:
            for k in shortcuts:
                if k in [
                    "defaults",
                    "direct",
                    "graph",
                    "ignored",
                    "indirect",
                    "jacs",
                    "nodes_original",
                    "parts",
                    "prop",
                    "propagate",
                    "requested",
                    "set_data",
                    "set_u",
                    "set_uncertainty",
                    "shortcuts",
                    "solve",
                    "u",
                    "uncertainty",
                    "v",
                    "valid",
                ]:
                    raise Exception(
                        f'Invalid shortcut "{k}" (reserved attribute)'
                    )
            self.shortcuts = ShortcutsDict(**shortcuts)
        else:
            self.shortcuts = ShortcutsDict()
        self.ignored = set()
        self.requested = set()
        self.nodes_original = set()
        self.grads = ShortcutDotDict(self.shortcuts)
        self.jacs = ShortcutDotDict(self.shortcuts)
        self.uncertainty = Uncertainties(self.shortcuts)
        self.u = self.uncertainty
        self.valid = Valids(self.shortcuts)
        self.v = self.valid

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
        parameters = {self.shortcuts[p] for p in parameters}
        self.requested |= parameters
        keys_known = list(self.data.keys())
        # Remove known nodes from a copy of self.graph, so that ancestors of
        # known nodes are not unnecessarily recomputed
        graph_unknown = self.graph.copy()
        graph_unknown.remove_nodes_from(
            [k for k in keys_known if k not in parameters]
        )
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
                self.data[p] = attrs["func"](
                    *[self.data[r] for r in attrs["args"]]
                )
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
        get_value_of.args_list = [
            n for n in self.nodes_original if n in nodes_vo_all
        ]
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
        return get_value_of_from_wrt

    def get_grad_func(self, var_of: str, var_wrt: str):
        get_value_of = self.get_func_of(var_of)
        get_value_of_from_wrt = self.get_func_of_from_wrt(
            get_value_of, var_wrt
        )
        return egrad(get_value_of_from_wrt)

    def get_grad(self, var_of, var_wrt):
        """Compute the derivative of `var_of` with respect to `var_wrt`.
        If there is already a value in `sys.grads[var_of][var_wrt]`,
        then that value is returned instead of recalculating.

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
            "`var_wrt` must be one of `self.nodes_original!`"
        )
        try:  # see if we've already calculated this value
            d_of__d_wrt = self.grads[var_of][var_wrt]
        except (
            KeyError
        ):  # only do the calculations if there isn't already a value
            # We need to know the shape of the variable that we want the grad
            # of, the easy way to get this is just to solve for it (if that
            # hasn't already been done)
            if var_of not in self.data:
                self.solve(var_of)
            # Next, we extract the originally set values, which are fixed
            # during the differentiation
            other_values_original = {
                k: self.data[k] for k in self.nodes_original
            }
            # We have to make sure the value we are differentiating with
            # respect to has the same shape as the value we want the
            # differential of
            value_wrt = other_values_original.pop(var_wrt) * np.ones_like(
                self.data[var_of]
            )
            # Here we compute the gradient
            grad_func = self.get_grad_func(var_of, var_wrt)
            d_of__d_wrt = grad_func(value_wrt, **other_values_original)
        return d_of__d_wrt

    def get_grads(self, vars_of, vars_wrt):
        """Compute the derivatives of `vars_of` with respect to `vars_wrt` and
        store them in `sys.grads[var_of][var_wrt]`.

        Parameters
        ----------
        vars_of : list
            The names of the variables to get the derivatives of.
        vars_wrt : list
            The names of the variables to get the derivatives with respect to.
            These must all be one of the fixed values listed `nodes_original`.

        Returns
        -------
        CO2System
            The `CO2System` with the additional gradients computed.
        """
        if isinstance(vars_of, str):
            vars_of = [vars_of]
        if isinstance(vars_wrt, str):
            vars_wrt = [vars_wrt]
        for var_of in vars_of:
            var_of = self.shortcuts[var_of]
            if var_of not in self.grads:
                self.grads[var_of] = ShortcutDotDict(self.shortcuts)
            for var_wrt in vars_wrt:
                var_wrt = self.shortcuts[var_wrt]
                self.grads[var_of][var_wrt] = self.get_grad(var_of, var_wrt)
            self.remove_jax_overhead(self.grads[var_of])
        return self

    def get_jac_func(self, var_of: str, var_wrt: str):
        get_value_of = self.get_func_of(var_of)
        get_value_of_from_wrt = self.get_func_of_from_wrt(
            get_value_of, var_wrt
        )
        return jacfwd(get_value_of_from_wrt)

    def get_jac(self, var_of: str, var_wrt: str):
        """Compute the Jacobian of `var_of` with respect to `var_wrt`.
        If there is already a value in `sys.jacs[var_of][var_wrt]`,
        then that value is returned instead of recalculating.

        Parameters
        ----------
        var_of : str
            The name of the variable to get the Jacobian of.
        var_wrt : str
            The name of the variable to get the Jacobian with respect to.
            This must be one of the fixed values listed in `nodes_original`.

        Returns
        -------
        float
            The Jacobian of `var_of` with respect to `var_wrt`.  Its dimensions
            are `*(np.shape(var_of), *np.shape(var_wrt))`.
        """
        var_of = self.shortcuts[var_of]
        var_wrt = self.shortcuts[var_wrt]
        assert var_wrt in self.nodes_original, (
            "`var_wrt` must be one of `sys.nodes_original!`"
        )
        try:  # see if we've already calculated this value
            d_of__d_wrt = self.jacs[var_of][var_wrt]
        except KeyError:  # Do the calculations only if needed
            if var_of not in self.data:
                self.solve(var_of)
            # Next, we extract the originally set values, which are fixed
            # during the differentiation
            other_values_original = {
                k: self.data[k] for k in self.nodes_original if k != var_wrt
            }
            # Here we compute the Jacobian
            jac_func = self.get_jac_func(var_of, var_wrt)
            d_of__d_wrt = jac_func(self.data[var_wrt], **other_values_original)
        return d_of__d_wrt

    def get_jacs(
        self,
        vars_of: str | list,
        vars_wrt: str | list,
        store_jacs: bool = True,
    ):
        """Compute the Jacobians of `vars_of` with respect to `vars_wrt` and
        store them in `sys.jacs[var_of][var_wrt]`.

        Parameters
        ----------
        vars_of : str | list
            The name(s) of the variable(s) to get the Jacobian(s) of.
        vars_wrt : str | list
            The name(s) of the variable(s) to get the Jacobian(s) with
            respect to.  These must all be one of the fixed parameters
            provided on initialisation, i.e., listed in `nodes_original`.
        """
        if isinstance(vars_of, str):
            vars_of = [vars_of]
        if isinstance(vars_wrt, str):
            vars_wrt = [vars_wrt]
        for var_of in vars_of:
            var_of = self.shortcuts[var_of]
            if var_of not in self.jacs:
                self.jacs[var_of] = ShortcutDotDict(self.shortcuts)
            for var_wrt in vars_wrt:
                var_wrt = self.shortcuts[var_wrt]
                self.jacs[var_of][var_wrt] = self.get_jac(var_of, var_wrt)
            self.remove_jax_overhead(self.jacs[var_of])
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
        self.propagate([self.shortcuts[k] for k in self.uncertainty])
        return self

    def propagate(
        self,
        uncertainty_into: str | list[str] = None,
        keep_cov: bool = True,
        store_parts: bool = True,
    ):
        """Propagate uncertainties from all parameters with assigned
        uncertainties into the requested set of parameters.

        Parameters
        ----------
        uncertainty_into : str | list[str], optional
            Which parameters to propagate uncertainty into, by default `None`,
            in which case the list of parameters in `self.requested` is used.
        keep_cov : bool, optional
            Whether to keep covariance terms in the final results, by default
            `True`.
        store_parts : bool, optional
            Whether the save the separate uncertainty components, by default
            `True`.

        Returns
        -------
        _type_
            _description_
        """
        if uncertainty_into is None:
            uncertainty_into = list(self.requested)
        elif isinstance(uncertainty_into, str):
            uncertainty_into = [uncertainty_into]
        uncertainty_into = {self.shortcuts[ui] for ui in uncertainty_into}
        for ui in uncertainty_into:
            self.u[ui] = 0
            for uf in self.u.assigned:
                x = self[uf]
                y = self[ui]
                ux = self.u.assigned[uf]
                if store_parts and ui not in self.u.parts:
                    self.u.parts[ui] = ShortcutDotDict(self.shortcuts)
                # To avoid potentially creating unnecessary large sparse
                # arrays, we only want to use a Jacobian if we really need to.
                # Otherwise, an element-wise grad will do.
                if not self.graph.nodes[uf]["coeffs"] and (
                    np.shape(ux) == () or np.shape(ux) == np.shape(x)
                ):
                    grad_yx = self.get_grad(ui, uf)
                    part = self._propagate_grad(grad_yx, ux)
                    self.u[ui] = self.u[ui] + self.expand_zero_cov(part)
                else:
                    jac = self.get_jac(ui, uf)
                    part = self._propagate_jac(x, y, jac, ux)
                    self.u[ui] = self.u[ui] + part
                if store_parts:
                    self.u.parts[ui][uf] = part
            if store_parts:
                self.remove_jax_overhead(self.u.parts[ui])
            if not keep_cov:
                self.u[ui] = self.cut_cov(self.u[ui])
        self.remove_jax_overhead(self.u)
        return self

    set_u = set_uncertainty
    prop = propagate

    def get_valid(self, parameters=None):
        if parameters is None:
            parameters = list(self.requested)
        elif isinstance(parameters, str):
            parameters = [parameters]
        parameters = {self.shortcuts[p] for p in parameters}
        # Add intermediate parameters that we need to know in order to
        # calculate the requested parameters
        parameters_all = parameters.copy()
        for p in parameters:
            parameters_all = parameters_all | nx.ancestors(self.graph, p)
        sgn = self.graph.nodes
        sv = self.valid
        for n in nx.topological_sort(self.graph):
            if n in parameters_all:
                if "func" in sgn[n] and hasattr(sgn[n]["func"], "valid"):
                    sv[n] = ~np.isnan(self[n])
                    sv.direct[n] = ShortcutDotDict(self.shortcuts)
                    for k, v in sgn[n]["func"].valid.items():
                        sv.direct[n][k] = (self[k] >= v[0]) & (self[k] <= v[1])
                        sv[n] &= sv.direct[n][k]
                for p in self.graph.predecessors(n):
                    if p in sv:
                        print(n, p)
                        sv.indirect[n] = ShortcutDotDict(self.shortcuts)
                        if n not in sv:
                            sv[n] = ~np.isnan(self[n])
                        sv.indirect[n][p] = sv[p]
                        sv[n] &= sv.indirect[n][p]
        return self

    @staticmethod
    def get_graph(funcs: dict) -> nx.DiGraph:
        """Construct a graph from a dict of functions."""
        graph = nx.DiGraph()
        for k, func in funcs.items():
            for f in signature(func).parameters.keys():
                graph.add_edge(f, k)
        nx.set_node_attributes(graph, funcs, name="func")
        args = {}
        coeffs = {}
        for node, attrs in graph.nodes.items():
            if "func" in attrs:
                args[node] = list(signature(attrs["func"]).parameters)
            if node.startswith("coeffs"):
                coeffs[node] = True
            else:
                coeffs[node] = False
        nx.set_node_attributes(graph, args, name="args")
        nx.set_node_attributes(graph, coeffs, name="coeffs")
        return graph

    @staticmethod
    def remove_jax_overhead(data: dict):
        """Remove the JAX overhead on all values in a dict."""
        for k, v in data.items():
            try:
                data[k] = v.item()
            except (AttributeError, ValueError):
                pass
            try:
                data[k] = v.__array__()
            except AttributeError:
                pass

    @staticmethod
    def get_einsum_code(
        x_ndims: int,
        y_ndims: int,
        ux_ndims: int,
    ) -> str:
        """Get the `np.einsum` subscripts for uncertainty propagation of `ux`
        from `x` to `y`.

        Parameters
        ----------
        x_ndims : int
            The number of dimensions of the variable to propagate uncertainties
            from.
        y_ndims : int
            The number of dimensions of the variable to propagate uncertainties
            into.
        ux_ndims : int
            The number of dimensions of the uncertainties for `x`.  Should be
            either the same as, or double, `x_ndims`.

        Returns
        -------
        subscripts : str
            The subscripts to use with `np.einsum`:
                `uy = np.einsum(subscripts, jac_yx, ux, jac_yx)`
        """
        i0 = 97
        A = ""
        for i in range(y_ndims):
            A += chr(i0)
            i0 += 1
        B = ""
        for i in range(x_ndims):
            B += chr(i0)
            i0 += 1
        if x_ndims == ux_ndims:
            C = B
        else:
            C = ""
            for i in range(x_ndims):
                C += chr(i0)
                i0 += 1
        D = ""
        for i in range(y_ndims):
            D += chr(i0)
            i0 += 1
        if B == C:
            return f"{A}{B},{B},{D}{C}->{A}{D}"
        else:
            return f"{A}{B},{B}{C},{D}{C}->{A}{D}"

    @staticmethod
    def _propagate_jac(
        x: float | np.ndarray,
        y: float | np.ndarray,
        jac: float | np.ndarray,
        ux: float | np.ndarray,
    ) -> np.ndarray:
        """Propagate uncertainties `ux` from `x` to `y` given the Jacobian of
        `y` with respect to `x` (`jac`).
        """
        x_ndims = len(np.shape(x))
        y_ndims = len(np.shape(y))
        ux_ndims = len(np.shape(ux))
        if ux_ndims == 0:
            ux_ndims = x_ndims
            ux = np.full_like(x, ux)
        subscripts = FunctionGraph.get_einsum_code(x_ndims, y_ndims, ux_ndims)
        uy = np.einsum(subscripts, jac, ux, jac)
        return uy

    @staticmethod
    def _propagate_grad(
        grad_yx: float | np.ndarray,
        ux: float | np.ndarray,
    ):
        """Propagate independent uncertainties `ux` from `x` to `y` given the
        derivative of `y` with respect to `x` (`grad_yx`).
        """
        uy = ux * grad_yx**2
        return uy

    @staticmethod
    def cut_cov(uncert):
        """Collapse a multidimensional uncertainty matrix to remove covariance
        terms, equivalent to taking the main diagonal from a 2D matrix.
        """
        ushape = np.shape(uncert)
        if ushape == ():
            return uncert
        else:
            # `ixs` is "aa->a", "abab->ab", "abcabc->abc", ...
            ixs = "".join(chr(97 + i) for i in range(int(len(ushape) / 2)))
            return np.einsum(ixs + ixs + "->" + ixs, uncert)

    @staticmethod
    def expand_zero_cov(v):
        """Inverse of `cut_cov`."""
        vc = np.zeros((*np.shape(v), *np.shape(v)))
        for i, val in enumerate(v.ravel()):
            ix = np.unravel_index(i, np.shape(v))
            vc = vc.at[*ix, *ix].set(val)
        return vc
