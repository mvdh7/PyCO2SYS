# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
from collections import UserDict
from inspect import signature

import jax
import jax.numpy as np
import networkx as nx
from jax import jacfwd

from ..meta import egrad, warn


jax.config.update("jax_enable_x64", True)


# NODE STATES
# ===========
# -1 = value unknown
#  0 = default value
#  1 = user-provided value
#  2 = calculated as intermediate
#  3 = calculated by explicit request


class FunctionGraphError(Exception):
    """FunctionGraph custom exception."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


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

    def __getitem__(self, key):
        return self.data[self._shortcuts[key]]


class Uncertainties(ShortcutDotDict):
    def __init__(self, shortcuts):
        super().__init__(shortcuts)
        self.assigned = ShortcutDotDict(shortcuts)
        self.parts = ShortcutDotDict(shortcuts)

    def assign(self, **uncertainties):
        for k, v in uncertainties.items():
            self.assigned[self._shortcuts[k]] = v


class Range(ShortcutDotDict):
    def __init__(self, shortcuts):
        super().__init__(shortcuts)

    def __repr__(self):
        text = "DIRECTLY ASSIGNED VALID RANGES"
        keys = list(self.data.keys())
        keys.sort()
        for k in keys:
            if k == keys[-1]:
                text += f"\n└─ {k}"
            else:
                text += f"\n├─ {k}"
            leys = list(self.data[k].keys())
            leys.sort()
            for l in leys:
                w = self.data[k][l]
                end = f"{l}: {w[0]} to {w[1]}"
                if k == keys[-1]:
                    if l == leys[-1]:
                        text += "\n   └─ " + end
                    else:
                        text += "\n   ├─ " + end
                else:
                    if l == leys[-1]:
                        text += "\n│  └─ " + end
                    else:
                        text += "\n│  ├─ " + end
        return text


class Valids(ShortcutDotDict):
    def __init__(self, shortcuts):
        super().__init__(shortcuts)
        self.direct = ShortcutDotDict(shortcuts)
        self.indirect = ShortcutDotDict(shortcuts)
        self.range = Range(shortcuts)

    def why(self, parameter):
        """Find out why a particular parameter is (in)valid.

        Parameters
        ----------
        parameter : str
            The name of the parameter to investigate (or its shortcut).

        Returns
        -------
        ShortcutDotDict
            A dict with containing the parent parameters of the
            investigated parameter that have an influence on its validity.
            The dict contains two keys, "direct" and "indirect":
              - "direct" contains parameters that the investigated
                parameter has defined validity ranges for.
              - "indirect" contains other parent parameters that may
                themselves be invalid for other reasons.
        """
        p = self._shortcuts[parameter]
        direct = ShortcutDotDict(self._shortcuts)
        if p in self.direct:
            direct.update({p: self.direct[p]})
        indirect = ShortcutDotDict(self._shortcuts)
        if p in self.indirect:
            indirect.update({p: self.indirect[p]})
        out = ShortcutDotDict(self._shortcuts)
        out["direct"] = direct
        out["indirect"] = indirect
        return out

    def get_graph(self):
        graph = nx.DiGraph()
        for d in ["direct", "indirect"]:
            for to, fr_dict in self.__getattr__(d).items():
                for fr, v in fr_dict.items():
                    pct = 100 * np.sum(v) / np.size(v)
                    graph.add_edge(fr, to, type=d, pct=pct)
        for n in graph.nodes:
            if n in self:
                nx.set_node_attributes(
                    graph,
                    {n: 100 * np.sum(self[n]) / np.size(self[n])},
                    name="pct",
                )
        return graph


class FunctionGraph(UserDict):
    def __init__(
        self,
        defaults: dict | None = None,
        graph: nx.DiGraph | None = None,
        funcs: dict | None = None,
        shortcuts: dict | None = None,
        no_store: set | None = None,
    ):
        super().__init__()
        if graph is not None:
            self.graph = graph.copy()
        else:
            if not isinstance(funcs, dict):
                raise FunctionGraphError(
                    "Either `graph` or `funcs` must be provided"
                )
            self.graph = self.get_graph(funcs)
        if defaults is not None:
            self.defaults = {
                k: v for k, v in defaults.items() if k in self.graph.nodes
            }
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
                    "no_store",
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
                    "why",
                ]:
                    raise FunctionGraphError(
                        f'Invalid shortcut "{k}" (reserved attribute)'
                    )
            self.shortcuts = ShortcutsDict(**shortcuts)
        else:
            self.shortcuts = ShortcutsDict()
        if no_store is not None:
            if isinstance(no_store, str):
                self.no_store = set([no_store])  # noqa: C405
            else:
                self.no_store = set(no_store)
        else:
            self.no_store = set()
        self.ignored = set()
        self._requested = set()
        self._nodes_user = set()
        self._nodes_defaults = set()
        self._nodes_original = set()
        self.grads = ShortcutDotDict(self.shortcuts)
        self.jacs = ShortcutDotDict(self.shortcuts)
        self.uncertainty = Uncertainties(self.shortcuts)
        self.u = self.uncertainty
        self.valid = Valids(self.shortcuts)
        self.v = self.valid
        for n in self.graph.nodes:
            sgnn = self.graph.nodes[n]
            if "func" in sgnn and hasattr(sgnn["func"], "valid"):
                self.valid.range[n] = ShortcutDotDict(self.shortcuts)
                self.valid.range[n].update(sgnn["func"].valid)

    def __getitem__(self, key):
        # When the user requests a dict key that hasn't been solved for yet,
        # then solve and provide the requested parameter
        self.solve(parameters=key)
        if isinstance(key, list):
            # If the user provides a list of keys to solve for, return all of
            # them as a dict
            return {k: self.data[self.shortcuts[k]] for k in key}
        else:
            # If a single key is requested, return the corresponding value(s)
            return self.data[self.shortcuts[key]]

    def __getattr__(self, attr):
        # This allows parameter values to be accessed with dot notation, purely
        # for convenience
        if self.shortcuts[attr] in self.graph.nodes:
            return self[attr]
        else:
            return object.__getattribute__(self, attr)

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
            warn(
                "Some arguments were not recognised or not valid for"
                + " this combination of known parameters and are"
                + " being ignored (see `ignored` attribute)",
                stacklevel=3,
            )
        self.ignored |= set(ignored)
        self._nodes_user = {
            self.shortcuts[k] for k, v in data.items() if v is not None
        }

        self._nodes_defaults = {
            self.shortcuts[k]
            for k, v in self_defaults.items()
            if v is not None
        }
        self._nodes_original = self._nodes_user | self._nodes_defaults
        return self

    def solve(
        self,
        parameters: set | list | str | None = None,
        store_steps: int = 1,
    ):
        """Solve for the requested parameter(s).

        Parameters
        ----------
        parameters : set | list | str | None, optional
            Which parameters (or their shortcuts) to solve for, by
            default None, in which case all possible parameters are
            solved for.
        store_steps : int, optional
            Whether to save no (0), some (1) or all (2) intermediate
            parameters while solving for the requested parameters.

        Returns
        -------
        FunctionGraph
            The FunctionGraph with the requested parameters calculated.
        """
        if store_steps not in [0, 1, 2]:
            raise FunctionGraphError("`store_steps` must be 0, 1 or 2")
        if parameters is None:
            parameters = list(self.graph.nodes)
        elif isinstance(parameters, str):
            parameters = [parameters]
        parameters = {self.shortcuts[p] for p in parameters}
        self._requested |= parameters
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
                    if store_steps == 2 or (
                        store_steps == 1 and p not in self.no_store
                    ):
                        nx.set_node_attributes(
                            self.graph, {p: 2}, name="state"
                        )
            except KeyError:
                raise FunctionGraphError(
                    f"{p} has no associated function in the graph"
                )
        if store_steps < 2:
            for p in parameters_all:
                if (
                    store_steps == 0 or p in self.no_store
                ) and p not in parameters:
                    self.data.pop(p)
        self.remove_jax_overhead(self.data)
        return self

    def get_func_of(self, var_of: str):
        """Create a function to compute `var_of` directly from an input
        set of values.

        The created function has the signature

            value_of = get_value_of(**kwargs)

        where the `kwargs` are the originally user-defined and default
        values, obtained with

            kwargs = {k: fg[k] for k in fg._nodes_original}
        """
        # We get a sub-graph of the node of interest and all its ancestors,
        # excluding originally fixed / user-defined values
        var_of = self.shortcuts[var_of]
        nodes_vo_all = nx.ancestors(self.graph, var_of)
        nodes_vo_all.add(var_of)
        nodes_vo = [n for n in nodes_vo_all if n not in self._nodes_original]
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
        for p in self._nodes_original:
            if p in nodes_vo_all:
                get_value_of.__doc__ += f"\n        {p}"
        get_value_of.__doc__ += "\n\nReturns\n-------"
        get_value_of.__doc__ += f"\n{var_of}"
        get_value_of.args_list = [
            n for n in self._nodes_original if n in nodes_vo_all
        ]
        return get_value_of

    def get_func_of_from_wrt(self, get_value_of, var_wrt):
        """Reorganise a function created with `_get_func_of` so that one
        of its kwargs is instead a positional arg (and which can thus be
        gradded).

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
        assert var_wrt in self._nodes_original, (
            "`var_wrt` must be one of `self._nodes_original!`"
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
                k: self.data[k] for k in self._nodes_original
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
        """Compute the derivatives of `vars_of` with respect to `vars_wrt`
        and store them in `sys.grads[var_of][var_wrt]`.

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
            The Jacobian of `var_of` with respect to `var_wrt`.
            Its dimensions are `*(np.shape(var_of), *np.shape(var_wrt))`.
        """
        var_of = self.shortcuts[var_of]
        var_wrt = self.shortcuts[var_wrt]
        assert var_wrt in self._nodes_original, (
            "`var_wrt` must be one of `sys._nodes_original!`"
        )
        try:  # see if we've already calculated this value
            d_of__d_wrt = self.jacs[var_of][var_wrt]
        except KeyError:  # Do the calculations only if needed
            if var_of not in self.data:
                self.solve(var_of)
            # Next, we extract the originally set values, which are fixed
            # during the differentiation
            other_values_original = {
                k: self.data[k] for k in self._nodes_original if k != var_wrt
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
        """Assign uncertainties for parameters.

        The values should be the uncertainty as a variance in each parameter.
        Each uncertainty can be can be
          - a single scalar value,
          - an array of the same shape as the corresponding parameter, or
          - a covariance matrix.
        """
        uset = []
        for k, v in kwargs.items():
            skl = self.shortcuts[k]
            if skl in uset:
                raise SyntaxError(
                    "Keyword argument repeated, "
                    + f"possibly with a different shortcut: {k}"
                )
            uset.append(skl)
            if skl not in self._nodes_original:
                raise FunctionGraphError(
                    "Uncertainty can be assigned only for "
                    + "user-provided parameters"
                )
            v_np = v
            if isinstance(v, list):
                v_np = np.array(v)
            self.uncertainty.assign(**{skl: v_np})
        # Recalculate any uncertainties that have already been propagated
        self.propagate([self.shortcuts[k] for k in self.uncertainty])
        return self

    def propagate(
        self,
        uncertainty_into: str | list[str] | None = None,
        keep_cov: bool = True,
        store_parts: bool = True,
    ):
        """Propagate uncertainties from all parameters with assigned
        uncertainties into the requested set of parameters.

        Parameters
        ----------
        uncertainty_into : str | list[str], optional
            Which parameters to propagate uncertainty into, by default
            None, in which case the list of parameters in
            self._requested is used.
        keep_cov : bool, optional
            Whether to keep covariance terms in the final results,
            by default True.
        store_parts : bool, optional
            Whether the save the separate uncertainty components,
            by default True.
        """
        if uncertainty_into is None:
            uncertainty_into = list(self._requested)
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
            if ui in self.u.parts and store_parts:
                self.remove_jax_overhead(self.u.parts[ui])
            if not keep_cov:
                self.u[ui] = self.cut_cov(self.u[ui])
        self.remove_jax_overhead(self.u)
        return self

    set_u = set_uncertainty
    prop = propagate

    def check_valid(self, parameters=None):
        if parameters is None:
            parameters = list(self._requested)
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
                    if n not in sv.direct:
                        sv.direct[n] = ShortcutDotDict(self.shortcuts)
                    for k, v in sgn[n]["func"].valid.items():
                        sv.direct[n][k] = (self[k] >= v[0]) & (self[k] <= v[1])
                        sv[n] &= sv.direct[n][k]
                for p in self.graph.predecessors(n):
                    if p in sv:
                        if n not in sv.indirect:
                            sv.indirect[n] = ShortcutDotDict(self.shortcuts)
                        if n not in sv:
                            sv[n] = ~np.isnan(self[n])
                        sv.indirect[n][p] = sv[p]
                        sv[n] &= sv.indirect[n][p]
        return self

    def keys_all(self):
        return tuple(self.graph.nodes)

    @staticmethod
    def get_graph(funcs: dict) -> nx.DiGraph:
        """Construct a graph from a dict of functions."""
        graph = nx.DiGraph()
        for k, func in funcs.items():
            for f in signature(func).parameters:
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
        """Get the `np.einsum` subscripts for uncertainty propagation of
        `ux` from `x` to `y`.

        Parameters
        ----------
        x_ndims : int
            The number of dimensions of the variable to propagate
            uncertainties from.
        y_ndims : int
            The number of dimensions of the variable to propagate
            uncertainties into.
        ux_ndims : int
            The number of dimensions of the uncertainties for `x`.
            Should be either the same as, or double, `x_ndims`.

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
        """Propagate uncertainties `ux` from `x` to `y` given the
        Jacobian of `y` with respect to `x` (`jac`).
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
        """Propagate independent uncertainties `ux` from `x` to `y`
        given the derivative of `y` with respect to `x` (`grad_yx`).
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
