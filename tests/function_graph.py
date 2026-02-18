from collections import UserDict
from inspect import signature
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


def remove_jax_overhead(data):
    for k, v in data.items():
        try:
            data[k] = v.item()
        except (AttributeError, ValueError):
            pass
        try:
            data[k] = v.__array__()
        except AttributeError:
            pass


def egrad(g):
    # From https://github.com/google/jax/issues/3556#issuecomment-649779759
    # modified to allow kwargs for g
    def wrapped(x, *args, **kwargs):
        y, g_vjp = jax.vjp(lambda x: g(x, *args, **kwargs), x)
        (x_bar,) = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped


class ShortcutDict(UserDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        try:
            return self.data[key.lower()]
        except KeyError:
            return key.lower()


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
            self.graph = get_graph(funcs)
        if shortcuts is not None:
            self.shortcuts = ShortcutDict(**shortcuts)
        else:
            self.shortcuts = ShortcutDict()
        self.ignored = set()
        self.requested = set()
        self.nodes_original = set()

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
        remove_jax_overhead(self.data)

    def get_func_of(self, var_of):
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

    def get_grad_func(self, var_of, var_wrt):
        get_value_of = self.get_func_of(var_of)
        get_value_of_from_wrt = self.get_func_of_from_wrt(get_value_of, var_wrt)
        return egrad(get_value_of_from_wrt)
