from warnings import warn

import networkx as nx

from .engine import set_node_labels


def plot_graph(
    self,
    ax=None,
    exclude_nodes=None,
    show_unknown=False,
    keep_unknown=None,
    show_isolated=False,
    skip_nodes=None,
    prog_graphviz=None,
    root_graphviz=None,
    args_graphviz="",
    nx_layout=nx.spring_layout,
    nx_args=None,
    nx_kwargs=None,
    node_kwargs=None,
    edge_kwargs=None,
    label_kwargs=None,
    mode="state",
    nan_invalid=False,
):
    """Draw a graph showing the relationships between the different parameters.

    Parameters
    ----------
    ax : matplotlib axes, optional
        The axes on which to plot.  If `None`, a new figure and axes are created.
    exclude_nodes : list of str, optional
        List of nodes to exclude from the plot, by default `None`.  Nodes in
        this list are not shown, nor are connections to them or through them.
    prog_graphviz : str, optional
        Name of Graphviz layout program, by default "neato".
    show_unknown : bool, optional
        Whether to show nodes for parameters that have not (yet) been calculated,
        by default `True`.
    show_isolated : bool, optional
        Whether to show nodes for parameters that are not connected to the
        graph, by default `True`.
    skip_nodes : bool, optional
        List of nodes to skip from the plot, by default `None`.  Nodes in this
        list are not shown, but the connections between their predecessors
        and children are still drawn.

    Returns
    -------
    matplotlib axes
        The axes on which the graph is plotted.
    """
    from matplotlib import pyplot as plt

    # NODE STATES
    # -----------
    # no state (grey) = unknwown
    # 1 (grass) = provided by user (or default) i.e. known but not calculated
    # 2 (azure) = calculated en route to a user-requested parameter
    # 3 (tangerine) = calculated after direct user request
    #
    # EDGE STATES
    # -----------
    # no state (grey) = calculation not performed
    # 2 = (azure) calculation performed
    #
    if ax is None:
        ax = plt.subplots(dpi=300, figsize=(8, 7))[1]
    if mode == "valid":
        if not self.checked_valid:
            self.check_valid(nan_invalid=nan_invalid)
    graph_to_plot = self.get_graph_to_plot(
        exclude_nodes=exclude_nodes,
        show_unknown=show_unknown,
        keep_unknown=keep_unknown,
        show_isolated=show_isolated,
        skip_nodes=skip_nodes,
    )
    pos = self.get_graph_pos(
        graph_to_plot=graph_to_plot,
        prog_graphviz=prog_graphviz,
        root_graphviz=root_graphviz,
        args_graphviz=args_graphviz,
        nx_layout=nx_layout,
        nx_args=nx_args,
        nx_kwargs=nx_kwargs,
    )
    if mode == "state":
        node_states = nx.get_node_attributes(graph_to_plot, "state", default=0)
        edge_states = nx.get_edge_attributes(graph_to_plot, "state", default=0)
        node_colour = [self.c_state[node_states[n]] for n in nx.nodes(graph_to_plot)]
        edge_colour = [self.c_state[edge_states[e]] for e in nx.edges(graph_to_plot)]
    elif mode == "valid":
        node_valid = nx.get_node_attributes(graph_to_plot, "valid", default=0)
        edge_valid = nx.get_edge_attributes(graph_to_plot, "valid", default=0)
        node_valid_p = nx.get_node_attributes(graph_to_plot, "valid_p", default=0)
        node_colour = [self.c_valid[node_valid[n]] for n in nx.nodes(graph_to_plot)]
        edge_colour = [self.c_valid[edge_valid[e]] for e in nx.edges(graph_to_plot)]
        node_edgecolors = [
            self.c_valid[node_valid_p[n]] for n in nx.nodes(graph_to_plot)
        ]
        node_linewidths = [[0, 2][node_valid_p[n]] for n in nx.nodes(graph_to_plot)]
    else:
        warn(f'mode "{mode}" not recognised, options are "state" or "valid".')
        node_colour = "xkcd:grey"
        edge_colour = "xkcd:grey"
    node_labels = {k: k for k in graph_to_plot.nodes}
    for k, v in set_node_labels.items():
        if k in node_labels:
            node_labels[k] = v
    if node_kwargs is None:
        node_kwargs = {}
    if edge_kwargs is None:
        edge_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}
    if mode == "valid":
        node_kwargs["edgecolors"] = node_edgecolors
        node_kwargs["linewidths"] = node_linewidths
    nx.draw_networkx_nodes(
        graph_to_plot,
        ax=ax,
        node_color=node_colour,
        pos=pos,
        **node_kwargs,
    )
    nx.draw_networkx_edges(
        graph_to_plot,
        ax=ax,
        edge_color=edge_colour,
        pos=pos,
        **edge_kwargs,
    )
    nx.draw_networkx_labels(
        graph_to_plot,
        ax=ax,
        labels=node_labels,
        pos=pos,
        **label_kwargs,
    )
    return ax
