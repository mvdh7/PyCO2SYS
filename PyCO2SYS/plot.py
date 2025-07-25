from warnings import warn

import networkx as nx
import numpy as np

from .engine import set_node_labels


def plot_graph(
    self,
    ax=None,
    backend="matplotlib",
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
    backend : str, optional
        Which backend to use to make the plot, either "matplotlib" (default) or
        "plotly".
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
    assert backend.lower() in ["matplotlib", "plotly"], (
        '`backend` must be either "matplotlib" or "plotly"'
    )
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
        warn('`mode` not recognised, options are "state" or "valid".')
        node_colour = self.c_state[0]
        edge_colour = self.c_state[0]
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
    if backend.lower() == "matplotlib":
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.subplots(dpi=300, figsize=(8, 7))[1]
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
    elif backend.lower() == "plotly":
        import plotly.graph_objects as go

        # Create edge traces
        edge_traces = []
        if mode.lower() == "state":
            colours = self.c_state
        elif mode.lower() == "valid":
            colours = self.c_valid
        for s, c in colours.items():
            edge_x = []
            edge_y = []
            for edge, state in nx.get_edge_attributes(
                graph_to_plot, mode.lower(), default=0
            ).items():
                if state == s:
                    edge_x.append(pos[edge[0]][0])
                    edge_x.append(pos[edge[1]][0])
                    edge_x.append(None)
                    edge_y.append(pos[edge[0]][1])
                    edge_y.append(pos[edge[1]][1])
                    edge_y.append(None)
            edge_traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(
                        color=c,
                        width=0.9,
                    ),
                    hoverinfo="none",
                    opacity=0.6,
                    mode="lines",
                    showlegend=False,
                )
            )
        # Create node traces
        node_text = []
        node_x = []
        node_y = []
        for n, (x, y) in pos.items():
            node_text.append(n)
            node_x.append(x)
            node_y.append(y)
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            hoverinfo="text",
            mode="markers",
            marker=dict(
                color=node_colour,
                size=10,
            ),
            showlegend=False,
        )
        node_trace.text = node_text
        if mode.lower() == "valid":
            node_trace.marker.line = dict(
                color=node_edgecolors,
                width=list(np.array(node_linewidths) * 0.6),
            )
        layout = go.Layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig = go.Figure([*edge_traces, node_trace], layout=layout)
        return fig
