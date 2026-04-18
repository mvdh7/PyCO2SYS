def _adjust_1p(
    self,
    temperature=None,
    bh=None,
    method_fCO2=1,
    which_fCO2_insitu=2,
):
    temperature = self._adjust_prep(temperature)
    bh = self._adjust_prep(bh)
    assert method_fCO2 in [1, 2, 3, 4, 5, 6]
    # To adjust to a different temperature/pressure, we need to know fCO2
    # for the original system.  First, we get the subgraph from the
    # original system that contains only fCO2 and all its ancestors.
    graph_pre = self.graph.subgraph(
        nx.ancestors(self.graph, "fCO2") | {"fCO2"}
    )
    # All of the nodes in graph_pre that are not condition-independent are
    # now renamed with "__pre" appended, to keep the distinct from the same
    # nodes under the adjusted conditions.  Pressure is also considered to
    # be condition-independent as it cannot currently be adjusted.
    no_pre = [*condition_independent, "pressure"]
    graph_pre = nx.relabel_nodes(
        graph_pre,
        {n: n if n in no_pre else n + "__pre" for n in graph_pre.nodes},
    )
    args = {}
    for node, attrs in graph_pre.nodes.items():
        if "func" in attrs:
            args[node] = [
                k if k in no_pre else k + "__pre"
                for k in signature(attrs["func"]).parameters.keys()
            ]
    nx.set_node_attributes(graph_pre, args, name="args")
    # graph_pre can now be merged with a new graph to compute everything
    # from fCO2.  The original system's `opts` are retained.
    graph_adj = nx.compose(graph_pre, assemble_graph(5, self.opts))
    # The new system will have the same set of user-provided parameter
    # values as the original, but the ones that are condition-dependent get
    # renamed with "__pre" appended.
    data_pre = self[self.nodes_original]
    for k, v in data_pre.copy().items():
        if k not in no_pre:
            data_pre[k + "__pre"] = data_pre.pop(k)
    # Here we add the functions that convert fCO2 across temperatures to
    # `graph_adj`, depending on the conversion option.
    cfuncs = {"fCO2": lambda fCO2__pre, exp_upsilon: fCO2__pre * exp_upsilon}
    if method_fCO2 == 1:
        assert which_fCO2_insitu in [1, 2]
        if which_fCO2_insitu == 1:
            cfuncs["bh"] = lambda temperature__pre, salinity, fCO2__pre: (
                upsilon.get_bh_H24(temperature__pre, salinity, fCO2__pre)
            )
        elif which_fCO2_insitu == 2:
            cfuncs["bh"] = (
                lambda temperature__pre, temperature, salinity, fCO2__pre, gas_constant: (
                    upsilon.get_bh_H24(
                        temperature__pre,
                        salinity,
                        fCO2__pre
                        * upsilon.expUps_TOG93_H24(
                            temperature__pre,
                            temperature,
                            gas_constant,
                        ),
                    )
                )
            )
        cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
    elif method_fCO2 == 2:
        cfuncs["bh"] = lambda: upsilon.bh_TOG93_H24
        cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
    elif method_fCO2 == 3:
        cfuncs["bh"] = lambda: upsilon.bh_enthalpy_H24
        cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
    elif method_fCO2 == 4:
        assert bh is not None, (
            "A `bh` value must be provided for `method_fCO2=4`."
        )
        data_pre["bh"] = bh
        no_pre.append("bh")
        cfuncs["exp_upsilon"] = upsilon.expUps_Hoff_H24
    elif method_fCO2 == 5:
        cfuncs["exp_upsilon"] = upsilon.expUps_linear_TOG93
    elif method_fCO2 == 6:
        cfuncs["exp_upsilon"] = upsilon.expUps_quadratic_TOG93
    for k, func in cfuncs.items():
        for f in signature(func).parameters.keys():
            graph_adj.add_edge(f, k)
    nx.set_node_attributes(graph_adj, cfuncs, name="func")
    args = {}
    for node, attrs in graph_adj.nodes.items():
        if node in cfuncs:
            args[node] = list(signature(attrs["func"]).parameters)
    nx.set_node_attributes(graph_adj, args, name="args")
    # Now we can create the new CO2System
    co2a = CO2System(graph=graph_adj, **data_pre, temperature=temperature)
    # Parameters that have already been solved for in the original system
    # are copied across, so that they don't need solving for again.
    for k, v in self.data.items():
        if k not in co2a:
            if k in no_pre:
                co2a.data[k] = v
            else:
                co2a.data[k + "__pre"] = v
    # Uncertainties that were assigned in the original system are copied
    # across.
    uncertainty_pre = {}
    for k, v in self.uncertainty.assigned.items():
        if k in no_pre:
            uncertainty_pre[k] = v
        else:
            uncertainty_pre[k + "__pre"] = v
    co2a.set_uncertainty(**uncertainty_pre)
    # Final housekeeping: the new CO2System will usually get its icase
    # wrong, because it doesn't recognise parameters with keys ending
    # "__pre".  Adjusted systems will get assigned whichever icase the
    # original system had.  This doesn't affect any calculations, but it
    # does affect __str__ and __repr__.
    # TODO make it actually affect __str__ and __repr__
    co2a.icase = self.icase
    co2a.adjusted = True
    co2a.solve(self.requested)
    return co2a


def adjust(self, **kwargs):
    """Adjust the `CO2System` to a different temperature and/or pressure.

    Works differently depending on whether one or two core marine carbonate
    system (MCS) parameters are known.

    If the original `CO2System` was created from a pandas `DataFrame` or
    xarray `Dataset` using the `data` kwarg, then the `temperature` and
    `pressure` provided to `adjust` can be pandas `Series`s or xarray
    `DataArray`s, as long as their index or dimensions are consistent with
    the original `data`.

    Any other system properties (e.g. `salinity`, total salt contents,
    optional settings) must be defined when creating the original,
    unadjusted `CO2System`.  They cannot be added in during the `adjust`
    step.

    Parameters when two core MCS parameters are known
    -------------------------------------------------
    temperature : array-like, optional
        The temperature to adjust to in °C, by default `None`, in which
        case temperature is not adjusted.
    pressure : array-like, optional
        The pressure to adjust to in °C, by default `None`, in which case
        pressure is not adjusted.

    Parameters when one core MCS parameter is known
    -----------------------------------------------
    temperature : array-like
        The temperature to adjust to in °C.
    method_fCO2 : int
        How to do the temperature conversion:
            `1`: using the parameterised υh equation of H24 (default).
            `2`: using the constant υh fitted to the TOG93 dataset by H24.
            `3`: using the constant theoretical υx of H24.
            `4`: following the H24 approach, but using a user-provided `bh`.
            `5`: using the linear fit of TOG93.
            `6`: using the quadratic fit of TOG93.

    Additional parameter when `method_fCO2` is `1`
    ----------------------------------------------
    * `which_fCO2_insitu`: whether the input- (`1`, default) or output-
    (`2`) condition pCO2, fCO2, [CO2(aq)] and/or xCO2 values are at in situ
    conditions, for determining bh with the parameterisation of H24.

    Additional parameter when `method_fCO2` is `4`
    ----------------------------------------------
    bh : array-like
        bh of H24 in J/mol.

    Returns
    -------
    CO2System
        A separate `CO2System` adjusted to the requested temperature and/or
        pressure.
    """
    self_requested = self.requested.copy()
    kwargs = {shortcuts[k.lower()]: v for k, v in kwargs.items()}
    if self.icase == 102:
        self_adjusted = self._adjust_102(**kwargs)
    elif self.icase > 100:
        self_adjusted = self._adjust_2p(**kwargs)
    elif self.icase in [4, 5, 8, 9]:
        self_adjusted = self._adjust_1p(**kwargs)
    else:
        warn("This system cannot be adjusted.")
        self_adjusted = self
    self.requested = self_requested
    return self_adjusted


def get_graph_to_plot(
    self,
    show_unknown=True,
    keep_unknown=None,
    exclude_nodes=None,
    show_isolated=True,
    skip_nodes=None,
):
    graph_to_plot = self.graph.copy()
    # Remove nodes as requested by user
    if not show_unknown:
        if keep_unknown is None:
            keep_unknown = []
        elif isinstance(keep_unknown, str):
            keep_unknown = [keep_unknown]
        node_states = nx.get_node_attributes(graph_to_plot, "state", default=0)
        to_remove = [
            n
            for n, s in node_states.items()
            if s == 0 and n not in keep_unknown
        ]
        graph_to_plot.remove_nodes_from(to_remove)
    # Connect across nodes that are missing due to store_steps=1 mode
    _graph_to_plot = graph_to_plot.copy()
    for n, properties in _graph_to_plot.nodes.items():
        if (
            "state" in properties
            and properties["state"] in [2, 3]
            and len(_graph_to_plot.pred[n]) == 0
            and len(nx.ancestors(self.graph, n)) > 0
        ):
            for a in nx.ancestors(self.graph, n):
                if a in _graph_to_plot.nodes:
                    graph_to_plot.add_edge(a, n, state=2)
    if exclude_nodes:
        # Excluding nodes just makes them disappear from the graph without
        # caring about what they were connected to
        if isinstance(exclude_nodes, str):
            exclude_nodes = [exclude_nodes]
        graph_to_plot.remove_nodes_from(exclude_nodes)
    if not show_isolated:
        graph_to_plot.remove_nodes_from(
            [n for n, d in dict(graph_to_plot.degree).items() if d == 0]
        )
    if skip_nodes:
        # Skipping nodes removes them but then shows their predecessors as
        # being directly connected to their children
        edge_states = nx.get_edge_attributes(graph_to_plot, "state", default=0)
        if isinstance(skip_nodes, str):
            skip_nodes = [skip_nodes]
        for n in skip_nodes:
            for p, s in itertools.product(
                graph_to_plot.predecessors(n), graph_to_plot.successors(n)
            ):
                graph_to_plot.add_edge(p, s)
                if edge_states[(p, n)] + edge_states[(n, s)] == 4:
                    new_state = {(p, s): 2}
                else:
                    new_state = {(p, s): 0}
                nx.set_edge_attributes(graph_to_plot, new_state, name="state")
                edge_states.update(new_state)
            graph_to_plot.remove_node(n)
    return graph_to_plot


def get_graph_pos(
    self,
    graph_to_plot=None,
    prog_graphviz=None,
    root_graphviz=None,
    args_graphviz="",
    nx_layout=nx.spring_layout,
    nx_args=None,
    nx_kwargs=None,
):
    if graph_to_plot is None:
        graph_to_plot = self.graph
    if prog_graphviz is not None:
        pos = nx.nx_agraph.graphviz_layout(
            graph_to_plot,
            prog=prog_graphviz,
            root=root_graphviz,
            args=args_graphviz,
        )
    else:
        if nx_args is None:
            nx_args = ()
        if nx_kwargs is None:
            nx_kwargs = {}
        pos = nx_layout(graph_to_plot, *nx_args, **nx_kwargs)
    return pos


def keys_all(self):
    """Return a tuple of all possible results keys, including those that have
    not yet been solved for.
    """
    return tuple(self.graph.nodes)


def sys(data=None, **kwargs):
    """Initialise a `CO2System`.

    Once initialised, various methods are available including `solve`, `adjust`
    and `propagate`.

    PARAMETERS PROVIDED AS KWARGS
    =============================
    There are many possible kwargs, so they are grouped into sections below.

    Data as separate variables
    --------------------------
    If the input data are all stored as separate variables, they can be
    provided with the appropriate kwargs from below.  In this case, each
    variable must be one of a scalar, a `list` or a NumPy `array`, and the
    shapes of these arrays must be mutually broadcastable.  For example:

      >>> import PyCO2SYS as pyco2, numpy as np
      >>> dic = [2100, 2150]
      >>> temperature = np.array([15, 13.5])
      >>> co2s = pyco2.sys(dic=dic, pH=8.1, temperature=temperature)

    Data variables in a container
    -----------------------------
    If the input data are gathered in a `dict`, pandas `DataFrame` or xarray
    `Dataset`, then this can be provided with the `data` kwarg.  The keys in
    the container dataset must be strings and they should correspond to the
    kwargs below.  If they do not correspond, then the kwarg can be passed with
    the corresponding key instead.  For example:

      >>> import PyCO2SYS as pyco2
      >>> data = {"dic": [2100, 2150], "pH_lab": 8.1, "t_lab": 25}
      >>> co2s = pyco2.sys(data=data, pH="pH_lab", temperature="t_lab")

    Core marine carbonate system parameters
    ---------------------------------------
    A maximum of two core parameters may be provided, and not all combinations
    are valid.  In some cases, a subset of calculations can still be carried
    out with only one parameter.  It is also possible to pass none of these
    parameters and still calculate e.g. equilibrium constants.

               Parameter | Description (unit)
    -------------------: | :---------------------------------------------------
              alkalinity | Total alkalinity (µmol/kg)
                     dic | Dissolved inorganic carbon (µmol/kg)
                      pH | Seawater pH on the scale given by `opt_pH_scale`
                    pCO2 | Seawater partial pressure of CO2 (µatm)
                    fCO2 | Seawater fugacity of CO2 (µatm)
                     CO2 | Aqueous CO2 content (µmol/kg)
                    HCO3 | Bicarbonate ion content (µmol/kg)
                     CO3 | Carbonate ion content (µmol/kg)
                    xCO2 | Seawater dry air mole fraction of CO2 (ppm)
      saturation_calcite | Saturation state with respect to calcite
    saturation_aragonite | Saturation state with respect to aragonite

    Hydrographic conditions
    -----------------------
    Middle column gives default values if not provided.

              Parameter | Df. | Description (unit)
    ------------------: | --: | :----------------------------------------------
               salinity |  35 | Practical salinity
            temperature |  25 | Temperature (°C)
               pressure |   0 | Hydrostatic pressure (dbar)
    pressure_atmosphere |   1 | Atmospheric pressure (atm)

    Nutrients and other solutes
    ---------------------------
    Middle column gives default values; S = calculated from salinity.

          Parameter | Df. | Description (unit)
    --------------: | :-: | :----------------------------------------------
     total_silicate |  0  | Total dissolved silicate (µmol/kg)
    total_phosphate |  0  | Total dissolved phosphate (µmol/kg)
      total_ammonia |  0  | Total dissolved ammonia (µmol/kg)
      total_sulfide |  0  | Total dissolved sulfide (µmol/kg)
      total_nitrite |  0  | Total dissolved nitrite (µmol/kg)
       total_borate |  S  | Total dissolved borate (µmol/kg)
     total_fluoride |  S  | Total dissolved fluoride (µmol/kg)
      total_sulfate |  S  | Total dissolved sulfate (µmol/kg)
                 Ca |  S  | Dissolved calcium (µmol/kg)

    SETTINGS PROVIDED AS KWARGS
    ===========================
    Options such as which pH scale to use and the choice of parameterisation
    for various e.g. equilibrium constants can also be altered with kwargs.
    These kwargs must all be scalar integers.

    Citations use a code with the initials of the first few authors' surnames
    plus the final two digits of the year of publication.  Refer to the online
    documentation for full references.

    pH scale
    --------
    opt_pH_scale: pH scale of `pH` (if provided); also used for calculating
    equilibrium constants.
        1: total [DEFAULT].
        2: seawater.
        3: free.
        4: NBS.

    Carbonic acid dissociation
    --------------------------
    opt_k_carbonic: parameterisation for carbonic acid dissociation (K1 and
    K2).
         1: RRV93.
         2: GP89.
         3: H73a and H73b refit by DM87.
         4: MCHP73 refit by DM87.
         5: H73a, H73b and MCHP73 refit by DM87.
         6: MCHP73 ("GEOSECS").
         7: MCHP73 ("GEOSECS-Peng").
         8: M79 (freshwater).
         9: CW98.
        10: LDK00 [DEFAULT].
        11: MM02.
        12: MPL02.
        13: MGH06.
        14: M10.
        15: WMW14.
        16: SLH20.
        17: SB21.
        18: PLR18.
        19: MMB25.
    opt_factor_k_H2CO3: pressure correction for the first carbonic acid
    dissociation constant (K1).
        1: M95 [DEFAULT].
        2: EG70 (GEOSECS).
        3: M83 (freshwater).
    opt_factor_k_HCO3: pressure correction for the second carbonic acid
    dissociation constant (K2).
        1: M95 [DEFAULT].
        2: EG70 (GEOSECS).
        3: M83 (freshwater).

    Other equilibrium constants
    ---------------------------
    opt_k_BOH3: parameterisation for boric acid equilibrium.
        1: D90b [DEFAULT].
        2: LTB69 (GEOSECS).
    opt_k_H2O: parameterisation for water dissociation.
        1: M95 [DEFAULT].
        2: M79 (GEOSECS).
        3: HO58 refit by M79 (freshwater).
    opt_k_HF: parameterisation for hydrogen fluoride dissociation.
        1: DR79 [DEFAULT].
        2: PF87.
    opt_k_HNO2: parameterisation for nitrous acid dissociation.
        1: BBWB24 for seawater [DEFAULT].
        2: BBWB24 for freshwater.
    opt_k_HSO4: parameterisation for bisulfate dissociation.
        1: D90a [DEFAULT].
        2: KRCB77.
        3: WM13/WMW14.
    opt_k_NH3: parameterisation for ammonium dissociation.
        1: CW95 [DEFAULT].
        2: YM95.
    opt_k_phosphate: parameterisation for the phosphoric acid dissocation
    constants.
        1: YM95 [DEFAULT].
        2: KP67 (GEOSECS).
    opt_k_Si: parameterisation for bisulfate dissociation.
        1: YM95 [DEFAULT].
        2: SMB64 (GEOSECS).
    opt_k_aragonite: parameterisation for aragonite solubility product.
        1: M83 [DEFAULT].
        2: ICHP73 (GEOSECS).
    opt_k_calcite: parameterisation for calcite solubility product.
        1: M83 [DEFAULT].
        2: I75 (GEOSECS).

    Other dissociation constant pressure corrections
    ------------------------------------------------
    opt_factor_k_BOH3: pressure correction for boric acid equilibrium.
        1: M79 [DEFAULT].
        2: EG70 (GEOSECS).
    opt_factor_k_H2O: pressure correction for water dissociation.
        1: M95 [DEFAULT].
        2: M83 (freshwater).

    Total salt contents
    -------------------
    These settings are ignored if their values are provided directly as kwargs.

    opt_total_borate: for calculating `total_borate` from `salinity`.
        1: U74 [DEFAULT].
        2: LKB10.
        3: KSK18 (Baltic Sea).
    opt_Ca: for calculating `Ca` from `salinity`.
        1: RT67 [DEFAULT].
        2: C65 (GEOSECS).

    Other settings
    --------------
    opt_gas_constant: the universal gas constant (R)
        1: DOEv2 (consistent with pre-July 2020 CO2SYS software).
        2: DOEv3.
        3: 2018 CODATA [DEFAULT].
    opt_fugacity_factor: how to convert between partial pressure and fugacity.
        1: with a fugacity factor [DEFAULT].
        2: assuming they are equal (GEOSECS).
    opt_HCO3_root: with known `dic` and `HCO3`, which root to solve for.
        1: the lower pH root.
        2: the higher pH root [DEFAULT].
    opt_fCO2_temperature: sensitivity of fCO2 to temperature.
        1: H24 parameterisation [DEFAULT].
        2: TOG93 linear fit.
        3: TOG93 quadratic fit.

    Equilibrium constants
    ---------------------
    Any equilibrium constant calculated within PyCO2SYS can be provided
    directly as an input instead.  The kwarg should use the same key as would
    be used to solve for this parameter.  See the docs for `CO2System.solve`
    for more detail.
    """
    # Check for double precision
    if np.array(1.0).dtype == np.dtype("float32"):
        warn(
            "JAX does not appear to be using double precision - "
            + "set the environment variable `JAX_ENABLE_X64=True`"
        )
    # Merge data with kwargs
    pd_index = None
    xr_dims = None
    xr_shape = None
    data_is_dict = isinstance(data, dict)
    keys_ignored = []
    kwargs_data = {}
    if data is not None:
        # First, check for string kwargs, which indicate keys in data that need
        # renaming
        renamer_user = {}
        for k, v in kwargs.items():
            if isinstance(v, str):
                if v in renamer_user:
                    # Can't repeat keys e.g. `data=df, dic="var", pH="var"`
                    raise Exception(
                        f'"{v}" cannot be used for {k} because'
                        + f" it is already being used for {renamer_user[v]}"
                    )
                else:
                    renamer_user[v] = shortcuts[k.lower()]
        # Next, go through keys of data and get shortcuts or renames for them
        renamer_data = {}
        for k in data:
            if k in renamer_user:
                renamer_data[k] = renamer_user[k]
            elif k in shortcuts:
                renamer_data[k] = shortcuts[k.lower()]
            else:
                keys_ignored.append(k)
        # Check for duplicates
        renamer_values = []
        for v in renamer_data.values():
            if v in renamer_values:
                raise SyntaxError(
                    f"`data` contains multiple keys corresponding to `{v}`, "
                    + "possibly under different shortcuts"
                )
            else:
                renamer_values.append(v)
        # Rename keys if data is a dict
        if data_is_dict:
            for k, v in renamer_data.items():
                kwargs_data[v] = data[k]
        # If `data` isn't a dict, it might be a pandas df
        else:
            data_is_pandas = False
            try:
                import pandas as pd

                data_is_pandas = isinstance(data, pd.DataFrame)
                # If `data` is a pandas df, we need to rename string keys,
                # convert Series to numpy arrays, and save the df index
                if data_is_pandas:
                    pd_index = data.index.copy()
                    for c in data.columns:
                        if c in renamer_data:
                            kwargs_data[renamer_data[c]] = data[c].to_numpy()
            except ImportError:
                warn("pandas could not be imported - ignoring `data`.")
            data_is_xarray = False
            if not data_is_pandas:
                try:
                    import xarray as xr

                    data_is_xarray = isinstance(data, xr.Dataset)
                    # If `data` is an xarray ds, we need to rename string keys,
                    # convert DataArrays to numpy arrays, and store all the
                    # dimensions
                    if data_is_xarray:
                        xr_dims = list(data.sizes.keys())
                        xr_shape = list(data.sizes.values())
                        for k, v in data.items():
                            if k in renamer_data:
                                kwargs_data[renamer_data[k]] = da_to_array(
                                    v, xr_dims
                                )
                except ImportError:
                    warn("xarray could not be imported - ignoring `data`.")
                if not data_is_xarray:
                    # If we reach this point, `data` is neither dict nor
                    # pandas df nor xarray ds, so it's ignored
                    warn("Type of `data` not recognised - it will be ignored.")
                    keys_ignored.append("data")
    # Check there aren't any duplicate kwargs with different aliases, and drop
    # any kwargs that are strings (used to identify `data` columns)
    kwargs_nodups = {}
    for k, v in kwargs.items():
        try:
            skl = shortcuts[k.lower()]
            if skl in kwargs_nodups:
                raise SyntaxError(
                    f"Repeated kwarg, possibly under a different shortcut: {k}"
                )
            elif not isinstance(v, str):
                kwargs_nodups[skl] = v
                if skl in kwargs_data:
                    warn(
                        f"{skl} found in both `data` and `kwargs`, possibly under "
                        + "different shortcuts - using the `kwargs` value"
                    )
        except KeyError:
            keys_ignored.append(k)
    # Merge data and user kwargs
    kwargs_data.update(kwargs_nodups)
    # Parse kwargs
    for k, v in kwargs_data.copy().items():
        # Convert lists to numpy arrays
        if isinstance(kwargs_data[k], list):
            kwargs_data[k] = np.array(kwargs_data[k])
        # Convert None to np.nan
        if kwargs_data[k] is None:
            kwargs_data[k] = np.nan
        # If opts are scalar, only take first value
        if k in opts_default:
            if np.isscalar(kwargs_data[k]):
                try:
                    kwargs_data[k] = kwargs_data[k].item()
                except (AttributeError, ValueError):
                    pass
            else:
                kwargs_data[k] = np.ravel(np.array(kwargs_data[k]))[0].item()
                warn(
                    f"`{k}` is not scalar, so only the first value will be used."
                )
            if isinstance(kwargs_data[k], float):
                kwargs_data[k] = int(kwargs_data[k])
        # For non-opts
        else:
            # Downgrade pd.Series and xr.DataArray to numpy arrays without
            # importing pandas or xarray---but the user should avoid doing this
            # because it doesn't take care of indices properly
            try:
                _ = kwargs_data[k].values
                raise Exception(
                    f"`{k}` provided as a `pd.Series` or `xr.DataArray`, "
                    + "which is not allowed."
                )
            except AttributeError:
                pass
            # Convert ints to floats
            if isinstance(kwargs_data[k], int):
                kwargs_data[k] = float(kwargs_data[k])
            elif hasattr(kwargs_data[k], "dtype"):
                try:
                    kwargs_data[k] = kwargs_data[k].astype(float)
                except ValueError:
                    pass
            # Turn not-allowed negatives to NaN
            if (
                k not in ["alkalinity", "pH", "temperature"]
                and not k.startswith("pH_")
                and not k.startswith("pk_")
                and not k.startswith("pkt_")
            ):
                kwargs_data[k] = np.where(
                    kwargs_data[k] < 0, np.nan, kwargs_data[k]
                )
    return CO2System(
        pd_index=pd_index,
        xr_dims=xr_dims,
        xr_shape=xr_shape,
        ignored=keys_ignored,
        **kwargs_data,
    )
