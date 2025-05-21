# Switching from v1 to v2

This explanation is designed as a detailed overview of differences for those who are already familiar with using PyCO2SYS with the `pyco2.sys` syntax from v1.  New or casual users may find the [general instructions for v2](detail.md) more helpful.

## Solving the carbonate system

On the surface, v2 looks similar to v1.  For example:

=== "v2.0"

    ```python
    import PyCO2SYS as pyco2

    # Set up the CO2System
    co2s = pyco2.sys(
        alkalinity=2250,
        pH=8.1,
        temperature=12.5,
        salinity=32.4,
        opt_k_carbonic=10,
    )
    # Solve for and retrieve a calculated parameter
    dic = co2s["dic"]
    ```

=== "v1.8"

    ```python
    import PyCO2SYS as pyco2

    # Solve the carbonate system
    co2s = pyco2.sys(
        par1=2250,
        par2=8.1,
        par1_type=1,
        par2_type=2,
        temperature=12.5,
        salinity=32.4,
        opt_k_carbonic=10,
    )
    # Retrieve a calculated parameter
    dic = co2s["dic"]
    ```

### One set of settings per call

The most obvious change above is that the known marine carbonate system parameters `alkalinity` and `pH` are provided directly as kwargs, instead of using the old approach with `par1`, `par2`, `par1_type` and `par2_type`.

This does mean that it's no longer possible to have multiple different parameter types within a single call to `pyco2.sys`.  Similarly, all settings parameters (anything beginning with `opt_`) can no longer be provided as arrays – each call to `pyco2.sys` can only have one combination of settings.

However, as before, all other parameters (e.g., `alkalinity`, `pH`, `temperature` and `salinity`) can be scalars or NumPy arrays of any shape, as long as they can all be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) together.

Also as before, it's possible to provide no carbonate system parameters, and just calculate equilibrium constants and total salt contents, or one carbonate system parameter, and do whatever calculations are possible with it alone.

### When the system is solved

In v1.8, the result `co2s` was a standard Python dict containing all possible parameters, which were computed at the moment that `pyco2.sys` was called.  However, in v2.0, `co2s` is a `CO2System` object, and it contains only the values of kwargs provided as arguments.  Other parameters are computed only when the user attempts to retrieve them (e.g., with `co2s["dic"]` above).

When a parameter is retrieved, the minimum set of intermediate parameters required to compute it are also calculated and stored in `co2s`.  These stored values will be used to compute any subsequently requested parameters – they will not be recomputed each time.

### Retrieving parameters

As mentioned above, parameters can be retrieved (and, if necessary, calculated) as if `co2s` were a standard Python dict.  However, there is also more flexibility:

#### Retrieving multiple parameters

A list of parameters can be provided with the same syntax as above:

=== "v2.0"

    ```python
    # Solve for / return DIC and fCO2
    params = co2s[["dic", "fCO2"]]

    # Access the results
    dic = params["dic"]
    fCO2 = params["fCO2"]
    ```

The result `params` is a standard dict containing the requested parameters.

#### The `solve` method

Parameters can be solved for using `solve`:

=== "v2.0"

    ```python
    # Solve for a calculated parameter
    co2s.solve("dic", store_steps=1)
    ```

This gives the added flexibility of the `store_steps` kwarg:
  
  * If `0`, then only the requested parameter is stored.
  * If `1` (default), then the standard set of intermediate parameters is stored in the `co2s`.
  * If `2`, then all possible intermediate parameters are stored.

#### Dot notation

If a parameter has been calculated, either directly or as an intermediate for another parameter, and its value is stored in the `co2s`, it can also be accessed with dot notation:

=== "v2.0"

    ```python
    # Access DIC, if it has been previously calculated
    dic = co2s.dic
    ```

This will throw an error if the requested parameter is not already available.

## Input and output conditions

In v1, a second set of temperature and/or pressure conditions could be specified as 'output' conditions with the suffix `_out` for their arguments and results.  In v2, each `CO2System` can only have one set of temperature and pressure conditions.  To adjust to a different set of conditions, use the `adjust` method:

=== "v2.0"

    ```python
    # Set up the CO2System
    co2s = pyco2.sys(
        alkalinity=2250,
        pH=8.1,
        temperature=12.5,
        salinity=32.4,
        opt_k_carbonic=10,
    )

    # Convert to different conditions
    co2s_adj = co2s.adjust(
        temperature=25,
        pressure=1000,
    )

    # Get fCO2 at the adjusted conditions
    fCO2_adj = co2s_adj["fCO2"]
    ```

=== "v1.8"

    ```python
    # Solve the carbonate system
    co2s = pyco2.sys(
        par1=2250,
        par2=8.1,
        par1_type=1,
        par2_type=2,
        temperature=12.5,
        temperature_out=25,
        pressure_out=1000,
        salinity=32.4,
        opt_k_carbonic=10,
    )

    # Get fCO2 at the adjusted conditions
    fCO2_adj = co2s["fCO2_out"]
    ```

The result `co2s_adj` is a separate `CO2System` at the requested temperature and pressure.  If the original `co2s` had two known parameters, then both are used to make the adjustment (via DIC and alkalinity).  Temperature can also be adjusted with only one known parameter, if its one of pCO<sub>2</sub>, fCO<sub>2</sub>, [CO<sub>2</sub>(aq)] or *x*CO<sub>2</sub>.  The kwargs `method_fCO2`, `opt_which_fCO2_insitu` and `bh_upsilon` allow for finer control of the one-parameter adjustment (see the [v2 general instructions](detail.md) for details).

## Uncertainty propagation

There are three main differences regarding uncertainty propagation:

  1.  Uncertainty propagation is carried out by using the `set_uncertainty` and `propagate` methods in v2.0, instead of by including `uncertainty_into` and `uncertainty_from` kwargs in the main `pyco2.sys` call. 

  2.  Uncertainty results are stored in nested dicts in `co2s.uncertainty` in v2.0, instead of as additional key-value pairs in `co2s` as in v1.8.

  3.  The derivatives used to propagate uncertainties are calculated with automatic differentiation in v2.0, but with forward finite differences in v1.8.

The arguments provided to the `propagate` method in v2.0 are exactly the same `uncertainty_into` and `uncertainty_from` as were provided directly to `pyco2.sys` in v1.8.

An example:

=== "v2.0"

    ```python
    # Set up the CO2System
    co2s = pyco2.sys(
        alkalinity=2250,
        pH=8.1,
        temperature=12.5,
        salinity=32.4,
        opt_k_carbonic=10,
    )

    # Set and propagate uncertainties
    co2s.set_uncertainty(alkalinity=2.1, pH=0.02)
    co2s.propagate("dic")

    # Retrieve uncertainties and their components
    dic_uncertainty = co2s.uncertainty["dic"]
    dic_uncertainty_from_pH = co2s.uncertainty.parts["dic"]["pH"]
    ```

=== "v1.8"

    ```python
    # Solve the carbonate system and propagate uncertainties
    co2s = pyco2.sys(
        par1=2250,
        par2=8.1,
        par1_type=1,
        par2_type=2,
        temperature=12.5,
        salinity=32.4,
        opt_k_carbonic=10,
        uncertainty_into="dic",
        uncertainty_from={"par1": 2.1, "par2": 0.02},
    )

    # Retrieve uncertainties and their components
    dic_uncertainty = co2s["u_dic"]
    dic_uncertainty_from_pH = co2s["u_dic__pH"]
    ```

## Settings

Before v2, changing `opt_k_carbonic` to a different set of carbonic acid dissociation constants could also cause other parameterisations to be switched behind the scenes (for e.g. the borate equilibrium constant and some pressure correction factors).  This behaviour was inherited from CO2SYS-MATLAB, but it has been eliminated in PyCO2SYS v2.  Instead, every parameterisation that has multiple options is controlled independently with its own setting.

This affects only `opt_k_carbonic` values `6`, `7`, and `8`, i.e., the GEOSECS and freshwater cases.  All other `opt_k_carbonic` options used the set of parameterisations that are now the defaults in v2.

<!-- ## Summary of differences

| v1 | v2 |
| - | - |
| All possible parameters are always calculated. | Only the requested parameters and the required intermediates are calculated. |
| Settings (kwargs beginning `opt_`) can be multidimensional. | Settings (`opts`) must all be scalars. |
| Known marine carbonate system parameters are provided as `par1` and `par2`, with their types given by `par1_type` and `par2_type`. | Known marine carbonate system core parameters are provided as `alkalinity`, `dic`, `pH`, etc. |
| Multiple different combinations of `par1_type` and `par2_type` can be provided. | Each `CO2System` instance can only contain one particular combination of core parameters. | -->
