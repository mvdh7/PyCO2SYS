# Uncertainty propagation

PyCO2SYS provides tools to propagate uncertainties both in input parameters and in variables evaluated internally through its marine carbonate system calculations.

!!! question "Evaluating the derivatives"

    All derivatives needed for uncertainty propagation are calculated using *forward finite-differences*.  Explicity, the derivative of output variable $f$ with respect to input $x$ is estimated with:

    $f'(x) \approx \frac{f(x + \Delta x) - f(x)}{\Delta x}$

    As the input variables span many orders of magnitude, PyCO2SYS by default uses $\Delta x = 10^{-6} \cdot \mathrm{median}(x)$, which is different for each input variable.  However, this behaviour can be adjusted (see [Settings](#settings) below).

## Independent uncertainties

If the uncertainty in each [input parameter](../co2sys/#inputs) is independent – there is no covariance between uncertainties in different parameters – then you can use `PyCO2SYS.uncertainty.propagate` to propagate the parameter uncertainties through into any [output variable](../co2sys/#outputs).

### Syntax

You must run either `PyCO2SYS.CO2SYS` (see [MATLAB-style CO2SYS](../co2sys)) or `PyCO2SYS.CO2SYS_nd` (see [New-style CO2SYS_nd](../co2sys_nd)) to generate the `co2dict` that is used as an input here.  Instructions for both interfaces are provided here.

    :::python
    import PyCO2SYS as pyco2

    # CO2SYS_nd style - get co2dict
    co2dict = pyco2.CO2SYS_nd(par1, par2, par1_type, par2_type, **kwargs)
    # CO2SYS_nd style - propagate uncertainties
    uncertainties, components = pyco2.uncertainty.propagate_nd(
        co2dict, uncertainties_into, uncertainties_from,
        dx=1e-6, dx_scaling="median", dx_func=None, **kwargs)

    # MATLAB-CO2SYS style - get co2dict
    co2dict = pyco2.CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
        PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
        NH3=0.0, H2S=0.0, KFCONSTANT=1, buffers_mode="auto",
        totals=None, equilibria_in=None, equilibria_out=None)
    # MATLAB-CO2SYS style - propagate uncertainties
    uncertainties, components = pyco2.uncertainty.propagate(
        co2dict, uncertainties_into, uncertainties_from,
        totals=None, equilibria_in=None, equilibria_out=None,
        dx=1e-6, dx_scaling="median", dx_func=None)

There are some examples of using the uncertainty propagation function in the [PyCO2SYS-examples](https://github.com/mvdh7/PyCO2SYS-examples/) repo on GitHub.

### Inputs

!!! info "`PyCO2SYS.uncertainty.propagate[_nd]` inputs"

    #### `CO2SYS` or `CO2SYS_nd` output dict

    The first input, `co2dict`, you must first generate with either `PyCO2SYS.CO2SYS` (see [MATLAB-style CO2SYS](../co2sys)) or `PyCO2SYS.CO2SYS_nd` (see [New-style CO2SYS_nd](../co2sys_nd)).

    #### Uncertainties

      * `uncertainties_into`: a list of strings of the **output variable names** to propagate the uncertainties into.

    The `uncertainties_into` list can contain any of the [outputs of `PyCO2SYS.CO2SYS`](../co2sys/#outputs) that can have an uncertainty (i.e. all aside from those relating to input settings).  You can use `uncertainties_into="all"` to get propagate uncertainties through into every output variable at once.

      * `uncertainties_from`: a dict of the **input parameter uncertainties** to propagate through `PyCO2SYS.CO2SYS`.

    The keys of `uncertainties_from` may come from the [inputs of `PyCO2SYS.CO2SYS`](../co2sys/#inputs) or [inputs of `PyCO2SYS.CO2SYS_nd`](../co2sys_nd/#inputs) that can have an uncertainty.  Additionally, the uncertainty in any parameter that can be provided as an [internal override](../co2sys/#internal-overrides) can be included within the `uncertainties_from` dict.  The keys for these parameters should be the same as the corresponding key in the main [`PyCO2SYS.CO2SYS` output dict](../co2sys/#outputs) or [`PyCO2SYS.CO2SYS_nd` output dict](../co2sys_nd/#outputs).
    
    For the equilibrium constants, if you need to propagate an uncertainty in terms of a p<i>K</i> value rather than *K*, simply prefix the corresponding key in `uncertainties_from` with a `"p"` (e.g. use `"pK1input"` instead of `"K1input"` in MATLAB-style, or equivalently `"pk_carbonic_1"` instead of `"k_carbonic_1"` in `CO2SYS_nd`-style).
    
    The "standard" uncertainties in the equilbrium constants used by CO2SYS for MATLAB following [OEDG18](../refs/#o) are available in the correct format for `uncertainties_in` at `PyCO2SYS.uncertainties.pKs_OEDG18`.

    The values of `uncertainties_from` are the uncertainties in each input parameter as a standard deviation.  You can provide a single value if all uncertainties are the same for a parameter, or an array the same size as the parameter if they are different.  Any parameters not included are assumed to have zero uncertainty.


    #### Internal overrides

    If you provided values for any of the optional [internal overrides](../co2sys/#internal-overrides) (`totals`, `equilibria_in` or `equilibria_out`) when running `PyCO2SYS.CO2SYS`, then you must provide exactly the same inputs again here.

    Similarly, any and all `kwargs` provided to `PyCO2SYS.CO2SYS_nd` must also be provided to the `propagate_nd` function.

    You do not need to provide an internal override value in order to propagate uncertainty in that variable.

    #### Settings

    These are all optional.

      * `dx`: the spacing multiplier for forward finite-difference derivatives (default = 10<sup>−6</sup>).
      * `dx_scaling`: determines the method used to scale `dx`.  Options are, for each input variable `var`:
        - `"median"` (default): `dxs[var] = dx * median(var)`.
        - `"none"`: `dxs[var] = dx`.
        - `"custom"`: `dxs[var] = dx_func(var)`, where:
      * `dx_func`: user-provided function to calculate `dx[var]` from `var` values.  Only used if `dx_scaling="custom"`.

### Outputs

!!! abstract "`PyCO2SYS.uncertainty.propagate[_nd]` outputs"

    * `uncertainties`: the **total uncertainty** in each output variable in `uncertainties_into`.
    * `components`: the **separate contribution of each input parameter** in `uncertainties_from` to the total uncertainty for each output variable in `uncertainties_into`.

    Both outputs are dicts with keys the same as `uncertainties_into`.

    Each entry in `components` is itself a dict with keys the same as `uncertainties_from`, containing the uncertainty in the output variable from each input parameter separately.  For example, the uncertainty in total borate arising from the uncertainty in input salinity could be accesed with `components["TB"]["SAL"]` in MATLAB style, or `components["total_borate"]["salinity"]` in `CO2SYS_nd` style.

    Each entry in `uncertainties` is the Pythagorean sum of all the different uncertainty components for each variable.  This calculation assumes that all uncertainties are independent from each other and that they are provided in terms of single standard deviations.

## Uncertainties with covariances

PyCO2SYS does not currently have a generalised function for the complete process of propagating uncertainties that co-vary.  However, it does allow you calculate the forward finite-difference derivative of any output with respect to any input, which you can use to propagate uncertainties in any specific case:

    :::python
    # CO2SYS_nd-style
    co2derivs, dxs = pyco2.uncertainty.forward_nd(co2dict, grads_of, grads_wrt,
        dx=1e-6, dx_scaling="median", dx_func=None, **kwargs)

    # MATLAB-style
    co2derivs, dxs = pyco2.uncertainty.forward(co2dict, grads_of, grads_wrt,
        totals=None, equilibria_in=None, equilibria_out=None,
        dx=1e-6, dx_scaling="median", dx_func=None)

The inputs to `PyCO2SYS.uncertainty.forward` are the same as [described above](#inputs) for `PyCO2SYS.uncertainty.propagate` with the following notes:

  * `grads_of` is the same as `uncertainties_into`.
  * `grads_wrt` can be the same as `uncertainties_from`, in which case the values are ignored, or more simply just `list(uncertainties_from.keys())`.

The output `co2derivs` is a dict with the same structure as the [`components` output](#outputs) of `PyCO2SYS.uncertainty.propagate`, containing the derivatives of each output variable in `grads_of` with respect to each input parameter in `grads_wrt`.

`dxs` is a dict containing the actual `dx` values used for each variable after scaling.
