!!! example "Try out PyCO2SYS v2!"
    PyCO2SYS v2 is currently in beta testing.  The v2 docs and installation instructions are at [mvdh.xyz/PyCO2SYS](https://mvdh.xyz/PyCO2SYS/).  For now, installing via `pip` or `conda` still gets you v1.8.3, matching the docs here ([PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/)).

    Please try it out and report any issues you encounter via [the GitHub repo](https://github.com/mvdh7/PyCO2SYS/issues)!
    
    The API has been kept as similar as possible to v1, but there are some breaking changes.  The v2 code runs significantly faster with lower memory overhead and is designed to be more intuitive to use.

# Uncertainty propagation

PyCO2SYS provides tools to propagate uncertainties in all arguments through to all results of its marine carbonate system calculations.

!!! question "Evaluating the derivatives"

    All derivatives needed for uncertainty propagation are calculated using *forward finite-differences*.  Explicity, the derivative of result $r$ with respect to argument $a$ is estimated with:

    $\frac{\partial r(a)}{\partial a} \approx \frac{r(a + \Delta a) - r(a)}{\Delta a}$

    As the different arguments span many orders of magnitude, PyCO2SYS uses a different $\Delta a$ value for each argument.

## Independent uncertainties

If the uncertainty in each [argument](../co2sys_nd/#arguments) is independent – i.e. there is no covariance between uncertainties in different parameters – then you can use `PyCO2SYS.uncertainty.propagate` to propagate the parameter uncertainties through into any [result](../co2sys_nd/#results).

### Syntax

Uncertainty propagation can be performed by adding some extra keyword arguments to [`pyco2.sys`](../co2sys_nd).

```python
import PyCO2SYS as pyco2

# Example arguments: known DIC and pH, at 10 degC
kwargs = {
    "par1": 2150.0,
    "par2": 8.1,
    "par1_type": 2,
    "par2_type": 3,
    "temperature": 10,
}

# Normal way to run PyCO2SYS, without uncertainties:
results = pyco2.sys(**kwargs)

# Alternatively, calculate uncertainties at the same time:
results = pyco2.sys(
    **kwargs,
    uncertainty_into=["alkalinity", "k_carbonic_1"],
    uncertainty_from={
        "par1": 2.0,
        "temperature": 0.05,
    }
)
```

### Arguments

!!! inputs "`pyco2.sys` uncertainty arguments"

    #### The same `kwargs` as for `pyco2.sys`

    Provide all the necessary keyword arguments for [`pyco2.sys`](../co2sys_nd).

    #### Uncertainties

      * `uncertainty_into`: a list of strings of the [results keys](../co2sys_nd/#results) to propagate uncertainties into.

      * `uncertainty_from`: a dict of the uncertainties in the arguments to propagate through `pyco2.sys`.

    The keys of `uncertainty_from` can include any [arguments of `pyco2.sys`](../co2sys_nd/#arguments) that can have an uncertainty.  The key for each uncertainty in `uncertainty_from` should be the same as the corresponding key in the main [`pyco2.sys` results dict](../co2sys_nd/#results).

    If you want to provide a fractional value for any uncertainty, append `"__f"` to the end of its key in `uncertainty_from`.
    
    For the equilibrium constants, if you wish to propagate an uncertainty in terms of a p<i>K</i> value rather than *K*, prefix the corresponding key in `uncertainty_from` with a `"p"` (e.g. use `"pk_carbonic_1"` instead of `"k_carbonic_1"`).  Uncertainties in equilibrium constants under input and output conditions are treated independently.  To use the same (covarying) uncertainty for both, append `"_both"` to the input condition key (e.g. `"k_carbonic_1_both"`).  In this case, you must have provided a value for either `temperature_out` or `pressure_out`.
    
    The "standard" uncertainties in the equilbrium constants and total borate used by CO2SYS for MATLAB following [OEDG18](../refs/#o) are available as a dict in the correct format for `uncertainty_from` at `pyco2.uncertainty_OEDG18`.

    The values of `uncertainty_from` are the uncertainties in each input parameter as a standard deviation.  You can provide a single value if all uncertainties are the same for a parameter, or an array the same size as the parameter if they are different.  Any parameters not included are assumed to have zero uncertainty.

### Results

!!! outputs "`pyco2.sys` uncertainty results"

    If `uncertainty_into` and `uncertainty_from` are provided, then the uncertainty results are added as additional entries to the standard `results` dict.  This includes the total uncertainty as well as individual components.
    
    * For each result in `uncertainty_into`, there is a new key `"u_<into>"` in the `results` dict, containing the total uncertainty in the result from all arguments combined.

    * For each result in `uncertainty_into` and argument in `uncertainty_from`, there is a new key `"u_<into>__<from>"` in the `results` dict, containing the uncertainty in the result from only the specified argument.

    The combined uncertainties are the Pythagorean sum of all the components.  This calculation assumes that all argument uncertainties are independent from each other and that they are provided in terms of single standard deviations.

## Uncertainties with covariances

PyCO2SYS does not currently have a generalised function for the complete process of propagating uncertainties that co-vary.  However, it does allow you calculate the forward finite-difference derivative of any result with respect to any argument.  The syntax is similar as described above for uncertainties:

```python
import PyCO2SYS as pyco2

# Example arguments: known DIC and pH, at 10 degC
kwargs = {
    "par1": 2150.0,
    "par2": 8.1,
    "par1_type": 2,
    "par2_type": 3,
    "temperature": 10,
}

# Normal way to run PyCO2SYS, without derivatives:
results = pyco2.sys(**kwargs)

# Alternatively, calculate derivatives at the same time:
results = pyco2.sys(
    **kwargs,
    grads_of=["alkalinity", "k_carbonic_1"],
    grads_wrt=["par1", "temperature"],
)
```

 In general, this works the same as the uncertainty propagation approach described in the previous section.  The main differences are:

  * `grads_of` is equivalent to `uncertainty_into`.
  * `grads_wrt` (w.r.t. = with respect to) is equivalent to `uncertainty_from`, but values are not required, so it can be a list.  A dict is also fine; its values are ignored.
  * The `"__f"` key extension cannot be used in `grads_wrt`.
  * For each result in `grads_of` and argument in `grads_wrt`, there is a new key `"d_<into>__d_<from>"` in the `results` dict, containing the derivative of the result with respect to the argument.
