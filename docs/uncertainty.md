# Uncertainty propagation

Independent uncertainties are defined and propagated with the `set_uncertainty` and `propagate` methods.

All derivatives needed for uncertainty propagation are calculated with automatic differentiation.

## Define independent uncertainties

If the uncertainty in each [argument](detail.md/#keyword-arguments) is independent – i.e. there is no covariance between the uncertainties in different parameters – then the `set_uncertainty` and `propagate` methods can be used to propagate the parameter uncertainties through into any [result](detail.md/#results):

```python
import PyCO2SYS as pyco2

# Set up the CO2System
co2s = pyco2.sys(dic=2100, alkalinity=2250, temperature=20)

# Define uncertainties in the known parameters
co2s.set_uncertainty(dic=2, alkalinity=1)
```

!!! inputs "`set_uncertainty` kwargs"

    `set_uncertainty` can take the same kwargs as `pyco2.sys`, excepting the settings (keys beginning with `"opt_"`).  It is not necessary to provide an uncertainty for every parameter - those that are not specified are assumed to have zero uncertainty.

    The values provided should be the 1<i>σ</i> uncertainty in each parameter.  They can be single scalar values or arrays matching the shape of the correpsonding parameter.

    To provide a fractional value for any uncertainty, append `"__f"` to the end of its key in `uncertainty_from`.

The "standard" uncertainties in the equilbrium constants and total borate used by CO2SYS for MATLAB following [OEDG18](refs.md/#o) are available as a dict in the correct format for `set_uncertainty` at `pyco2.uncertainty_OEDG18`:

```python
# Also include equilibrium constant uncertainties
co2s.set_uncertainty(**pyco2.uncertainty_OEDG18)
```

If `set_uncertainty` is run multiple times on the same `CO2System`, each successive call adds to the existing set of uncertainties, overwriting where an uncertainty for that parameter was already declared.

!!! warn "`set_uncertainty` after `propagate`"

    If `set_uncertainty` is run after running `propagate` on a system, then `propagate` will automatically be run again with the new set uncertainties, so that the values in `co2s.uncertainty` are all correct for the current set of assigned uncertainties.

## Propagate independent uncertainties

```python
# Propagate uncertainties set with set_uncertainty
co2s.propagate(["pH", "fCO2"])

# Access uncertainty results
uncert_fCO2 = co2s.uncertainty["fCO2"]
uncert_pH_due_to_dic = co2s.uncertainty["pH"]["dic"]
```

The total uncertainties are the Pythagorean sum of all the components.  This calculation assumes that all argument uncertainties are independent from each other and that they are provided in terms of single standard deviations.

!!! inputs "`propagate` arguments"

    * `uncertainty_into`: a list of the parameter keys that uncertainties are to be propagated into.

    If `propagate` is run with no arguments, then uncertainties will be propagated into all results that have been currently solved for.

!!! outputs "`propagate` results"

    The uncertainty results are stored in `co2s.uncertainty`, for which `co2s.u` can be used as a shortcut.

    * For each result `into` in `uncertainty_into`, there is a new sub-dict `co2s.uncertainty[into]` containing the total and component uncertainties in that result.
  
    * The total uncertainty is in `co2s.uncertainty[into]`.
  
    * The uncertainties from each argument `from` that has had an uncertainty defined with `set_uncertainty` are also in the sub-dict with the corresponding keys: `co2s.uncertainty[into][from]`.

    All `into` and `from` values can be accessed with dot notation instead of with square brackets, and the [shortcuts](detail.md/#arguments-and-results) can be used.

## Uncertainties with covariances

PyCO2SYS does not currently have a generalised function for the complete process of propagating uncertainties that co-vary.  However, it does allow the derivative of any result with respect to any argument to be calculated:

```python
grads_of = ["pH"]  # Get derivatives of pH...
grads_wrt = ["dic", "alkalinity"]  # ... with respect to DIC and alkalinity
co2s.get_grads(grads_of, grads_wrt)

# Access derivatives
dpH_ddic = co2s.grads["pH"]["dic"]
dpH_dalk = co2s.grads["pH"]["alkalinity"]
```

!!! inputs "`get_grads` arguments"

    * `grads_of`: a list of parameter keys for which the derivatives of are to be calculated.
    * `grads_wrt`: a list of parameter keys for which the derivatives with respect to are to be calculated.

!!! outputs "`get_grads` results"

    For each result `of` in `grads_of` and argument `wrt` in `grads_wrt`, the corresponding derivative is stored in `co2s.grads[of][wrt]`.

!!! warning "No shortcuts here"

    The shortcuts cannot be used in the arguments `grads_of` and `grads_wrt` in the `get_grads` function, nor when accessing keys from `co2s.grads`.