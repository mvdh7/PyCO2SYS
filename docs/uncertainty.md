!!! danger "PyCO2SYS v2 beta"

    These are the docs for the beta release of PyCO2SYS v2!

    If you're here to test it, then thank you! and please report any issues via [the GitHub repo](https://github.com/mvdh7/PyCO2SYS/issues).

    **These instructions will not work for the current version 1.8** that can be installed through `pip` and `conda` - please see [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/) for documentation for the latest stable release.

# Uncertainty propagation

Uncertainties are assigned and propagated with the `set_u`(1) and `prop`(2) methods.
{ .annotate }

1.  Shortcut for `set_uncertainty`.
2.  Shortcut for `propagate`.

## Assign uncertainties

All uncertainty values must be provided as **variances**, not standard deviations.

### Constant, independent

Provide a **single scalar value** if the uncertainty in each element of a parameter has the same value and is independent from the other elements.

```python
co2s = pyco2.sys(t=[5, 10, 15]).set_u(t=0.2)
```

> Above: three temperature values, each with an independent 1*σ* uncertainty of √0.2 °C.

### Variable, independent

Provide an **array with the same shape as the parameter** if the uncertainty in each element is different, but still is independent from the other elements.

```python
co2s = pyco2.sys(t=[5, 10, 15]).set_u(t=[0.1, 0.2, 0.3])
```

> Above: three temperature values, each with a different independent uncertainty.

### With covariances

Provide a **NumPy array with the shape of the parameter, squared** if there are covariances between uncertainties in different elements.

```python
import numpy as np

co2s = (
    pyco2.sys(t=[5, 10, 15])
    .set_u(t=np.array(
        [[0.1, 0.05, 0.05],
         [0.05, 0.2, 0.05],
         [0.05, 0.05, 0.3]])))
```

> Above: three temperature values with a three-by-three covariance matrix for their uncertainties.

### Standard set from OEDG18

To assign the "standard" set of uncertainties in the equilbrium constants and total borate assigned by [OEDG18](refs/#o), use the method `set_u_OEDG18`:

```python
co2s = (
    pyco2.sys(
        dic=2100,
        ta=2250,
        t=[5, 10, 15],
    )
    .set_u(t=0.1)
    .set_u_OEDG18()
)
```

Running `set_u` multiple times on the same `CO2System` adds to the existing set, overwriting existing entries.

!!! warning "`set_u` after `prop`"

    If `set_u` is run after running `prop` on a system, then `prop` will automatically be run again with the new set uncertainties, so that the values in `co2s.u` are all correct for the current set of assigned uncertainties.

### Finding assigned values

The assigned uncertainty values are stored in `co2s.u.assigned`(1).
{ .annotate }

1.  Shortcut for `co2s.uncertainty.assigned`.

## Propagate uncertainties

```python
# Propagate uncertainties that were set with set_u
co2s.prop(["pH", "fCO2"])

# Access uncertainty results
uncert_fCO2 = co2s.u["fCO2"]
uncert_pH_due_to_dic = co2s.u.parts["pH"]["t"]

# You can also use dot notation and shortcuts here
uncert_fCO2 = co2s.u.fCO2
uncert_pH_due_to_dic = co2s.u.parts.pH.t
```

The total uncertainties are the Pythagorean sum of all the components.  This calculation assumes that all argument uncertainties are independent from each other and that they are provided in terms of single standard deviations.

!!! inputs "`prop` arguments"

    * `uncertainty_into`: a list of the parameter keys that uncertainties are to be propagated into.

    If `prop` is run with no arguments, then uncertainties will be propagated into all results that have been currently solved for.

!!! outputs "`prop` results"

    The uncertainty results are stored in `co2s.uncertainty`, for which `co2s.u` can be used as a shortcut.

    * For each result `into` in `uncertainty_into`, there is a new sub-dict `co2s.u[into]` containing the total and component uncertainties in that result.
  
    * The total uncertainty is in `co2s.u[into]`.
  
    * The uncertainties from each argument `from` that has had an uncertainty defined with `set_u` are also in a sub-dict with the corresponding keys: `co2s.u.parts[into][from]`.

    All `into` and `from` values can be accessed with dot notation instead of with square brackets, and the [shortcuts](detail/#arguments-and-results) can be used.
