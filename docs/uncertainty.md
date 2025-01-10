# Uncertainty propagation

PyCO2SYS provides tools to propagate uncertainties in all arguments through to all results of its marine carbonate system calculations.  All derivatives needed for uncertainty propagation are calculated with automatic differentiation.

## Independent uncertainties

If the uncertainty in each [argument](detail.md/#arguments) is independent – i.e. there is no covariance between the uncertainties in different parameters – then you can use the `propagate` method to propagate the parameter uncertainties through into any [result](detail.md/#results).

### Syntax

Uncertainty propagation can be performed with the `propagate` method, which has the syntax:

```python
co2s.propagate(uncertainty_from, uncertainty_into)
```

where `co2s` is a `CO2System`.

#### Arguments

  * `uncertainty_into` is a list of strings of the [results keys](detail.md/#results) to propagate uncertainties into.

  * `uncertainty_from` is a dict of the uncertainties in the arguments to propagate through `pyco2.sys`.

The keys of `uncertainty_from` can include any `CO2System` [arguments](detail.md/#arguments) that can have an uncertainty.  The key for each uncertainty in `uncertainty_from` should be the same as the corresponding key in the `CO2System` [results](detail.md/#results).

Some additional considerations:

  * To provide a fractional value for any uncertainty, append `"__f"` to the end of its key in `uncertainty_from`.

  * For the equilibrium constants, to propagate an uncertainty in terms of a p<i>K</i> value rather than <i>K</i>, prefix the corresponding key in `uncertainty_from` with a `"p"` (e.g. use `"pk_H2CO3"` instead of `"k_H2CO3"`).

  * The "standard" uncertainties in the equilbrium constants and total borate used by CO2SYS for MATLAB following [OEDG18](refs.md/#o) are available as a dict in the correct format for `uncertainty_from` at `pyco2.uncertainty_OEDG18`.

  * The values of `uncertainty_from` are the uncertainties in each input parameter as a standard deviation.  You can provide a single value if all uncertainties are the same for a parameter, or an array of the same size as the parameter if they are different.  Any parameters not included are assumed to have zero uncertainty.

#### Results

The uncertainty results are added to `co2s.uncertainty`.
    
  * For each result `into` in `uncertainty_into`, there is a new sub-dict `co2s.uncertainty[into]` containing the total and component uncertainties in that result.

  * The total uncertainty is in `co2s.uncertainty[into]["total"]`.

  * The uncertainties from each argument `from` in `uncertainty_from` are also in the sub-dict with the corresponding keys: `co2s.uncertainty[into][from]`.

The total uncertainties are the Pythagorean sum of all the components.  This calculation assumes that all argument uncertainties are independent from each other and that they are provided in terms of single standard deviations.

#### Example calculation

An example calculation, explained below:

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

# Propagate uncertainties
co2s.propagate(
    ["dic", "fCO2"],
    {
        "alkalinity": 2,
        "pH": 0.02,
        "pk_H2O": 0.01,
    }
)

# Access propagated uncertainties
dic_uncertainty = co2s.uncertainty["dic"]["total"]
fCO2_uncertainty_from_pH = co2s.uncertainty["fCO2"]["pH"]
```

Above, we propagated independent uncertainties in alkalinity (`alkalinity`; 2&nbsp;µmol&nbsp;kg<sup>–1</sup>), pH (`pH`, 0.02) and p<i>K</i>*(H<sub>2</sub>O) (`pk_H2O`; 0.01) through to DIC (`dic`) and fCO<sub>2</sub> (`fCO2`).

The results of the propagation can be accessed in the `co2s.uncertainty` dict, which includes both individual component contributions to the final uncertainty (e.g., `co2s.uncertainty["fCO2"]["pH"]`) as well as the total uncertainty calculated assuming the components are independent (e.g., `co2s.uncertainty["dic"]["total"]`).

## Uncertainties with covariances

PyCO2SYS does not currently have a generalised function for the complete process of propagating uncertainties that co-vary.  However, it does allow you calculate the derivative of any result with respect to any argument.  The syntax is similar as described above for uncertainties:

```python
co2s.get_grads(grads_of, grads_wrt)
```

 In general, this works the same as the uncertainty propagation approach described in the previous section.  The main differences are:

  * `grads_of` is equivalent to `uncertainty_into`.
  * `grads_wrt` (w.r.t. = with respect to) is equivalent to `uncertainty_from`, but values are not required, so it can be a list.  A dict is also fine; its values are ignored.
  * The `"__f"` key extension cannot be used in `grads_wrt`.
  * For each result `of` in `grads_of` and argument `wrt` in `grads_wrt`, the corresponding derivative is added to `co2s.grads[of][wrt]`.
