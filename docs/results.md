# Advanced results access

The results can be solved and accessed in several ways.

## As if it were a dict

First is to treat the `co2s` as a dict and access with the keys given in [Arguments and results](detail.md/#results).

```python
import PyCO2SYS as pyco2

# Set up the CO2System
co2s = pyco2.sys(alkalinity=2250, dic=2100)

# Solve for and return pH
pH = co2s["pH"]
```

The initial call to `pyco2.sys` does not solve for any parameters.  The pH value is determined only when it is accessed with `co2s["pH"]`.  Intermediate parameters (e.g., equilibrium constants) computed along the way are also stored in the `CO2System` at this point, so a subsequent call to calculate e.g. pCO<sub>2</sub> will use these (and the now-known pH value) instead of repeating those calculations.

## Multiple values at once

We can also return multiple values at once by providing their keys as a list:

```python
# Solve for and return pH and pCO2
results = co2s[["pH", "pCO2"]]
```

The `results` are a dict containing the values of pH and pCO<sub>2</sub>.

## Solve without returning

We can solve for a parameter without returning its value using the `solve` method.  This gives more control over how intermediate parameters are handled:

```python
co2s.solve(["pH", "pCO2"], store_steps=1)
```

The kwarg `store_steps` allows us to determine which intermediate parameters are stored internally after the calculation is complete.  It can be set to:

  * `0`: store only the specifically requested parameters.
  * **`1`: store the most used set of intermediate parameters (default).**
  * `2`: store the complete set of parameters