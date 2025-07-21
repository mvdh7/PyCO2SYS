!!! danger "PyCO2SYS v2 beta"

    These are the docs for the beta release of PyCO2SYS v2!

    If you're here to test it, then thank you! and please report any issues via [the GitHub repo](https://github.com/mvdh7/PyCO2SYS/issues).

    **These instructions will not work for the current version 1.8** that can be installed through `pip` and `conda` - please see [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/) for documentation for the latest stable release.

# Advanced tips and tricks

## Accessing results

### As if the `CO2System` were a dict

First is to treat the `CO2System` as a dict and access with the keys given in [Arguments and results](detail.md/#results):

```python
import PyCO2SYS as pyco2

# Set up the CO2System
co2s = pyco2.sys(alkalinity=2250, dic=2100)

# Solve for and return pH
pH = co2s["pH"]
```

The initial call to `pyco2.sys` does not solve for any parameters.  The pH value is determined only when it is accessed with `co2s["pH"]`.  Intermediate parameters (e.g., equilibrium constants) computed along the way are also stored in the `CO2System` at this point, so a subsequent call to calculate e.g. pCO<sub>2</sub> will use these (and the now-known pH value) instead of repeating those calculations.

### Multiple values at once

Multiple values can be solved for and returned at once by providing their keys as a list:

```python
# Solve for and return pH and pCO2
results = co2s[["pH", "pCO2"]]
```

`results` is a dict containing the values of pH and pCO<sub>2</sub>.

### Solve without returning

A parameter can be solved for without returning its value, using the `solve` method.  This gives more control over how intermediate parameters are handled:

```python
co2s.solve(parameters=None, store_steps=1)
```

!!! inputs "`solve` keyword arguments"

    `parameters`: a single parameter key as a string or list of parameter keys to solve for.  If `None` (default), then all possible parameters are solved for

    `store_steps` determines which intermediate parameters are stored internally after the calculation is complete:

      * `0`: store only the specifically requested parameters.
      * **`1`: store the most used set of intermediate parameters (default).**
      * `2`: store the complete set of parameters.

## Chaining methods

All of the `CO2System` methods can be chained together into a single "line" of code:

```python
co2s = (
    pyco2.sys(alkalinity=2300, pH=8.1, temperature=25)
    .set_uncertainty(alkalinity=2, pH=0.01, temperature=0.01)
    .adjust(temperature=12, pressure=1200)
    .solve(["saturation_aragonite", "fCO2"])
    .propagate("saturation_aragonite")
)
```

## Use shortcuts

All parameter keys are case-insensitive and some also have shorter versions that can be used instead (see [Arguments and results](detail.md)).

The example above (for [Chaining methods](#chaining-methods)) is great if the focus is on writing clear and human-readable final code that someone else can follow.  But if the focus is on quickly running some calculations, the same thing could be written much more concisely:

```python
co2s = (
    pyco2.sys(ta=2300, ph=8.1, t=25)
    .set_u(ta=2, ph=0.01, t=0.01)
    .adjust(t=12, p=1200)
    .solve(["oa", "fco2"])
    .prop("oa")
)
```

## Setting up the `CO2System`

Running the `pyco2.sys` function performs some conditioning of the arguments (converts `int` to `float` and all iterables to NumPy arrays) before passing these into the constructor for a `CO2System` object, which is returned.  If all arguments are already well-conditioned, then they can be passed directly to `PyCO2SYS.CO2System`, thus skipping the (minor) extra overhead of `pyco2.sys`.
