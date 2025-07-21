!!! danger "PyCO2SYS v2 beta"

    These are the docs for the beta release of PyCO2SYS v2!

    If you're here to test it, then thank you! and please report any issues via [the GitHub repo](https://github.com/mvdh7/PyCO2SYS/issues).

    **These instructions will not work for the current version 1.8** that can be installed through `pip` and `conda` - please see [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/) for documentation for the latest stable release.

# Adjust conditions

The `adjust` method can be used to adjust a `CO2System` to a different set of temperature and/or pressure conditions.  How the adjustment is done depends on how many core carbonate system parameters are known.

## Two parameters known

```python
import PyCO2SYS as pyco2

# Set up the initial CO2System (e.g. under lab conditions for pH)
co2s_lab = pyco2.sys(dic=2100, pH=8.1, temperature=25, pressure=0, salinity=32)

# Adjust to a different temperature and pressure (e.g. in situ conditions)
co2s_insitu = co2s_lab.adjust(temperature=10.5, pressure=1500)

# Solve for and return fCO2 under the lab and in situ conditions
fCO2_lab = co2s_lab["fCO2"]
fCO2_insitu = co2s_insitu["fCO2"]
```

Both `co2s_lab` and `co2s_insitu` are fully functional and independent `CO2System`s.

!!! inputs "Allowed kwargs for `adjust` with two known parameters"

      * `temperature`: the temperature to adjust to in °C.
      * `pressure`: the hydrostatic pressure to adjust to in dbar.

    If either `temperature` or `pressure` is not provided or `None`, then the 

    If the original `co2s` was set up with the `data` kwarg from a pandas `DataFrame` or xarray `Dataset`, then the `temperature` and `pressure` provided to `adjust` can be pandas `Series`s or xarray `DataArray`s, as long as their index or dimensions are consistent with the original `data`.

    Any other system properties (e.g. `salinity`, total salt contents, optional settings) must be defined when creating the original, unadjusted `CO2System`.  They cannot be added in during the `adjust` step.

`co2s_insitu` retains the minimum set of values under initial conditions that are needed to make the adjustment.  These pre-adjustment values all have the suffix `"__pre"` within the `CO2System`:

```python
# Get the pre-adjustment pH value in the adjusted CO2System
pH_lab = co2s_insitu["pH__pre"]  # same as co2s_lab["pH"]

# Solve for and return the adjusted pH value
pH_insitu = co2s_insitu["pH"]
```

Any uncertainties that were defined for these values are carried across to the new system.

!!! info "How it works"

    To go from an initial to and adjusted set of temperature and/or pressure conditions, we

      1. Solve *for* DIC and alkalinity under the initial conditions, and then
      2. Solve *from* DIC and alkalinity under the adjusted conditions.

### DIC and total alkalinity

In this case, `adjust` is not needed.  DIC and alkalinity are sensitive to neither temperature nor pressure.

Running `adjust` in this case will therefore raise an exception.  To solve at a different temperature and/or pressure, create a new `CO2System` with `pyco2.sys`, providing the different values directly.

```python
import PyCO2SYS as pyco2

# Solve from DIC and alkalinity at a given T/P
co2s = pyco2.sys(dic=2100, alkalinity=2250, temperature=25, pressure=0)

# Solve again at a different T/P
co2a = pyco2.sys(dic=2100, alkalinity=2250, temperature=10.5, pressure=1500)
```

### Two T/P-sensitive parameters at different T/P

As in previous versions of (Py)CO2SYS, there is no built-in way to handle the (rare) case where both known parameters are temperature- and/or pressure-sensitive **and** the two known parameters are at a different temperature and/or pressure from each other.

## One parameter known

The `adjust` method can also be used to adjust temperature (but not pressure) if only one of pCO<sub>2</sub>, fCO<sub>2</sub>, [CO<sub>2</sub>(aq)] or *x*CO<sub>2</sub> is known.

```python
import PyCO2SYS as pyco2

# Set up the initial CO2System (e.g. under lab conditions for xCO2)
co2s_lab = pyco2.sys(xCO2=425, temperature=25, pressure=0, salinity=32)

# Adjust to a different temperature (e.g. in situ conditions)
co2s_insitu = co2s_lab.adjust(temperature=10.5)

# Solve for and return fCO2 under the lab and in situ conditions
fCO2_lab = co2s_lab["fCO2"]
fCO2_insitu = co2s_insitu["fCO2"]
```

!!! inputs "Allowed kwargs for `adjust` with one known parameter"

    * `temperature`: the temperature to adjust to in °C.

    * `method_fCO2`: how to do the temperature conversion:
        * **`1`: using the parameterised <i>υ<sub>h</sub></i> equation of [H24](refs.md/#h) (default)**. 
        * `2`: using the constant <i>υ<sub>h</sub></i> fitted to the [TOG93](refs.md/#t) dataset by [H24](refs.md/#h).
        * `3`: using the constant theoretical <i>υ<sub>x</sub></i> of [H24](refs.md/#h).
        * `4`: following the [H24](refs.md/#h) approach, but using a user-provided $b_h$ value (see `bh` below).
        * `5`: using the linear fit of [TOG93](refs.md/#t).
        * `6`: using the quadratic fit of [TOG93](refs.md/#t).
  
    Only when `method_fCO2` is `1`:

    * `which_fCO2_insitu`: whether the **input (`1`, default)** or output (`2`) condition pCO<sub>2</sub>, fCO<sub>2</sub>, [CO<sub>2</sub>(aq)] and/or <i>x</i>CO<sub>2</sub> values are at in situ conditions, for determining $b_h$ with the parameterisation of [H24](refs.md/#h).
  
    Only when `method_fCO2` is `4`:

    * `bh`: $b_h$ in J/mol.
