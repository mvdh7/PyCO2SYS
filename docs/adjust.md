# Adjust conditions

Use `adjust` to adjust the system to a different set of temperature and/or pressure conditions:

```python
import PyCO2SYS as pyco2

# Set up an initial CO2System
co2s = pyco2.sys(
    alkalinity=2250,
    pH=8.1,
    temperature=25,
    salinity=32,
)

# Create a new CO2System at a different temperature & pressure
co2s_adj = co2s.adjust(
    temperature=25,
    pressure=1000,
# Advanced kwargs, usually not needed:
    store_steps=1,
    method_fCO2=1,
    opt_which_fCO2_insitu=1,
    bh_upsilon=None,
)

# Calculate pCO2 at the adjusted conditions
pCO2_adj = co2s_adj["pCO2"]
```

The result `co2s_adj` is a new `CO2System` with all values at the new conditions (above, temperature of 25 °C and hydrostatic pressure of 1000 dbar).

If the original `co2s` was set up with the `data` kwarg from a pandas `DataFrame` or xarray `Dataset`, then the `temperature` and `pressure` provided to `adjust` can be pandas `Series`s or xarray `DataArrays` as long as their index or dimensions are consistent with the original `data`.

For more on the `store_steps` kwarg, see [Advanced results access](results.md/#solve-without-returning).

The `adjust` method can be used if any two carbonate system parameters are known, but also if only one of pCO<sub>2</sub>, fCO<sub>2</sub>, [CO<sub>2</sub>(aq)] or *x*CO<sub>2</sub> is known.  In this case, `adjust` can take additional kwargs:

  * `method_fCO2`: how to do the temperature conversion.
    * **`1`: using the parameterised <i>υ<sub>h</sub></i> equation of [H24](refs.md/#h) (default)**. 
    * `2`: using the constant <i>υ<sub>h</sub></i> fitted to the [TOG93](refs.md/#t) dataset by [H24](refs.md/#h).
    * `3`: using the constant theoretical <i>υ<sub>x</sub></i> of [H24](refs.md/#h).
    * `4`: following the [H24](refs.md/#h) approach but using a user-provided $b_h$ value (given with the additional kwarg `bh_upsilon`).
    * `5`: using the linear fit of [TOG93](refs.md/#t).
    * `6`: using the quadratic fit of [TOG93](refs.md/#t) (default before v1.8.3).
  
  * `opt_which_fCO2_insitu`: whether the **input (`1`, default)** or output (`2`) condition pCO<sub>2</sub>, fCO<sub>2</sub>, [CO<sub>2</sub>(aq)] and/or <i>x</i>CO<sub>2</sub> values are at in situ conditions, for determining $b_h$ with the parameterisation of [H24](refs.md/#h).  Only applies when `method_fCO2` is `1`.

  * `bh_upsilon`: If this is a single-parameter system and `method_fCO2` is `4`, then the value of $b_h$ in J/mol must be specified here.
