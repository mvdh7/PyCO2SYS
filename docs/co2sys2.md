# PyCO2SYS v2

## Syntax

You can reate a `CO2System` by providing values of all known parameters (`values`) as a dict.  For example:

```python
from PyCO2SYS import CO2System

values = dict(
    alkalinity=2250,
    pH=8.1,
    temperature=12.5,
    salinity=32.4,
)

sys = CO2System(values)
```

Although only scalar values are provided above, each parameter can be a NumPy array of any shape, as long as all the arrays can be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) together.  A full list of the allowed keywords in `values` is provided below.

To calculate unknown parameters, use the `solve` method:

```python
# Calculate DIC and pCO2
results = sys.solve(["dic", "pCO2"])

# Or, extract the DIC and pCO2 results
dic = sys.values["dic"]
pCO2 = sys.values["pCO2"]
```

This calculates the requested parameter(s) and stores them in the dict `results`.  Together with any intermediate parameters calculated along the way (e.g., equilibrium constants), they are also stored within the `CO2System` in a dict, `sys.values`.

## Differences with v1's `pyco2.sys`

| v1 `pyco2.sys` | v2 `CO2System` |
| -------------- | -------------- |
| All possible parameters are always calculated. | Only the requested parameters and the required intermediates are calculated when calling `CO2System.solve`. |
| Known parameters and settings are all provided as kwargs.  | Known parameters (`values`) and settings (`opts`) are provided as separate dicts when initalising a `CO2System`. |
| Settings (kwargs beginning `opt_`) can be multidimensional. | Settings (`opts`) must all be scalars. |
| Known marine carbonate system parameters are provided as `par1` and `par2`, with their types given by `par1_type` and `par2_type`. | Known marine carbonate system core parameters are provided as `alkalinity`, `dic`, `pH`, etc. |
| Multiple different combinations of `par1_type` and `par2_type` can be provided. | Each `CO2System` instance can only contain one particular combination of core parameters. |

```python
# v1
kwargs = dict(
    par1=2300,
    par2=2150,
    par1_type=1,
    par2_type=2,
    opt_k_carbonic=1,
)
results = pyco2.sys(**kwargs)

# v2
values = dict(alkalinity=2300, dic=2150)
opts = dict(opt_k_carbonic=1)
sys = CO2System(values, opts)
sys.solve()
results = sys.values
```

```python
# v1
kwargs = dict(
    par1=2300,
    par2=8.1,
    par1_type=1,
    par2_type=3,
    temperature=25,
    temperature_out=15,
    pressure=0,
    pressure_out=100,
)
results = pyco2.sys(**kwargs)
pH_out = results["pH_out"]

# v2
values = dict(alkalinity=2300, pH=8.1, temperature=25, pressure=0)
sys = CO2System(values)
values_out = dict(temperature=15, pressure=100)
adj = sys.adjust(values_out)
adj.solve("pH")
pH_out = adj.values["pH"]
```
