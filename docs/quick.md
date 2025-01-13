# Quick-start guide

For most users, the only function needed from PyCO2SYS is `pyco2.sys`.  This uses the following basic syntax:

```python
import PyCO2SYS as pyco2

co2s = pyco2.sys(**kwargs)
result = co2s[result_key]
```

Results can be calculated and accessed with square brackets, as if `co2s` were a dict.  (It isn't a dict, it's a `CO2System`, which means that it can also do some other things too.)

The full sets of `kwargs` that can be provided and results parameters that can be calculated are given in [Arguments and results](detail.md).  [Advanced results access](results.md) gives a more detailed overview of how results can be accessed from a `CO2System`.  A few commonly used examples are also given below.

## Solve the marine carbonate system

When two marine carbonate system parameters are known, such as total alkalinity and dissolved inorganic carbon (DIC), we can calculate other parameters such as pH:

```python
# Set up the CO2System
co2s = pyco2.sys(alkalinity=2250, dic=2100, temperature=15, salinity=34)

# Solve for and return the value of pH
pH = co2s["pH"]

# Solve for and return pCO2 and fCO2 at the same time
results = co2s[["pCO2", "fCO2"]]
```

Each call of `pyco2.sys` may include a maximum of two known marine carbonate system parameters.

## Calculations without solving the system

Some properties (mainly equilibrium constants and total salt contents) can be calculated without solving the marine carbonate system, so `pyco2.sys` can be run with no marine carbonate system parameters:

```python
# Set up a CO2System under default conditions 
# (temperature 25 °C, salinity 35, hydrostatic pressure 0 dbar
#   - other values could be specified with the appropriate kwargs)
co2s = pyco2.sys()

# Get water dissociation constant
k_H2O = co2s["k_H2O"]
```

## Use different parameterisations

PyCO2SYS contains many different options for the parameterisations of equilibrium constants and total salt contents.  These can be selected using `kwargs` beginning with `opt_`, for example:

```python
# Set up a CO2System with non-default equilibrium constants for carbonic acid
# and non-default total borate from salinity
co2s = pyco2.sys(opt_k_carbonic=3, opt_total_borate=2)
```

All settings arguments must be single scalar values.

## Convert to different temperatures and/or pressures

> Discussed in more detail in [Adjust conditions](adjust.md).

### With two known parameters

To convert parameters to different temperatures and/or pressures, use the `adjust` method.  For example, if we had measured alkalinity and pH in the laboratory at 25 °C, but wanted to calculate the saturation state with respect to aragonite under in situ conditions:

```python
# Set up the initial CO2System
co2s_lab = pyco2.sys(alkalinity=2250, pH=8.1, temperature=25)

# Adjust to in situ conditions (10 °C and 1500 dbar hydrostatic pressure)
co2s_insitu = co2s_lab.adjust(temperature=10, pressure=1500)
saturation_aragonite = co2s_insitu["saturation_aragonite"]
```

### With one known parameter

Any of the partial pressure (`pCO2`), fugacity (`fCO2`), dry-air mole fraction (`xCO2`) or aqueous content (`CO2`) of CO<sub>2</sub> is known can be converted to different temperatures without a second parameter:

```python
# Set up the initial CO2System
co2s_lab = pyco2.sys(pCO2=400, temperature=25)

# Adjust to in situ conditions(10 °C and 1500 dbar hydrostatic pressure)
co2s_insitu = co2s_lab.adjust(temperature=10, pressure=1500)
pCO2_insitu = co2s_insitu["pCO2"]
```

## Propagate uncertainties

> Discussed in more detail in [Uncertainty propagation](uncertainty.md).

Uncertainty propagation uses the `propagate` method.  To get the total uncertainty in pH from independent uncertainties in alkalinity and DIC in the example above, use:

```python
# Uncertainties in alkalinity and DIC are both 2 µmol/kg
co2s.propagate("pH", {"alkalinity": 2, "dic": 2})
pH_uncertainty = co2s.uncertainty["pH"]["total"]
```

The individual components of the total uncertainty can also be found, for example:

```python
pH_uncertainty_from_dic = co2s.uncertainty["pH"]["dic"]
```

## Multidimensional data

### NumPy arrays

All arguments other than settings can be provided as lists or multidimensional numpy arrays.  The dimensions of different arguments can be different as long as they can be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) together (which they will be!).

```python
# Define multidimensional arguments
dic = np.array([2000, 2100, 2200])
pCO2 = np.array([400, 450, 485])

# Set up a CO2System
co2s_1D = pyco2.sys(dic=dic, pCO2=pCO2)
alkalinity_1D = co2s_1D["alkalinity"]  # this has the shape (3,)

co2s_2D = pyco2.sys(dic=dic, pCO2=np.vstack(pCO2))
alkalinity_2D = co2s_2D["alkalinity"]  # this has the shape (3, 3)
```

### Pandas DataFrames

If your data are in a pandas `DataFrame`, you can provide this as `data`, and return the results as a pandas `Series` or `DataFrame` with consistent indexing:

```python
# Define known parameters
df = pd.DataFrame({"dic": [2000, 2100, 2200], "pCO2": [400, 450, 485]})

# Set up a CO2System
co2s = pyco2.sys(data=df, total_silicate=1.5, opt_k_carbonic=9)

# Return a parameter as a Series
pH = co2s.to_pandas("pH")

# Return parameters as a DataFrame
df_results = co2s.to_pandas(["pH", "alkalinity"])
```

Running `to_pandas` with no arguments will return a `DataFrame` containing all currently calculated parameters.

### Xarray Datasets

If your data are in an xarray `Dataset`, you can provide this as `data`, and return the results as an xarray `DataArray` or `Dataset` with consistent dimensions:

```python
# Define known parameters
ds = xr.Dataset({
    "temperature": ("dim_t", np.arange(0, 35)),
    "salinity": ("dim_s", np.arange(30, 40)),
})

# Set up a CO2System
co2s = pyco2.sys(data=ds)

# Return a parameter as a DataArray
k_CO2 = co2s.to_xarray("k_CO2")

# Return parameters as a Dataset
ds_results = co2s.to_xarray(["k_H2CO3", "k_HCO3"])
```

Running `to_xarray` with no arguments will return a `Dataset` containing all currently calculated parameters.
