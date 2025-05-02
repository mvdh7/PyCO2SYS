# Quick-start guide

All the code examples assume the following import convention:

```python
import PyCO2SYS as pyco2
```

## Solve the marine carbonate system

The only function most users will need from PyCO2SYS is `pyco2.sys`.  For example:

```python
# Set up a CO2System
co2s = pyco2.sys(alkalinity=2250, dic=2100, temperature=15, salinity=34)

# Solve for and return the value of pH
pH = co2s["pH"]

# Solve for and return pCO2 and fCO2 at the same time
results = co2s[["pCO2", "fCO2"]]
```

Results can be calculated and accessed with square brackets, as if `co2s` were a dict.  It isn't a dict, it's a `CO2System`, so it can do some other things too.

Each call of `pyco2.sys` may include up to two known core marine carbonate system parameters, which are DIC, total alkalinity, pH, <i>p</i>CO<sub>2</sub>, <i>f</i>CO<sub>2</sub>, <i>x</i>CO<sub>2</sub>, (bi)carbonate ion content, and the saturation state of aragonite and calcite.

!!! tip "Find out more"

    See [User guide / Arguments and results](detail.md) for the full sets of keyword arguments that can be provided to `pyco2.sys` and the results parameters that can be calculated.

    See [User guide / Advanced results access](results.md) for a more detailed overview of how results can be solved for and accessed from a `CO2System`.

A few common examples are given below.

## Calculate without core parameters

Some properties (mainly equilibrium constants and total salt contents) can be calculated without solving the marine carbonate system, so `pyco2.sys` can be run with no core parameters:

```python
# Set up a CO2System under default conditions 
# (temperature 25 °C, salinity 35, hydrostatic pressure 0 dbar
#   - other values could be specified with the appropriate kwargs)
co2s = pyco2.sys()

# Get water dissociation constant
pk_H2O = co2s["pk_H2O"]
```

## Use different parameterisations

PyCO2SYS contains many different options for the parameterisations of equilibrium constants and total salt contents.  These can be selected using `kwargs` beginning with `opt_`, for example:

```python
# Set up a CO2System with non-default equilibrium constants for carbonic acid
# and non-default total borate from salinity
co2s = pyco2.sys(opt_k_carbonic=3, opt_total_borate=2)
```

All settings arguments must be single, scalar, integer values.

## Convert to different temperatures and/or pressures

!!! tip "Find out more"

    See [User guide / Adjust conditions](adjust.md) for more detail on temperature and pressure conversions.

### With two known parameters

To convert parameters to different temperatures and/or pressures, use the `adjust` method.

For example, to calculate the saturation state with respect to aragonite under in situ conditions from alkalinity and pH measured in the laboratory at 25 °C:

```python
# Set up the initial CO2System under lab conditions
co2s_lab = pyco2.sys(alkalinity=2250, pH=8.1, temperature=25)

# Adjust to in situ conditions (10 °C and 1500 dbar hydrostatic pressure)
co2s_insitu = co2s_lab.adjust(temperature=10, pressure=1500)
saturation_aragonite = co2s_insitu["saturation_aragonite"]
```

### With one known parameter

The partial pressure (`pCO2`), fugacity (`fCO2`), dry-air mole fraction (`xCO2`) and aqueous content (`CO2`) of CO<sub>2</sub> can be interconverted and adjusted to different temperatures without a second parameter:

```python
# Set up the initial CO2System with known pCO2
co2s_lab = pyco2.sys(pCO2=400, temperature=25)

# Calculate fCO2 under lab conditions (optional step)
fCO2_lab = co2s_lab["fCO2"]

# Adjust fCO2 to in situ conditions(10 °C and 1500 dbar hydrostatic pressure)
co2s_insitu = co2s_lab.adjust(temperature=10, pressure=1500)
fCO2_insitu = co2s_insitu["fCO2"]
```

## Propagate uncertainties

!!! tip "Find out more"

    See [User guide / Uncertainty propagation](uncertainty.md) for more detail on propagating uncertainties.

Uncertainties are defined and propagated using the `set_uncertainty` and `propagate` methods:

  * `set_uncertainty` is used to define the independent uncertainties in input parameters.  The kwargs used are the same as for the main `pyco2.sys` function.

  * `propagate` propagates the defined uncertainties through to the calculated results.

For example, to get the total uncertainty in pH from independent uncertainties in alkalinity and DIC:

```python
# Set up a CO2System
co2s = pyco2.sys(alkalinity=2250, dic=2100, temperature=15, salinity=34)

# Uncertainties in alkalinity and DIC are both 2 µmol/kg
co2s.set_uncertainty(alkalinity=2, dic=2)

# Propagate through to pH
co2s.propagate("pH")

# Retrieve total uncertainty in pH
pH_uncertainty = co2s.uncertainty["pH"]["total"]

# Retrieve component of pH uncertainty due to DIC
pH_uncertainty = co2s.uncertainty["pH"]["dic"]
```

## Multidimensional data

### NumPy arrays

All arguments other than settings can be provided as lists or multidimensional numpy arrays.  The dimensions of different arguments can be different as long as they can be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) together.

```python
# Define multidimensional arguments
dic = np.array([2000, 2100, 2200])
pCO2 = np.array([400, 450, 485])

# Set up a CO2System
co2s_1D = pyco2.sys(dic=dic, pCO2=pCO2)
alkalinity_1D = co2s_1D["alkalinity"]  # shape is (3,)

co2s_2D = pyco2.sys(dic=dic, pCO2=np.vstack(pCO2))
alkalinity_2D = co2s_2D["alkalinity"]  # shape is (3, 3)
```

### Data structures

Some common data structures can be provided to `pyco2.sys` using the `data` kwarg.

#### Dict(ionarie)s

If your data are in a `dict`, you can provide this as `data`:

```python
# Define known parameters
df = {"dic": [2000, 2100, 2200], "pCO2": [400, 450, 485]}

# Set up a CO2System, including an extra parameter that was not
# in the dict (total_silicate), and an optional setting
co2s = pyco2.sys(data=df, total_silicate=1.5, opt_k_carbonic=9)
```

If some of the `dict` keys do not match the kwargs expected by `pyco2.sys`, the correct keys can be given as the corresponding kwargs:

```python
# Define known parameters
df = {"dic_calibrated": [2000, 2100, 2200], "pCO2_data": [400, 450, 485]}

# Set up a CO2System
co2s = pyco2.sys(
    data=df,
    dic="dic_calibrated",
    pCO2="pCO2_data",
    total_silicate=1.5,
    opt_k_carbonic=9,
)
```

#### Pandas DataFrames

!!! warning "`DataFrames` not `Series`"
    Pandas data must be collected into a `DataFrame` and passed together through the `data` kwarg.  It's not possible to pass individual pandas `Series`s separately as kwargs to `pyco2.sys`.  If this is necessary, then each `Series` must first be converted into a NumPy array.

If data are in a pandas `DataFrame`, this can be provided as `data`, and results returned as a pandas `Series` or `DataFrame` with consistent indexing:

```python
# Define known parameters
df = pd.DataFrame({"dic": [2000, 2100, 2200], "pCO2": [400, 450, 485]})

# Set up a CO2System
co2s = pyco2.sys(data=df, total_silicate=1.5, opt_k_carbonic=9)

# Solve for and return a parameter as a Series
pH = co2s.to_pandas("pH")

# Solve for and return parameters as a DataFrame
df_results = co2s.to_pandas(["pH", "alkalinity"])
```

Running `to_pandas` with no arguments will return a `DataFrame` containing all currently calculated parameters.

#### Xarray Datasets

!!! warning "`Dataset` not `DataArray`"
    Xarray data must be collected into a `Dataset` and passed together through the `data` kwarg.  It's not possible to pass individual xarray `DataArray`s separately as kwargs to `pyco2.sys`.  If this is necessary, then each `DataArray` must first be converted into a NumPy array.

If data are in an xarray `Dataset`, this can be provided as `data`, and results returned as an xarray `DataArray` or `Dataset` with consistent dimensions:

```python
# Define known parameters
ds = xr.Dataset({
    "temperature": ("dim_t", np.arange(0, 35)),
    "salinity": ("dim_s", np.arange(30, 40)),
})

# Set up a CO2System
co2s = pyco2.sys(data=ds)

# Solve for and return a parameter as a DataArray
k_CO2 = co2s.to_xarray("k_CO2")

# Solve for and return parameters as a Dataset
ds_results = co2s.to_xarray(["k_H2CO3", "k_HCO3"])
```

Running `to_xarray` with no arguments will return a `Dataset` containing all currently calculated parameters.
