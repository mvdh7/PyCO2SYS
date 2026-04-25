!!! danger "PyCO2SYS v2 beta"

    These are the docs for the beta release of PyCO2SYS v2!

    If you're here to test it, then thank you! and please report any issues via [the GitHub repo](https://github.com/mvdh7/PyCO2SYS/issues).

    **These instructions will not work for the current version 1.8** that can be installed through `pip` and `conda` - please see [PyCO2SYS.readthedocs.io](https://pyco2sys.readthedocs.io/en/latest/) for documentation for the latest stable release.

# Quick-start guide

All the code examples assume the following import convention:

```python
import PyCO2SYS as pyco2
```

---

## Solve the marine carbonate system

The only function needed is `pyco2.sys`.  For example:

```python
# Set up a CO2System
co2s = pyco2.sys(
    alkalinity=2250,
    dic=2100,
    temperature=15.1,
    salinity=34.4,
    total_silicate=8.3,
    opt_total_borate=2,  # use LKB10 for total borate
)

# Solve for and return the value of pH (either option below works)
pH = co2s.pH
pH = co2s["pH"]

# Solve for and return pCO2 and fCO2 at the same time:
# `results` is a dict with keys "pCO2" and "fCO2"
results = co2s[["pCO2", "fCO2"]]
```

Each call of `pyco2.sys` may include up to two known core marine carbonate system parameters, any other ther relevant parameters like hydrographic properties and nutrient contents, and any settings to control which parameterisations are used.

All parameters can be scalars, lists or multi-dimensional NumPy arrays.  For pandas DataFrames and xarray Datasets, use the `data` kwarg as described below in [Data structures](#data-structures).  All settings (beginning with `opt_`) must be scalar integers.

Parameter names are all case-insensitive and many have shortcuts that can be used instead.  See [Arguments and results](detail.md) for a list of all the options.

---

## Adjust to different temperatures and/or pressures

To convert parameters to different temperatures and/or pressures, use the `adjust` method.  See [Adjust conditions](adjust.md) for more detail on temperature and pressure conversions.

### With two known parameters

For example, to calculate the saturation state with respect to aragonite under in situ conditions from alkalinity and pH measured in the laboratory at 25 °C:

```python
# Set up the initial CO2System under lab conditions
co2s_lab = pyco2.sys(alkalinity=2250, pH=8.1, temperature=25, salinity=32)

# Adjust to in situ conditions (10 °C and 1500 dbar hydrostatic pressure)
co2s_insitu = co2s_lab.adjust(temperature=10, pressure=1500)
saturation_aragonite = co2s_insitu["saturation_aragonite"]
```

### With one known parameter

The partial pressure (`pCO2`), fugacity (`fCO2`), dry-air mole fraction (`xCO2`) and aqueous CO<sub>2</sub> content (`CO2`) can be interconverted and adjusted to different temperatures without a second parameter ([H24](refs/#h)):

```python
# Set up the initial CO2System with known pCO2
co2s_lab = pyco2.sys(pCO2=400, temperature=25)

# Convert to fCO2 under initial conditions (optional)
fCO2_lab = co2s_lab.fCO2

# Calculate fCO2 in situ (10 °C)
co2s_insitu = co2s_lab.adjust(temperature=10)
fCO2_insitu = co2s_insitu.fCO2
```

---

## Propagate uncertainties

Uncertainties are assigned using the `set_u`(1) method and propagated with `prop`(2).  For more details, see [Uncertainty propagation](uncertainty.md).
{ .annotate }

1.  Shortcut for `set_uncertainty`.
2.  Shortcut for `propagate`.

!!! warning "Variance, not standard deviation"

    Uncertainty values must be assigned in terms of variances, not as standard deviations.

    Calculated uncertainties are also returned as variances.

The uncertainty for each parameter can be given either as a scalar, an array of the same shape as the parameter, or as a covariance matrix.

For example, to get the total uncertainty in pH from uncertainties in alkalinity and DIC and including the set of uncertainties for equilibrium constants and total borate content proposed by [OEDG18](refs/#o):

```python
# Set up a CO2System
co2s = pyco2.sys(alkalinity=2250, dic=2100, temperature=15, salinity=34)

# 1-sigma uncertainties in alkalinity and DIC are both 2 µmol/kg
co2s.set_u(alkalinity=2**2, dic=2**2)

# Also assign all OEDG18 uncertainties
co2s.set_u_OEDG18()

# Propagate through to pH
co2s.prop("pH")

# Retrieve total uncertainty as a variance in pH
# (take the square root to get back to a standard deviation)
pH_var = co2s.u.pH

# Retrieve component of pH uncertainty due to DIC
pH_var_dic = co2s.u.parts["pH"]["dic"]
```

---

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

#### Dicts

If your data are in a dict, you can provide this as `data`:

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

#### pandas DataFrames

!!! warning "DataFrame not Series"
    pandas data must be collected into a DataFrame and passed together through the `data` kwarg.  It's not possible to pass individual pandas Series separately as kwargs to `pyco2.sys`.  If this is necessary, then each Series must first be converted into a NumPy array.

If data are in a pandas DataFrame, this can be provided as `data`, and results returned as a pandas Series or DataFrame with consistent indexing:

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

Running `to_pandas` with no arguments will return a DataFrame containing all currently calculated parameters.

#### xarray Datasets

!!! warning "Dataset not DataArray"
    xarray data must be collected into a Dataset and passed together through the `data` kwarg.  It's not possible to pass individual xarray DataArrays separately as kwargs to `pyco2.sys`.  If this is necessary, then each DataArray must first be converted into a NumPy array.

If data are in an xarray Dataset, this can be provided as `data`, and results returned as an xarray DataArray or Dataset with consistent dimensions:

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

Running `to_xarray` with no arguments will return a Dataset containing all currently calculated parameters.
