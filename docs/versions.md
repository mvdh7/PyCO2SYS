# Version history

Version numbering follows [semantic versioning](https://semver.org/). Therefore:

  * New patch versions (e.g. 1.1.**0** to 1.1.**1**) make minor changes that do not alter fuctionality or calculated results.
  * New minor versions (e.g. 1.**0**.1 to 1.**1**.0) add new functionality, but will not break your code. They will not alter the results of calculations with default settings.
  * New major versions (e.g. **1**.1.1 to **2**.0.0) may break your code and require you to rewrite things. They may alter the results of calculations with default settings.

*Will (not) break your code* refers **only** to how you use the main `CO2SYS` function as imported with:

```python
from PyCO2SYS import CO2SYS
```

**However, the structure of the underlying modules and their functions is not yet stable and, for now, may change in any version increment.** Such changes will be described in the release notes below.

---

## 1.3

### 1.3.0

**Release date:** forthcoming [doi:](https://doi.org/)

  * Rename module `extra` to `buffers`.
  * All functions in `solve` updated to be [Autograd](https://github.com/HIPS/autograd)-able.
  * Relocated `_CaSolubility` function from root into new module `solubility`.

---

## 1.2

Adding additional buffer factor calculations that are not currently included in CO<sub>2</sub>SYS for MATLAB. New releases henceforth assigned DOIs from Zenodo.

### 1.2.1

**Release date:** 9 April 2020 [[doi:10.5281/zenodo.3746347](https://doi.org/10.5281/zenodo.3746347)]

  * Fixed typo in [ESM10](../refs/#ESM10) equations that had been carried through into `extra.buffers_ESM10` function (thanks [Jim Orr](https://twitter.com/James1Orr/status/1248216403355803648)!).

### 1.2.0

**Release date:** 8 April 2020 [[doi:10.5281/zenodo.3744276](https://doi.org/10.5281/zenodo.3744276)]

  * Added module `extra` containing functions to calculate variables not included in CO2SYS for MATLAB:
    * `buffers_ESM10` calculates the buffer factors of [ESM10](../refs/#ESM10), corrected for the typos noted by [RAH18](../refs/#RAH18).
    * `bgc_isocap` calculates the "exact" isocapnic quotient of [HDW18](../refs/#HDW18), Eq. 8.
    * `bgc_isocap_approx` calculates the approximate isocapnic quotient of [HDW18](../refs/#HDW18), Eq. 7.
    * `psi` calculates the $\psi$ factor of [FCG94](../refs/#FCG94).
  * Added all functions in `extra` to the `CO2dict` output of the main `CO2SYS` function, and documented in the [Github repo README](https://github.com/mvdh7/PyCO2SYS#pyco2sys).

---

## 1.1

Adding extra optional inputs for consistency with Pierrot et al.'s tentatively forthcoming MATLAB "v1.21". Continuing to reorganise subfunctions into more Pythonic modules, while avoiding changing the actual mechanics of calculations.

### 1.1.1

**Release date:** 20 March 2020

  * Removed unnecessary `WhoseTB` input to `assemble.equilibria`.

### 1.1.0

**Release date:** 19 March 2020

  * Updated pH-solving iterative functions so that iteration stops separately for each row once it reaches the tolerance threshold.
  * Extracted all functions for solving the CO<sub>2</sub> system into a separate module (`solve`).
  * Extracted other key subfunctions into module `assemble`.
  * Added total ammonium (`NH3`) and hydrogen sulfide (`H2S`) concentrations as optional inputs to be included in the alkalinity model.
  * Added optional input to choose between different equations for hydrogen fluoride dissociation constant (`KFCONSTANT`).
  * Added functions to enable carbonate ion as an input carbonate system variable.
  * Output is now only the `CO2dict` dict, not the original `DATA`, `HEADERS` and `NICEHEADERS`.
  * Eliminated all global variables throughout the entire program.

---

## 1.0

### 1.0.1

**Release date:** 28 February 2020

Starting to make things more Pythonic.

  * Extracted all equations for concentrations and equilibrium constants into functions in separate modules (`concentrations` and `equilibria`).
  * Eliminated all global variables from the `_Constants` function.
  * Moved the as-close-as-possible version into module `original`. The default `from PyCO2SYS import CO2SYS` now imports the more Pythonic implementation.

### 1.0.0

**Release date:** 3 February 2020

An as-close-as-possible clone of MATLAB CO2SYS v2.0.5, obtained from [github.com/jamesorr/CO2SYS-MATLAB](https://github.com/jamesorr/CO2SYS-MATLAB).

  * The first output `DICT` is new: a dict containing a separate entry for each variable in the original output `DATA`, with the keys named following the original output `HEADERS`.
  * The output `DATA` is transposed relative to the MATLAB version because Numpy is row-major while MATLAB is column-major.
  * Every combination of input options was tested against the MATLAB version with no significant differences (i.e. all differences can be attributed to floating point errors).
