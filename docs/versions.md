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

## 1.0.1

**Release date:** 28 February 2020

Starting to make things more Pythonic.

  * Extracted all equations for concentrations and equilibrium constants into functions in separate modules (`concentrations` and `equilibria`).
  * Eliminated all global variables from the `_Constants` function.
  * Moved the as-close-as-possible version into module `original`. The default `from PyCO2SYS import CO2SYS` now imports the more Pythonic implementation.

## 1.0.0

**Release date:** 3 February 2020

An as-close-as-possible clone of MATLAB CO2SYS v2.0.5, obtained from [github.com/jamesorr/CO2SYS-MATLAB](https://github.com/jamesorr/CO2SYS-MATLAB).

  * The first output `DICT` is new: a dict containing a separate entry for each variable in the original output `DATA`, with the keys named following the original output `HEADERS`.
  * The output `DATA` is transposed relative to the MATLAB version because Numpy is row-major while MATLAB is column-major.
  * Every combination of input options was tested against the MATLAB version with no significant differences (i.e. all differences can be attributed to floating point errors).
