# Calculate everything with `CO2SYS`

## Syntax

The simplest way to use PyCO2SYS is to follow the approach of previous versions of CO<sub>2</sub>SYS and calculate every possible variable of interest at once. We can do this using the top-level `CO2SYS` function:

    :::python
    from PyCO2SYS import CO2SYS
    CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
        PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
        NH3=0.0, H2S=0.0, KFCONSTANT=1)

## Inputs

Most of the inputs should be familiar to previous users of CO<sub>2</sub>SYS for MATLAB, and they work exactly the same here. Each input can either be a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.

!!! info "`PyCO2SYS.CO2SYS` inputs"
    ***Carbonate system parameters***

    * `PAR1` and `PAR2`: values of two different carbonate system parameters.
    * `PAR1TYPE` and `PAR2TYPE`: which types of parameter `PAR1` and `PAR2` are.

    These can be any pair of:

    * **Total alkalinity** (type `1`) in μmol·kg<sup>−1</sup>.
    * **Dissolved inorganic carbon** (type `2`) in μmol·kg<sup>−1</sup>.
    * **p[H<sup>+</sup>]** (type `3`) on the Total, Seawater, Free or NBS scale[^1]. Which scale is given by the input `pHSCALEIN`.
    * **Partial pressure** (type `4`) or **fugacity of CO<sub>2</sub>** (type `5`) in μatm.
    * **Carbonate ion concentration** (type `6`) in μmol·kg<sup>−1</sup>

    For all inputs in μmol·kg<sup>−1</sup>, the "kg" refers to the total solution, not H<sub>2</sub>O. These are therefore accurately termed *molinity* values (as opposed to *concentration* or *molality*).

    ---

    ***Hydrographic conditions***

    * `SAL`: practical salinity (dimensionless).
    * `TEMPIN`: temperature at which `PAR1` and `PAR2` inputs are provided in °C.
    * `TEMPOUT`: temperature at which output results will be calculated in °C.
    * `PRESIN`: pressure at which `PAR1` and `PAR2` inputs are provided in dbar.
    * `PRESOUT`: pressure at which output results will be calculated in dbar.

    For example, if a sample was collected at 1000 dbar pressure (~1 km depth) at an in situ water temperature of 2.5 °C and subsequently measured in a lab at 25 °C, then the correct values would be `TEMPIN = 25`, `TEMPOUT = 2.5`, `PRESIN = 0`, and `PRESIN = 1000`.

    ---

    ***Nutrients***

    *Required:*

    * `SI`: total silicate in μmol·kg<sup>−1</sup>.
    * `PO4`: total phosphate in μmol·kg<sup>−1</sup>.

    *Optional (these default to zero if not specified):*

    * `NH3`: total ammonia in μmol·kg<sup>−1</sup>.
    * `H2S`: total hydrogen sulfide in μmol·kg<sup>−1</sup>.

    Again, the "kg" in μmol·kg<sup>−1</sup> refers to the total solution, not H<sub>2</sub>O. These are therefore accurately termed *molinity* values (as opposed to *concentration* or *molality*).

    ---

    ***Settings***

    *Required:*

    * `pHSCALEIN`: which pH scale was used for any pH entries in `PAR1` or `PAR2`, as defined by [ZW01](../refs/#z):
        * `1`: Total, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$.
        * `2`: Seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$.
        * `3`: Free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$.
        * `4`: NBS, i.e.

    * `K1K2CONSTANTS`: which set of equilibrium constants to use to model carbonic acid dissociation.
        * `1`: [RRV93](../refs/#r) (0 < *T* < 45 °C, 5 < *S* < 45, Total, artificial seawater).

    * `KSO4CONSTANTS`: which equilibrium constant to use to model bisulfate ion dissociation **and** which boron:salinity relationship to use to estimate total borate.

    The bisulfate dissociation options are either [KRCB77](../refs/#k) or [D90a](../refs/#d). The borate:salinity options are either [U74](../refs/#u) or [LKB10](../refs/#l). The somewhat inelegant approach for combining these into a single option is inherited from CO<sub>2</sub>SYS for MATLAB and retained here for consistency.

    |        |  U74  | LKB10 |
    |-------:|:-----:|:-----:|
    | KRCB77 |  `1`  |  `3`  |
    |   D90a |  `2`  |  `4`  |

    *Optional:*

    * `KFCONSTANT`: which equilibrium constant to use for hydrogen fluoride dissociation.

[^1]: See [ZW01](../refs/#z) for definitions of the different pH scales.
