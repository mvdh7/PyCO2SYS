# Do it like in MATLAB

## Syntax

Up until v1.5.0, the simplest and safest way to use PyCO2SYS was to follow the approach of previous versions of CO2SYS and calculate every possible variable of interest at once using a MATLAB-style syntax.  A new, more powerful and more Pythonic interface has now been introduced, which you can [read about here](../co2sys_nd).  New developments to PyCO2SYS will focus on the other interface; this one is no longer being actively developed.

Read further on this page if you want to stick with the MATLAB-style syntax.  This is accessed using the top-level `CO2SYS` function:

```python
# Import the function
from PyCO2SYS import CO2SYS

# Run CO2SYS
CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
    PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
    NH3=0.0, H2S=0.0, KFCONSTANT=1, buffers_mode=1,
    totals=None, equilibria_in=None, equilibria_out=None, WhichR=1)

# Get (e.g.) aragonite saturation state, output conditions
OmegaARout = CO2dict["OmegaARout"]
```

Each input can either be a single scalar value or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values, with the excepton of the optional `totals`, `equilibria_in` and `equilibria_out` inputs, which should be dicts of scalars or arrays (if provided, see [Internal overrides](#internal-overrides)).  The output is a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) containing a series of NumPy arrays with all the calculated variables.  These are described in detail in the following sections.

### Using the Pythonic API

Alternatively, a more Pythonic API can be used to interface with `CO2SYS`.  This returns a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) in place of the dict, with the same names for the various outputs.  The function uses keyword arguments, meaning that only the specified marine carbonate system parameters have to be entered.

```python
from PyCO2SYS.api import CO2SYS_wrap as co2sys

# Call with defaults
df1 = co2sys(dic=2103, alk=2360)

# The above is equivalent to:
df1 = co2sys(
    dic=2103, alk=2360, pco2=None, fco2=None, pH=None,
    carb=None, bicarb=None, co2aq=None,
    temp_in=25, temp_out=25, pres_in=0, pres_out=0,
    sal=35, si=0, po4=0, nh3=0, h2s=0,
    K1K2_constants=4, KSO4_constants=1, KF_constant=1, pHscale_in=1,
    buffers_mode=1, verbose=True)
```

!!! warning "`CO2SYS_wrap`: incomplete functionality"
    
    In the main `PyCO2SYS.CO2SYS` function, each input row of `PAR1` and `PAR2` can contain a different combination of parameter types.  This is not currently possible with `PyCO2SYS.api.CO2SYS_wrap`: each call to the function may only have a single input pair combination, with the others all set to `None`.

    `CO2SYS_wrap` also does not support the `totals`, `equilibria_in` and `equilibria_out` optional inputs to `CO2SYS`.

This wrapper function will also accept NumPy arrays, pandas.Series or xarray.DataArrays as inputs.  Scalar or default values will be broadcast to match any vector inputs.

## Inputs

Most of the inputs should be familiar to previous users of CO2SYS for MATLAB, and they work exactly the same here.  Each input can either be a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.  If arrays are used then they must all be the same size as each other, but a combination of same-size arrays and single scalar values is allowed.

!!! inputs "`PyCO2SYS.CO2SYS` inputs"
    #### Carbonate system parameters

    * `PAR1` and `PAR2`: values of two different carbonate system parameters.
    * `PAR1TYPE` and `PAR2TYPE`: which types of parameter `PAR1` and `PAR2` are.

    These can be any pair of:

    * **Total alkalinity** (type `1`) in μmol·kg<sup>−1</sup>.
    * **Dissolved inorganic carbon** (type `2`) in μmol·kg<sup>−1</sup>.
    * **pH** (type `3`) on the Total, Seawater, Free or NBS scale[^1].  Which scale is given by the input `pHSCALEIN`.
    * **Partial pressure** (type `4`) or **fugacity** (type `5`) **of CO<sub>2</sub>** in μatm or **aqueous CO<sub>2</sub>** (type `8`) in μmol·kg<sup>−1</sup>.
    * **Carbonate ion** (type `6`) in μmol·kg<sup>−1</sup>.
    * **Bicarbonate ion** (type `7`) in μmol·kg<sup>−1</sup>.

    For all inputs in μmol·kg<sup>−1</sup>, the "kg" refers to the total solution, not H<sub>2</sub>O.  These are therefore most accurately termed *molinity* values (as opposed to *concentration* or *molality*).

    #### Hydrographic conditions

    * `SAL`: **practical salinity** (dimensionless).
    * `TEMPIN`: **temperature** at which `PAR1` and `PAR2` inputs are provided in °C.
    * `TEMPOUT`: **temperature** at which output results will be calculated in °C.
    * `PRESIN`: **pressure** at which `PAR1` and `PAR2` inputs are provided in dbar.
    * `PRESOUT`: **pressure** at which output results will be calculated in dbar.

    For example, if a sample was collected at 1000 dbar pressure (~1 km depth) at an in situ water temperature of 2.5 °C and subsequently measured in a lab at 25 °C, then the correct values would be `TEMPIN = 25`, `TEMPOUT = 2.5`, `PRESIN = 0`, and `PRESIN = 1000`.

    #### Nutrients and other solutes

    *Required:*

    * `SI`: **total silicate** in μmol·kg<sup>−1</sup>.
    * `PO4`: **total phosphate** in μmol·kg<sup>−1</sup>.

    *Optional (these default to zero if not specified):*

    * `NH3`: **total ammonia** in μmol·kg<sup>−1</sup>.
    * `H2S`: **total hydrogen sulfide** in μmol·kg<sup>−1</sup>.

    Again, the "kg" in μmol·kg<sup>−1</sup> refers to the total solution, not H<sub>2</sub>O. These are therefore most accurately termed *molinity* values (as opposed to *concentration* or *molality*).

    #### Settings

    *Required:*

    * `pHSCALEIN`: which **pH scale** was used for any pH entries in `PAR1` or `PAR2`, as defined by [ZW01](../refs/#z):
        * `1`: Total, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$.
        * `2`: Seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$.
        * `3`: Free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$.
        * `4`: NBS, i.e. relative to [NBS/NIST](https://www.nist.gov/history/nist-100-foundations-progress/nbs-nist) reference standards.

    * `K1K2CONSTANTS`: which set of equilibrium constants to use to model **carbonic acid dissociation:**
        * `1`: [RRV93](../refs/#r) (0 < *T* < 45 °C, 5 < *S* < 45, Total scale, artificial seawater).
        * `2`: [GP89](../refs/#g) (−1 < *T* < 40 °C, 10 < *S* < 50, Seawater scale, artificial seawater).
        * `3`: [H73a](../refs/#h) and [H73b](../refs/#h) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `4`: [MCHP73](../refs/#m) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `5`: [H73a](../refs/#h), [H73b](../refs/#h) and [MCHP73](../refs/#m) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `6`: [MCHP73](../refs/#m) aka "GEOSECS" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `7`: [MCHP73](../refs/#m) without certain species aka "Peng" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `8`: [M79](../refs/#m) (0 < *T* < 50 °C, *S* = 0, freshwater only).
        * `9`: [CW98](../refs/#c) (2 < *T* < 35 °C, 0 < *S* < 49, NBS scale, real and artificial seawater).
        * `10`: [LDK00](../refs/#l) (2 < *T* < 35 °C, 19 < *S* < 43, Total scale, real seawater).
        * `11`: [MM02](../refs/#m) (0 < *T* < 45 °C, 5 < *S* < 42, Seawater scale, real seawater).
        * `12`: [MPL02](../refs/#m) (−1.6 < *T* < 35 °C, 34 < *S* < 37, Seawater scale, field measurements).
        * `13`: [MGH06](../refs/#m) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).
        * `14`: [M10](../refs/#m) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).
        * `15`: [WMW14](../refs/#w) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).
        * `16`: [SLH20](../refs/#s)  (−1.67 < *T* < 31.80 °C, 30.73 < *S* < 37.57, Total scale, field measurements).
        * `17`: [SB21](../refs/#s) (15 < *T* < 35 °C, 19.6 < *S* < 41, Total scale).

    The brackets above show the valid temperature (*T*) and salinity (*S*) ranges, original pH scale, and type of material measured to derive each set of constants.

    * `KSO4CONSTANTS`: (1) which equilibrium constant to use to model **bisulfate ion dissociation** and (2) which **boron:salinity** relationship to use to estimate total borate:

        * `1`: [D90a](../refs/#d) for bisulfate dissociation and [U74](../refs/#u) for borate:salinity.
        * `2`: [KRCB77](../refs/#k) for bisulfate dissociation and [U74](../refs/#u) for borate:salinity.
        * `3`: [D90a](../refs/#d) for bisulfate dissociation and [LKB10](../refs/#l) for borate:salinity.
        * `4`: [KRCB77](../refs/#k) for bisulfate dissociation and [LKB10](../refs/#l) for borate:salinity.

    The somewhat inelegant approach for combining these into a single option is inherited from CO2SYS for MATLAB and retained here for consistency.

    *Optional:*

    * `KFCONSTANT`: which equilibrium constant to use for **hydrogen fluoride dissociation:**
        * `1`: [DR79](../refs/#d) (default, consistent with CO2SYS for MATLAB).
        * `2`: [PF87](../refs/#p).

    * `buffers_mode`: how to calculate the various buffer factors (or not).
        * `1`: using automatic differentiation, which accounts for the effects of all equilibrating solutes (default).
        * `2`: using explicit equations reported in the literature, which only account for carbonate, borate and water alkalinity.
        * `0`: not at all.

    For `buffers_mode`, `1` is the recommended and most accurate calculation, and it is a little faster to compute than `2`.  If `0` is selected, then the corresponding outputs have the value `nan`.

    * `WhichR`: what value to use for the ideal gas constant *R*:
        * `1`: DOEv2 (default, consistent with all previous CO2SYS software).
        * `2`: DOEv3.
        * `3`: [2018 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?r).

    #### Internal overrides

    You can optionally use the `totals`, `equilibria_in` and `equilibria_out` inputs to override some or all parameter values that PyCO2SYS normally estimates internally from salinity, temperature and pressure.  If used, these inputs should each be a dict containing one or more of the following items.

      * `totals`: any of the output variables listed below in [Totals estimated from salinity](#totals-estimated-from-salinity) in μmol·kg<sup>−1</sup>.
      * `equilibria_in`: any of the output variables listed below in [Equilibrium constants](#equilibrium-constants), the [fugacity factor](#dissolved-inorganic-carbon) and/or the [activitiy coefficient of H<sup>+</sup>](#ph-and-water), all at input conditions and with the word `input` removed from the end of each dict key.
      * `equilibria_out`: like `equilibria_in`, but for the output conditions.

    Like all other `PyCO2SYS.CO2SYS` input parameters, each field in these dicts can be either a single value or a NumPy array the same size as all other input arrays.

## Outputs

The results of `CO2SYS` calculations are stored in a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) of [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html). The keys to the dict are the same as the entries in the output `HEADERS` in CO2SYS for MATLAB and are listed in the section below.

!!! outputs "`PyCO2SYS.CO2SYS` outputs"
    The only output is a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) of [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html). Its keys are as follows:

    #### Dissolved inorganic carbon

    * `"TCO2"`: **dissolved inorganic carbon** in μmol·kg<sup>−1</sup>.
    * `"CO3in"`/`"CO3out"`: **carbonate ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"HCO3in"`/`"HCO3out"`: **bicarbonate ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"CO2in"`/`"CO2out"`: **aqueous CO<sub>2</sub>** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"pCO2in"`/`"pCO2out"`: **seawater partial pressure of CO<sub>2</sub>** at input/output conditions in μatm.
    * `"fCO2in"`/`"fCO2out"`: **seawater fugacity of CO<sub>2</sub>** at input/output conditions in μatm.
    * `"xCO2in"`/`"xCO2out"`: **seawater mole fraction of CO<sub>2</sub>** at input/output conditions in ppm.
    * `"FugFacinput"`/`"FugFacoutput"`: **fugacity factor** at input/output conditions for converting between CO<sub>2</sub> partial pressure and fugacity.

    #### Alkalinity and its components

    * `"TAlk"`: **total alkalinity** in μmol·kg<sup>−1</sup>.
    * `"BAlkin"`/`"BAlkout"`: **borate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"PAlkin"`/`"PAlkout"`: **phosphate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"SiAlkin"`/`"SiAlkout"`: **silicate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"NH3Alkin"`/`"NH3Alkout"`: **ammonia alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"H2SAlkin"`/`"H2SAlkout"`: **hydrogen sulfide alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"PengCorrection"`: the **"Peng correction"** for alkalinity (applies only for `K1K2CONSTANTS = 7`) in μmol·kg<sup>−1</sup>.

    #### pH and water

    * `"pHin"`/`"pHout"`: **pH** at input/output conditions on the scale specified by input `pHSCALEIN`.
    * `"pHinTOTAL"`/`"pHoutTOTAL"`: **pH** at input/output conditions on the **Total scale**.
    * `"pHinSWS"`/`"pHoutSWS"`: **pH** at input/output conditions on the **Seawater scale**.
    * `"pHinFREE"`/`"pHoutFREE"`: **pH** at input/output conditions on the **Free scale**.
    * `"pHinNBS"`/`"pHoutNBS"`: **pH** at input/output conditions on the **NBS scale**.
    * `"HFreein"`/`"HFreeout"`: **"free" proton** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"OHin"`/`"OHout"`: **hydroxide ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"fHinput"`/`"fHoutput"`: **activity coefficient of H<sup>+</sup>** at input/output conditions for pH-scale conversions to and from the NBS scale.

    #### Carbonate mineral saturation

    * `"OmegaCAin"`/`"OmegaCAout"`: **saturation state of calcite** at input/output conditions.
    * `"OmegaARin"`/`"OmegaARout"`: **saturation state of aragonite** at input/output conditions.

    #### Buffer factors

    Whether these are evaluated using automatic differentiation, with explicit equations, or not at all is controlled by the input `buffers_mode`.

    * `"RFin"`/`"RFout"`: **Revelle factor** at input/output conditions[^2].
    * `"psi_in"`/`"psi_out"`: *ψ* of [FCG94](../refs/#f) at input/output conditions.
    * `"gammaTCin"`/`"gammaTCout"`: **buffer factor *γ*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions[^3].
    * `"betaTCin"`/`"betaTCout"`: **buffer factor *β*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"omegaTCin"`/`"omegaTCout"`: **buffer factor *ω*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"gammaTAin"`/`"gammaTAout"`: **buffer factor *γ*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"betaTAin"`/`"betaTAout"`: **buffer factor *β*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"omegaTAin"`/`"omegaTAout"`: **buffer factor *ω*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"isoQin"`/`"isoQout"`: **isocapnic quotient** of [HDW18](../refs/#h) at input/output conditions.
    * `"isoQapprox_in"`/`"isoQapprox_out"`: **approximate isocapnic quotient** of [HDW18](../refs/#h) at input/output conditions.

    #### Biological properties

    Seawater properties related to the marine carbonate system that have a primarily biological application.

    * `"SIRin"`/`"SIRout"`: **substrate:inhibitor ratio** of [B15](../refs/#b) at input/output conditions in mol(HCO<sub>3</sub><sup>−</sup>)·μmol(H<sup>+</sup>)<sup>−1</sup>.

    #### Totals estimated from salinity

    * `"TB"`: **total borate** in μmol·kg<sup>−1</sup>.
    * `"TF"`: **total fluoride** μmol·kg<sup>−1</sup>.
    * `"TSO4"`: **total sulfate** in μmol·kg<sup>−1</sup> (or `"TS"`, deprecated).
    * `"TCa"`: **total calcium** in μmol·kg<sup>−1</sup>.

    #### Equilibrium constants

    All equilibrium constants are returned on the pH scale of input `pHSCALEIN` except for `"KFinput"`/`"KFoutput"` and `"KSO4input"`/`"KSO4output"`, which are always on the Free scale.

    * `"K0input"`/`"K0output"`: **Henry's constant for CO<sub>2</sub>** at input/output conditions.
    * `"K1input"`/`"K1output"`: **first carbonic acid** dissociation constant at input/output conditions.
    * `"K2input"`/`"K2output"`: **second carbonic acid** dissociation constant at input/output conditions.
    * `"pK1input"`/`"pK1output"`: **-log<sub>10</sub>(`K1input`)**/**-log<sub>10</sub>(`K1output`)**.
    * `"pK2input"`/`"pK2output"`: **-log<sub>10</sub>(`K2input`)**/**-log<sub>10</sub>(`K2output`)**.
    * `"KWinput"`/`"KWoutput"`: **water** dissociation constant at input/output conditions.
    * `"KBinput"`/`"KBoutput"`: **boric acid** dissociation constant at input/output conditions.
    * `"KFinput"`/`"KFoutput"`: **hydrogen fluoride** dissociation constant at input/output conditions.
    * `"KSO4input"`/`"KSO4output"`: **bisulfate** dissociation constant at input/output conditions (or `"KSinput"`/`"KSoutput"`, deprecated).
    * `"KP1input"`/`"KP1output"`: **first phosphoric acid** dissociation constant at input/output conditions.
    * `"KP2input"`/`"KP2output"`: **second phosphoric acid** dissociation constant at input/output conditions.
    * `"KP3input"`/`"KP3output"`: **third phosphoric acid** dissociation constant at input/output conditions.
    * `"KSiinput"`/`"KSioutput"`: **silicic acid** dissociation constant at input/output conditions.
    * `"KNH3input"`/`"KNH3output"`: **ammonium** equilibrium constant at input/output conditions.
    * `"KH2Sinput"`/`"KH2Soutput"`: **hydrogen sulfide** equilibrium constant at input/output conditions.
    * `"KCainput"`/`"KCaoutput"`: **calcite solubility product** at input/output conditions.
    * `"KArinput"`/`"KARoutput"`: **aragonite solubility product** at input/output conditions.

    The ideal gas constant used in the calculations is also returned.  Note the unusual unit:

    * `"RGas"`: **ideal gas constant** in ml·bar<sup>−1</sup>·mol<sup>−1</sup>·K<sup>−1</sup>.

    #### Function inputs

    * `"PAR1"`/`"PAR2"`: inputs `PAR1`/`PAR2`.
    * `"TEMPIN"`/`"TEMPOUT"`: inputs `TEMPIN`/`TEMPOUT`.
    * `"PRESIN"`/`"PRESOUT"`: inputs `PRESIN`/`PRESOUT`.
    * `"PAR1TYPE"`/`"PAR2TYPE"`: inputs `PAR1TYPE`/`PAR2TYPE`.
    * `"K1K2CONSTANTS"`: input `K1K2CONSTANTS`.
    * `"KSO4CONSTANTS"`: input `KSO4CONSTANTS`.
    * `"KFCONSTANT"`: input `KFCONSTANT`.
    * `"pHSCALEIN"`: input `pHSCALEIN`.
    * `"SAL"`: input `SAL`.
    * `"PO4"`: input `PO4`.
    * `"SI"`: input `SI`.
    * `"NH3"`: input `NH3`.
    * `"H2S"`: input `H2S`.
    * `"WhichR"`: input `WhichR`.

    Finally, `CO2SYS` splits up the input `KSO4CONSTANTS` into two separate settings variables internally, which are also returned in the `CO2dict` output:

    * `"KSO4CONSTANT"`:
        * `1` for `KSO4CONSTANTS in [1, 3]` (i.e. bisulfate dissociation from [D90a](../refs/#d)).
        * `2` for `KSO4CONSTANTS in [2, 4]` (i.e. bisulfate dissociation from [KRCB77](../refs/#k)).
    * `"BORON"`:
        * `1` for `KSO4CONSTANTS in [1, 2]` (i.e. borate:salinity from [U74](../refs/#u)).
        * `2` for `KSO4CONSTANTS in [3, 4]` (i.e. borate:salinity from [LKB10](../refs/#l)).


[^1]: See [ZW01](../refs/#z) for definitions of the different pH scales.

[^2]: In `buffers_mode=2`, the Revelle factor is calculated using a simple finite difference scheme, just like the MATLAB version of CO2SYS.

[^3]: Equations for the buffer factors of [ESM10](../refs/#e) in `buffers_mode=2` have all been corrected for typos following [RAH18](../refs/#r) and [OEDG18](../refs/#o).

## The original CO2SYS clone

Originally, the main `CO2SYS` function in PyCO2SYS was an as-close-as-possible clone of CO2SYS v2.0.5 for MATLAB ([from here](https://github.com/jamesorr/CO2SYS-MATLAB)). Since then the code has been substantially reorganised and made more Pythonic behind the scenes and it is this Pythonised version that is now called up by `from PyCO2SYS import CO2SYS`.

!!! warning
    We strongly recommend that you use a Pythonised version [e.g. described above](#syntax)!

If you do need to use the as-close-as-possible clone instead, this is still available via:

    :::python
    # Import the original CO2SYS clone
    from PyCO2SYS.original import CO2SYS

    # Run CO2SYS
    DATA, HEADERS, NICEHEADERS = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE,
        SAL, TEMPIN, TEMPOUT, PRESIN, PRESOUT, SI, PO4,
        pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS)

The inputs are the same [as described above](#inputs) except:

  * `PAR1TYPE` and `PAR2TYPE` can only take values from `1` to `5` inclusive.
  * The optional extra inputs (`NH3`, `H2S` and `KFCONSTANT`) are not allowed. This is equivalent to using `NH3 = 0`, `H2S = 0` and `KFCONSTANT = 1` in the Pythonic version.

The outputs are also the same [as described above](#outputs), except:

  * There are no buffer factors other than the Revelle factor.
  * The Revelle factor at output conditions does not include the "Peng correction" (applicable only for `K1K2CONSTANTS = 7`).
  * The `KSO4CONSTANTS` input is not split into `KSO4CONSTANT` and `BORON`.
  * There are none of the outputs associated with the `NH3` and `H2S` equilibria.
  * `TCa` is not provided.
  * They are reported in the original MATLAB style:
    *  `DATA` contains a matrix of all calculated values.
    *  `HEADERS` indicate the variable in each column of `DATA`.
    *  `NICEHEADERS` is an alternative to `HEADERS` containing a little more information about each variable.

To convert these MATLAB-style outputs into a dict comparable to `CO2dict`:

    :::python
    CO2dict = {header: DATA[:, h] for h, header in enumerate(HEADERS)}
