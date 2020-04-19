# Calculate everything with `CO2SYS`

## Syntax

The simplest way to use PyCO2SYS is to follow the approach of previous versions of CO<sub>2</sub>SYS and calculate every possible variable of interest at once. We can do this using the top-level `CO2SYS` function:

    :::python
    from PyCO2SYS import CO2SYS
    CO2dict = CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
        PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
        NH3=0.0, H2S=0.0, KFCONSTANT=1)

Each input can either be a single scalar value or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values. The output is a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) containing a series of NumPy arrays with all the calculated variables. These are described in detail in the following sections.

## Inputs

Most of the inputs should be familiar to previous users of CO<sub>2</sub>SYS for MATLAB, and they work exactly the same here. Each input can either be a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values. If arrays are used then they must all be the same size as each other, but a combination of same-size arrays and single scalar values is allowed.

!!! info "`PyCO2SYS.CO2SYS` inputs"
    ***Carbonate system parameters***

    * `PAR1` and `PAR2`: values of two different carbonate system parameters.
    * `PAR1TYPE` and `PAR2TYPE`: which types of parameter `PAR1` and `PAR2` are.

    These can be any pair of:

    * **Total alkalinity** (type `1`) in μmol·kg<sup>−1</sup>.
    * **Dissolved inorganic carbon** (type `2`) in μmol·kg<sup>−1</sup>.
    * **pH** (type `3`) on the Total, Seawater, Free or NBS scale[^1]. Which scale is given by the input `pHSCALEIN`.
    * **Partial pressure** (type `4`) or **fugacity** (type `5`) **of CO<sub>2</sub>** in μatm.
    * **Carbonate ion** (type `6`) in μmol·kg<sup>−1</sup>

    For all inputs in μmol·kg<sup>−1</sup>, the "kg" refers to the total solution, not H<sub>2</sub>O. These are therefore most accurately termed *molinity* values (as opposed to *concentration* or *molality*).

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

    Again, the "kg" in μmol·kg<sup>−1</sup> refers to the total solution, not H<sub>2</sub>O. These are therefore most accurately termed *molinity* values (as opposed to *concentration* or *molality*).

    ---

    ***Settings***

    *Required:*

    * `pHSCALEIN`: which pH scale was used for any pH entries in `PAR1` or `PAR2`, as defined by [ZW01](../refs/#z):
        * `1`: Total, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$.
        * `2`: Seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$.
        * `3`: Free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$.
        * `4`: NBS, i.e. relative to [NBS/NIST](https://www.nist.gov/history/nist-100-foundations-progress/nbs-nist) reference standards.

    * `K1K2CONSTANTS`: which set of equilibrium constants to use to model carbonic acid dissociation:
        * `1`: [RRV93](../refs/#r) (0 < *T* < 45 °C, 5 < *S* < 45, Total scale, artificial seawater).
        * `2`: [GP89](../refs/#g) (-1 < *T* < 40 °C, 10 < *S* < 50, Seawater scale, artificial seawater).
        * `3`: [H73a](../refs/#h) and [H73b](../refs/#h) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `4`: [MCHP73](../refs/#m) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `5`: [H73a](../refs/#h), [H73b](../refs/#h) and [MCHP73](../refs/#m) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `6`: [MCHP73](../refs/#m) aka "GEOSECS" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `7`: [MCHP73](../refs/#m) without certain species aka "Peng" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `8`: [M79](../refs/#m) (0 < *T* < 50 °C, *S* = 0, freshwater only).
        * `9`: [CW98](../refs/#c) (2 < *T* < 35 °C, 0 < *S* < 49, NBS scale, real and artificial seawater).
        * `10`: [LDK00](../refs/#l) (2 < *T* < 35 °C, 19 < *S* < 43, Total scale, real seawater).
        * `11`: [MM02](../refs/#m) (0 < *T* < 45 °C, 5 < *S* < 42, Seawater scale, real seawater).
        * `12`: [MPL02](../refs/#m) (-1.6 < *T* < 35 °C, 34 < *S* < 37, Seawater scale, field measurements).
        * `13`: [MGH06](../refs/#m) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).
        * `14`: [M10](../refs/#m) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).
        * `15`: [WMW14](../refs/#w) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).

    The brackets above show the valid temperature (*T*) and salinity (*S*) ranges, original pH scale, and type of material measured to derive each set of constants.

    * `KSO4CONSTANTS`: which equilibrium constant to use to model bisulfate ion dissociation **and** which boron:salinity relationship to use to estimate total borate:

        * `1`: [D90a](../refs/#d) for bisulfate dissociation and [U74](../refs/#u) for borate:salinity.
        * `2`: [KRCB77](../refs/#k) for bisulfate dissociation and [U74](../refs/#u) for borate:salinity.
        * `3`: [D90a](../refs/#d) for bisulfate dissociation and [LKB10](../refs/#l) for borate:salinity.
        * `4`: [KRCB77](../refs/#k) for bisulfate dissociation and [LKB10](../refs/#l) for borate:salinity.

    The somewhat inelegant approach for combining these into a single option is inherited from CO<sub>2</sub>SYS for MATLAB and retained here for consistency.

    *Optional:*

    * `KFCONSTANT`: which equilibrium constant to use for hydrogen fluoride dissociation.
        * `1`: [DR79](../refs/#d) (default, consistent with CO<sub>2</sub>SYS for MATLAB).
        * `2`: [PF87](../refs/#p).

## Outputs

The results of `CO2SYS` calculations are stored in a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries). The keys to the dict are the same as the entries in the output `HEADERS` in CO<sub>2</sub>SYS for MATLAB and are listed in the section below.

As an example, to find the saturation state of aragonite under the output conditions (i.e. at `TEMPOUT` and `PRESOUT`):

    :::python
    OmegaARout = CO2dict['OmegaARout']

!!! abstract "`PyCO2SYS.CO2SYS` outputs"
    The only output is a dict. Its keys are as follows:

    ***Dissolved inorganic carbon***

    * `'TCO2'`: **dissolved inorganic carbon** in μmol·kg<sup>−1</sup>.
    * `'CO3in'`/`'CO3out'`: **carbonate ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'HCO3in'`/`'HCO3out'`: **bicarbonate ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'CO2in'`/`'CO2out'`: **aqueous CO<sub>2</sub>** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'pCO2in'`/`'pCO2out'`: **seawater partial pressure of CO<sub>2</sub>** at input/output conditions in μatm.
    * `'fCO2in'`/`'fCO2out'`: **seawater fugacity of CO<sub>2</sub>** at input/output conditions in μatm.
    * `'xCO2in'`/`'xCO2out'`: **seawater mole fraction of CO<sub>2</sub>** at input/output conditions in ppm.

    ***Alkalinity and its components***

    * `'TAlk'`: **total alkalinity** in μmol·kg<sup>−1</sup>.
    * `'BAlkin'`/`'BAlkout'`: **borate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'PAlkin'`/`'PAlkout'`: **phosphate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'SiAlkin'`/`'SiAlkout'`: **silicate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'NH3Alkin'`/`'NH3Alkout'`: **ammonia alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'H2SAlkin'`/`'H2SAlkout'`: **hydrogen sulfide alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.

    ***pH and water***

    * `'pHin'`/`'pHout'`: **pH** at input/output conditions on the scale specified by input `pHSCALEIN`.
    * `'pHinTOTAL'`/`'pHoutTOTAL'`: **pH** at input/output conditions on the **Total scale**.
    * `'pHinSWS'`/`'pHoutSWS'`: **pH** at input/output conditions on the **Seawater scale**.
    * `'pHinFREE'`/`'pHoutFREE'`: **pH** at input/output conditions on the **Free scale**.
    * `'pHinNBS'`/`'pHoutNBS'`: **pH** at input/output conditions on the **NBS scale**.
    * `'HFreein'`/`'HFreeout'`: **"free" proton** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `'OHin'`/`'OHout'`: **hydroxide ion** at input/output conditions in μmol·kg<sup>−1</sup>.

    ***Carbonate mineral saturation***

    * `'OmegaCAin'`/`'OmegaCAout'`: **saturation state of calcite** at input/output conditions.
    * `'OmegaARin'`/`'OmegaARout'`: **saturation state of aragonite** at input/output conditions.

    ***Buffer factors***

    * `'RFin'`/`'RFout'`: **Revelle factor** at input/output conditions.
    * `'psi_in'`/`'psi_out'`: *ψ* of [FCG94](../refs/#f) at input/output conditions.
    * `'gammaTCin'`/`'gammaTCout'`: **buffer factor *γ*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions[^2].
    * `'betaTCin'`/`'betaTCout'`: **buffer factor *β*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `'omegaTCin'`/`'omegaTCout'`: **buffer factor *ω*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `'gammaTAin'`/`'gammaTAout'`: **buffer factor *γ*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `'betaTAin'`/`'betaTAout'`: **buffer factor *β*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `'omegaTAin'`/`'omegaTAout'`: **buffer factor *ω*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `'isoQin'`/`'isoQout'`: **isocapnic quotient** of [HDW18](../refs/#h) at input/output conditions.
    * `'isoQapprox_in'`/`'isoQapprox_out'`: **approximate isocapnic quotient** of [HDW18](../refs/#h) at input/output conditions.

    ***Totals estimated from salinity***

    * `'TB'`: **total borate** in μmol·kg<sup>−1</sup>.
    * `'TF'`: **total fluoride** μmol·kg<sup>−1</sup>.
    * `'TS'`: **total sulfate** in μmol·kg<sup>−1</sup>.

    ***Equilibrium constants***

    * `'K0input'`/`'K0output'`: **Henry's constant for CO<sub>2</sub>** at input/output conditions.
    * `'K1input'`/`'K1output'`: **first carbonic acid** dissociation constant at input/output conditions.
    * `'K2input'`/`'K2output'`: **second carbonic acid** dissociation constant at input/output conditions.
    * `'pK1input'`/`'pK1output'`: **-log<sub>10</sub>(`K1input`)**/**-log<sub>10</sub>(`K1output`)**.
    * `'pK2input'`/`'pK2output'`: **-log<sub>10</sub>(`K2input`)**/**-log<sub>10</sub>(`K2output`)**.
    * `'KWinput'`/`'KWoutput'`: **water** dissociation constant at input/output conditions.
    * `'KBinput'`/`'KBoutput'`: **boric acid** dissociation constant at input/output conditions.
    * `'KFinput'`/`'KFoutput'`: **hydrogen fluoride** dissociation constant at input/output conditions.
    * `'KSinput'`/`'KSoutput'`: **bisulfate** dissociation constant at input/output conditions.
    * `'KP1input'`/`'KP1output'`: **first phosphoric acid** dissociation constant at input/output conditions.
    * `'KP2input'`/`'KP2output'`: **second phosphoric acid** dissociation constant at input/output conditions.
    * `'KP3input'`/`'KP3output'`: **third phosphoric acid** dissociation constant at input/output conditions.
    * `'KSiinput'`/`'KSioutput'`: **silicic acid** dissociation constant at input/output conditions.
    * `'KNH3input'`/`'KNH3output'`: **ammonium** equilibrium constant at input/output conditions.
    * `'KH2Sinput'`/`'KH2Soutput'`: **hydrogen sulfide** equilibrium constant at input/output conditions.

    ***Function inputs***

    * `'TEMPIN'`/`'TEMPOUT'`: inputs `TEMPIN`/`TEMPOUT`.
    * `'PRESIN'`/`'PRESOUT'`: inputs `PRESIN`/`PRESOUT`.
    * `'PAR1TYPE'`/`'PAR2TYPE'`: inputs `PAR1TYPE`/`PAR2TYPE`.
    * `'K1K2CONSTANTS'`: input `K1K2CONSTANTS`.
    * `'KSO4CONSTANTS'`: input `KSO4CONSTANTS`.
    * `'KFCONSTANT'`: input `KFCONSTANT`.
    * `'pHSCALEIN'`: input `pHSCALEIN`.
    * `'SAL'`: input `SAL`.
    * `'PO4'`: input `PO4`.
    * `'SI'`: input `SI`.
    * `'NH3'`: input `NH3`.
    * `'H2S'`: input `H2S`.

    Finally, `CO2SYS` splits up the input `KSO4CONSTANTS` into two separate settings variables internally, which are also returned in the `CO2dict` output:

    * `'KSO4CONSTANT'`:
        * `1` for `KSO4CONSTANTS in [1, 3]` (i.e. bisulfate dissociation from [D90a](../refs/#d))
        * `2` for `KSO4CONSTANTS in [2, 4]` (i.e. bisulfate dissociation from [KRCB77](../refs/#k)).
    * `'BORON'`:
        * `1` for `KSO4CONSTANTS in [1, 2]` (i.e. borate:salinity from [U74](../refs/#u)).
        * `2` for `KSO4CONSTANTS in [3, 4]` (i.e. borate:salinity from [LKB10](../refs/#l)).


[^1]: See [ZW01](../refs/#z) for definitions of the different pH scales.

[^2]: Equations for the buffer factors of [ESM10](../refs/#e) have all been corrected for typos following [RAH18](../refs/#r) and [OEDG18](../refs/#o).
