# Calculate everything with `CO2SYS`

## Syntax

### Simplest possible

From v1.5.0, the recommended way to run PyCO2SYS is to calculate everything you need at once with the top-level `CO2SYS_nd` function.  The simplest possible syntax is:

    :::python
    # Import the package
    import PyCO2SYS as pyco2

    # Define seawater conditions
    par1 = 2300  # total alkalinity in μmol/kg-sw
    par2 = 8.1  # pH on the Total scale
    par1_type = 1  # "par1 is a total alkalinity value"
    par2_type = 3  # "par2 is a pH value"

    # Run CO2SYS
    results = pyco2.CO2SYS_nd(par1, par2, par1_type, par2_type)

    # Get (e.g.) aragonite saturation state
    saturation_aragonite = results["saturation_aragonite"]

Each input can either be a single scalar value or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.  The output is a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) containing a series of NumPy arrays with all the calculated variables.

### Specify the pH scale



### Include environmental conditions

Many more optional keyword arguments can be provided to `CO2SYS_nd`.  The next level of control would be to also specify the environmental conditions and nutrient molinities, rather than just using the defaults (default values are shown in the examples here):

    :::python
    # Also specify the environmental conditions
    results = pyco2.CO2SYS_nd(par1, par2, par1_type, par2_type,
        salinity=35, temperature=25, pressure=0,
        total_ammonia=0, total_phosphate=0, total_silicate=0, total_sulfide=0)

### From lab to field: input and output conditions

Sometimes, the carbonate system parameters will be not have been measured at their in situ temperature and pressure.  In this case, we can solve the carbonate system both at the input (measurement) conditions and also at the output (in situ) conditions by also providing `temperature_out` and `pressure_out` arguments:

    :::python
    # Calculate at output conditions too
    results = pyco2.CO2SYS_nd(par1, par2, par1_type, par2_type,
        temperature=25, temperature_out=None, pressure=0, pressure_out=None)

    # Get (e.g.) aragonite saturation state
    saturation_aragonite_input = results["saturation_aragonite"]
    saturation_aragonite_output = results["saturation_aragonite_out"]

(The default `None`s should be replaced by the actual values.)

As shown above for the aragonite saturation state, each member of the `results` dict that is different under input and output conditions now has values under both sets of conditions.  The key to the result output conditions is always the same as that at input conditions plus the suffix `_out`.

### Control the constants

If you wish to vary which parameterisations of different equilibrium constants and the borate:chlorinity ratio are used, just add the relevant arguments:

    :::python
    # Try some different parameterisations
    results = pyco2.CO2SYS_nd(par1, par2, par1_type, par2_type,
        bisulfate_opt=1, borate_opt=1, carbonic_opt=16, fluoride_opt=1)

The full syntax is:

    :::python
    results = pyco2.CO2SYS_nd(
        # Compulsory arguments:
        par1, par2, par1_type, par2_type,
        # Common keyword arguments:
        salinity=35, temperature=25, temperature_out=None, pressure=0, pressure_out=None,
        total_ammonia=0, total_phosphate=0, total_silicate=0, total_sulfide=0,
        pH_scale_opt=1, bisulfate_opt=1, borate_opt=2, carbonic_opt=16,
        # Advanced keyword arguments:
        total_borate=None, total_calcium=None, total_fluoride=None, total_sulfate=None,
        fugacity_factor=None, fugacity_factor_out=None, gas_constant=None,
        gas_constant_out=None, k_ammonia=None, k_ammonia_out=None, k_borate=None,
        k_borate_out=None, k_bisulfate=None, k_bisulfate_out=None, k_carbon_dioxide=None,
        k_carbon_dioxide_out=None, k_carbonic_1=None, k_carbonic_1_out=None,
        k_carbonic_2=None, k_carbonic_2_out=None, k_fluoride=None, k_fluoride_out=None,
        k_phosphate_1=None, k_phosphate_1_out=None, k_phosphate_2=None,
        k_phosphate_2_out=None, k_phosphate_3=None, k_phosphate_3_out=None,
        k_silicate=None, k_silicate_out=None, k_sulfide=None, k_sulfide_out=None,
        k_water=None, k_water_out=None, fluoride_opt=1, gas_constant_opt=3,
        buffers_mode="auto")

## Inputs and outputs, arguments and results

## Arguments

Each argument can either be a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.  A combination of different array shapes and sizes is allowed as long as they can all be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) with each other.

!!! info "`PyCO2SYS.CO2SYS_nd` arguments"
    #### Carbonate system parameters

    * `par1` and `par2`: values of two different carbonate system parameters.
    * `par1_type` and `par2_type`: which types of parameter `par1` and `par2` are.

    These can be any pair of:

    * **Total alkalinity** (type `1`) in μmol·kg<sup>−1</sup>.
    * **Dissolved inorganic carbon** (type `2`) in μmol·kg<sup>−1</sup>.
    * **pH** (type `3`) on the Total, Seawater, Free or NBS scale[^1].  Which scale is given by the input `pHSCALEIN`.
    * **Partial pressure** (type `4`) or **fugacity** (type `5`) **of CO<sub>2</sub>** in μatm or **aqueous CO<sub>2</sub>** (type `8`) in μmol·kg<sup>−1</sup>.
    * **Carbonate ion** (type `6`) in μmol·kg<sup>−1</sup>.
    * **Bicarbonate ion** (type `7`) in μmol·kg<sup>−1</sup>.

    For all arguments in μmol·kg<sup>−1</sup>, the "kg" refers to the total solution, not H<sub>2</sub>O.  These are therefore most accurately termed *molinity* or *amount content* values (as opposed to *concentration* or *molality*).

    #### Hydrographic conditions

    * `salinity`: **practical salinity** (dimensionless).
    * `temperature`: **temperature** at which `par1` and `par2` arguments are provided in °C ("input conditions").
    * `temperature_out`: **temperature** at which results will be calculated in °C ("output conditions").
    * `pressure`: **pressure** at which `PAR1` and `PAR2` arguments are provided in dbar ("input conditions").
    * `pressure_out`: **pressure** at which results will be calculated in dbar ("output conditions").

    For example, if a sample was collected at 1000 dbar pressure (~1 km depth) at an in situ water temperature of 2.5 °C and subsequently measured in a lab at 25 °C, then the correct values would be `temperature = 25`, `temperature_out = 2.5`, `pressure = 0`, and `pressure_out = 1000`.

    If neither `temperature_out` nor `pressure_out` is provided, then calculations will only be performed at the conditions specified by `temperature` and `pressure`.  None of the results with keys ending with `_out` will be returned in the `CO2_results` dict.  If only one of `temperature_out` nor `pressure_out` is provided, then we assume that the other one has the same values for the input and output calculations.

    #### Nutrients and other solutes

    * `total_silicate`: **total silicate** in μmol·kg<sup>−1</sup>.
    * `total_phosphate`: **total phosphate** in μmol·kg<sup>−1</sup>.
    * `total_ammonia`: **total ammonia** in μmol·kg<sup>−1</sup>.
    * `total_sulfide`: **total hydrogen sulfide** in μmol·kg<sup>−1</sup>.

    Again, the "kg" in μmol·kg<sup>−1</sup> refers to the total solution, not H<sub>2</sub>O.

    #### Settings

    * `pH_scale_opt`: which **pH scale** was used for any pH entries in `par1` or `par2`, as defined by [ZW01](../refs/#z):
        * `1`: Total, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$.
        * `2`: Seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$.
        * `3`: Free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$.
        * `4`: NBS, i.e. relative to [NBS/NIST](https://www.nist.gov/history/nist-100-foundations-progress/nbs-nist) reference standards.

    * `carbonic_opt`: which set of equilibrium constants to use to model **carbonic acid dissociation:**
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

    The brackets above show the valid temperature (*T*) and salinity (*S*) ranges, original pH scale, and type of material measured to derive each set of constants.

    * `bisulfate_opt`: which equilibrium constant to use to model **bisulfate ion dissociation**:

        * `1`: [D90a](../refs/#d) (default).
        * `2`: [KRCB77](../refs/#k).

    * `borate_opt`: which **boron:salinity** relationship to use to estimate total borate:

        * `1`: [U74](../refs/#u) (default).
        * `2`: [LKB10](../refs/#l).

    * `fluoride_opt`: which equilibrium constant to use for **hydrogen fluoride dissociation:**
        * `1`: [DR79](../refs/#d) (default).
        * `2`: [PF87](../refs/#p).

    * `buffers_mode`: how to calculate the various buffer factors (or not).
        * `"auto"`: using automatic differentiation, which accounts for the effects of all equilibrating solutes (default).
        * `"explicit"`: using explicit equations reported in the literature, which only account for carbonate, borate and water alkalinity.
        * `"none"`: not at all.

    For `buffers_mode`, `"auto"` is the recommended and most accurate calculation, and it is a little faster to compute than `"explicit"`.  If `"none"` is selected, then the corresponding outputs have the value `nan`.

    * `gas_constant_opt`: what value to use for the ideal gas constant *R*:
        * `1`: DOEv2 (consistent with other CO2SYS software before July 2020).
        * `2`: DOEv3.
        * `3`: [2018 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?r) (default).

## Results

The results of `CO2SYS_nd` calculations are stored in a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) of [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).  The keys to the dict are listed in the section below.

!!! abstract "`PyCO2SYS.CO2SYS` results dict"
    The only output is a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) of [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).  Its keys are as follows:

    #### Dissolved inorganic carbon

    * `"dic"`: **dissolved inorganic carbon** in μmol·kg<sup>−1</sup>.
    * `"carbonate"`/`"carbonate_out"`: **carbonate ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"bicarbonate"`/`"bicarbonate_out"`: **bicarbonate ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"aqueous_CO2"`/`"aqueous_CO2_out"`: **aqueous CO<sub>2</sub>** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"pCO2"`/`"pCO2_out"`: **seawater partial pressure of CO<sub>2</sub>** at input/output conditions in μatm.
    * `"fCO2"`/`"fCO2_out"`: **seawater fugacity of CO<sub>2</sub>** at input/output conditions in μatm.
    * `"xCO2"`/`"xCO2_out"`: **seawater mole fraction of CO<sub>2</sub>** at input/output conditions in ppm.
    * `"fugacity_factor"`/`"fugacity_factor_out"`: **fugacity factor** at input/output conditions for converting between CO<sub>2</sub> partial pressure and fugacity.

    #### Alkalinity and its components

    * `"alkalinity"`: **total alkalinity** in μmol·kg<sup>−1</sup>.
    * `"alkalinity_borate"`/`"alkalinity_borate_out"`: **borate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"alkalinity_phosphate"`/`"alkalinity_phosphate_out"`: **phosphate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"alkalinity_silicate"`/`"alkalinity_silicate_out"`: **silicate alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"alkalinity_ammonia"`/`"alkalinity_ammonia_out"`: **ammonia alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"alkalinity_sulfide"`/`"alkalinity_sulfide_out"`: **hydrogen sulfide alkalinity** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"peng_correction"`: the **"Peng correction"** for alkalinity (applies only for `carbonic_opt = 7`) in μmol·kg<sup>−1</sup>.

    #### pH and water

    * `"pH"`/`"pH_out"`: **pH** at input/output conditions on the scale specified by input `pH_scale_opt`.
    * `"pH_total"`/`"pH_total_out"`: **pH** at input/output conditions on the **Total scale**.
    * `"pH_sws"`/`"pH_sws_out"`: **pH** at input/output conditions on the **Seawater scale**.
    * `"pH_free"`/`"pH_free_out"`: **pH** at input/output conditions on the **Free scale**.
    * `"pH_nbs"`/`"pH_nbs_out"`: **pH** at input/output conditions on the **NBS scale**.
    * `"HFreein"`/`"HFreeout"`: **"free" proton** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"hydroxide"`/`"hydroxide_out"`: **hydroxide ion** at input/output conditions in μmol·kg<sup>−1</sup>.
    * `"fH"`/`"fH_out"`: **activity coefficient of H<sup>+</sup>** at input/output conditions for pH-scale conversions to and from the NBS scale.

    #### Carbonate mineral saturation

    * `"saturation_calcite"`/`"saturation_calcite_out"`: **saturation state of calcite** at input/output conditions.
    * `"saturation_aragonite"`/`"saturation_aragonite_out"`: **saturation state of aragonite** at input/output conditions.

    #### Buffer factors

    Whether these are evaluated using automatic differentiation, with explicit equations, or not at all is controlled by the input `buffers_mode`.

    * `"revelle_factor"`/`"revelle_factor_out"`: **Revelle factor** at input/output conditions[^2].
    * `"psi"`/`"psi_out"`: *ψ* of [FCG94](../refs/#f) at input/output conditions.
    * `"gamma_dic"`/`"gamma_dic_out"`: **buffer factor *γ*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions[^3].
    * `"beta_dic"`/`"beta_dic_out"`: **buffer factor *β*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"omega_dic"`/`"omega_dic_out"`: **buffer factor *ω*<sub>DIC</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"gamma_alk"`/`"gamma_alk_out"`: **buffer factor *γ*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"beta_alk"`/`"beta_alk_out"`: **buffer factor *β*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"omega_alk"`/`"omega_alk_out"`: **buffer factor *ω*<sub>TA</sub>** of [ESM10](../refs/#e) at input/output conditions.
    * `"isoQin"`/`"isoQout"`: **isocapnic quotient** of [HDW18](../refs/#h) at input/output conditions.
    * `"isoQapprox_in"`/`"isoQapprox_out"`: **approximate isocapnic quotient** of [HDW18](../refs/#h) at input/output conditions.

    #### Biological properties

    Seawater properties related to the marine carbonate system that have a primarily biological application.

    * `"substrate_inhibitor_ratio"`/`"substrate_inhibitor_ratio_out"`: **substrate:inhibitor ratio** of [B15](../refs/#b) at input/output conditions in mol(HCO<sub>3</sub><sup>−</sup>)·μmol(H<sup>+</sup>)<sup>−1</sup>.

    #### Totals estimated from salinity

    * `"total_borate"`: **total borate** in μmol·kg<sup>−1</sup>.
    * `"total_fluoride"`: **total fluoride** μmol·kg<sup>−1</sup>.
    * `"total_sulfate"`: **total sulfate** in μmol·kg<sup>−1</sup> (or `"TS"`, deprecated).
    * `"total_calcium"`: **total calcium** in μmol·kg<sup>−1</sup>.

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
    * `"KSO4input"`/`"KSO4output"`: **bisulfate** dissociation constant at input/output conditions.
    * `"KP1input"`/`"KP1output"`: **first phosphoric acid** dissociation constant at input/output conditions.
    * `"KP2input"`/`"KP2output"`: **second phosphoric acid** dissociation constant at input/output conditions.
    * `"KP3input"`/`"KP3output"`: **third phosphoric acid** dissociation constant at input/output conditions.
    * `"KSiinput"`/`"KSioutput"`: **silicic acid** dissociation constant at input/output conditions.
    * `"KNH3input"`/`"KNH3output"`: **ammonium** equilibrium constant at input/output conditions.
    * `"KH2Sinput"`/`"KH2Soutput"`: **hydrogen sulfide** equilibrium constant at input/output conditions.

    The ideal gas constant used in the calculations is also returned.  Note the unusual unit:

    * `"gas_constant"`: **ideal gas constant** in ml·bar<sup>−1</sup>·mol<sup>−1</sup>·K<sup>−1</sup>.

    #### Function arguments

    All the function arguments not already mentioned here are also returned as results with the same keys.

[^1]: See [ZW01](../refs/#z) for definitions of the different pH scales.

[^2]: In `buffers_mode='explicit'`, the Revelle factor is calculated using a simple finite difference scheme, just like the MATLAB version of CO2SYS.

[^3]: Equations for the buffer factors of [ESM10](../refs/#e) in `buffers_mode='explicit'` have all been corrected for typos following [RAH18](../refs/#r) and [OEDG18](../refs/#o).
