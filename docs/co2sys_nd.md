# Calculate everything with `CO2SYS_nd`

## Syntax

From v1.5.0, the recommended way to run PyCO2SYS is to calculate everything you need at once with the top-level `CO2SYS_nd` function.  The syntax is:

    :::python
    import PyCO2SYS as pyco2
    results = pyco2.CO2SYS_nd(par1, par2, par1_type, par2_type, **kwargs)

The simplest possible syntax above only requires values for two carbonate system parameters (`par1` and `par2`) and the types of these parameters (`par1_type` and `par2_type`).  Everything else is assigned default values.  To override the default values, add in the relevant `kwargs` from below.

If you wish to also calculate [uncertainties](../uncertainty), you should put the `kwargs` into a dict and splat this into `CO2SYS_nd` with `**` as shown above, as you will need to use it again later.

## Arguments

Each argument to `CO2SYS_nd` can either be a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.  A combination of different multidimensional array shapes and sizes is allowed as long as they can all be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) with each other.

!!! info "`PyCO2SYS.CO2SYS_nd` arguments"

    #### Carbonate system parameters

    * `par1` and `par2`: values of two different carbonate system parameters.
    * `par1_type` and `par2_type`: which types of parameter `par1` and `par2` are.

    These can be any pair of:

    * **Total alkalinity** (type `1`) in μmol·kg<sup>−1</sup>.
    * **Dissolved inorganic carbon** (type `2`) in μmol·kg<sup>−1</sup>.
    * **pH** (type `3`) on the Total, Seawater, Free or NBS scale[^1].  Which scale is given by the argument `opt_pH_scale`.
    * **Partial pressure** (type `4`) or **fugacity** (type `5`) **of CO<sub>2</sub>** in μatm or **aqueous CO<sub>2</sub>** (type `8`) in μmol·kg<sup>−1</sup>.
    * **Carbonate ion** (type `6`) in μmol·kg<sup>−1</sup>.
    * **Bicarbonate ion** (type `7`) in μmol·kg<sup>−1</sup>.

    For all arguments in μmol·kg<sup>−1</sup>, the "kg" refers to the total solution, not H<sub>2</sub>O.  These are therefore most accurately termed *molinity* or *amount content* values (as opposed to *concentration* or *molality*).

    #### Hydrographic conditions

    * `salinity`: **practical salinity** (dimensionless).
    * `temperature`: **temperature** at which `par1` and `par2` arguments are provided in °C (default 25 °C).
    * `pressure`: **water pressure** at which `par1` and `par2` arguments are provided in dbar (default 0 dbar).

    If you also want to calculate outputs at a different temperature and pressure from the original measurements, then you can also use:

    * `temperature_out`: **temperature** at which results will be calculated in °C ("output conditions").
    * `pressure_out`: **water pressure** at which results will be calculated in dbar ("output conditions").

    For example, if a sample was collected at 1000 dbar pressure (~1 km depth) at an in situ water temperature of 2.5 °C and subsequently measured in a lab at 25 °C, then the correct values would be `temperature=25`, `temperature_out=2.5`, `pressure=0`, and `pressure_out=1000`.

    If neither `temperature_out` nor `pressure_out` is provided, then calculations will only be performed at the conditions specified by `temperature` and `pressure`, and none of the results with keys ending with `_out` will be returned in the `CO2_results` dict.  If only one of `temperature_out` nor `pressure_out` is provided, then we assume that the other one has the same values for the input and output calculations.

    #### Nutrients and other solutes

    Some default to zero if not provided:

    * `total_silicate`: **total silicate** in μmol·kg<sup>−1</sup>.
    * `total_phosphate`: **total phosphate** in μmol·kg<sup>−1</sup>.
    * `total_ammonia`: **total ammonia** in μmol·kg<sup>−1</sup>.
    * `total_sulfide`: **total hydrogen sulfide** in μmol·kg<sup>−1</sup>.

    Others, PyCO2SYS estimates from salinity if not provided:

    * `total_borate`: **total borate** in μmol·kg<sup>−1</sup>.
    * `total_calcium`: **total calcium** in μmol·kg<sup>−1</sup>.
    * `total_fluoride`: **total fluoride** in μmol·kg<sup>−1</sup>.
    * `total_sulfate`: **total sulfate** in μmol·kg<sup>−1</sup>.

    If `total_borate` is provided, then the `opt_total_borate` argument is ignored.

    Again, the kg in μmol·kg<sup>−1</sup> refers to the total solution, not H<sub>2</sub>O.

    #### Settings

    * `opt_pH_scale`: which **pH scale** was used for any pH entries in `par1` or `par2`, as defined by [ZW01](../refs/#z):
        * `1`: Total, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$.
        * `2`: Seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$.
        * `3`: Free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$.
        * `4`: NBS, i.e. relative to [NBS/NIST](https://www.nist.gov/history/nist-100-foundations-progress/nbs-nist) reference standards.

    * `opt_k_carbonic`: which set of equilibrium constant parameterisations to use to model **carbonic acid dissociation:**
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

    * `opt_k_bisulfate`: which equilibrium constant parameterisation to use to model **bisulfate ion dissociation**:

        * `1`: [D90a](../refs/#d) (default).
        * `2`: [KRCB77](../refs/#k).

    * `opt_total_borate`: which **boron:salinity** relationship to use to estimate total borate (ignored if the `total_borate` argument is provided):

        * `1`: [U74](../refs/#u).
        * `2`: [LKB10](../refs/#l) (default).

    * `opt_k_fluoride`: which equilibrium constant parameterisation to use for **hydrogen fluoride dissociation:**
        * `1`: [DR79](../refs/#d) (default).
        * `2`: [PF87](../refs/#p).

    * `buffers_mode`: how to calculate the various **buffer factors** (or not).
        * `"auto"`: using automatic differentiation, which accounts for the effects of all equilibrating solutes (default).
        * `"explicit"`: using explicit equations reported in the literature, which only account for carbonate, borate and water alkalinity.
        * `"none"`: not at all.

    For `buffers_mode`, `"auto"` is the recommended and most accurate calculation, and it is a little faster to compute than `"explicit"`.  If `"none"` is selected, then the corresponding outputs have the value `nan`.

    * `opt_gas_constant`: what value to use for the **gas constant** (*R*):
        * `1`: DOEv2 (consistent with other CO2SYS software before July 2020).
        * `2`: DOEv3.
        * `3`: [2018 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?r) (default).

    #### Override equilibrium constants

    All the equilibrium constants needed by PyCO2SYS are estimated internally from temperature, salinity and pressure, and returned in the results.  However, you can also directly provide your own values for any of these constants instead.
    
    To do this, the arguments have the same keywords as the corresponding [results dict keys](#equilibrium-constants).  For example, to provide your own water dissociation constant value at input conditions of $10^{-14}$, use `k_water=1e-14`.

## Results

The results of `CO2SYS_nd` calculations are stored in a [dict](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) of [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).  The keys to the dict are listed in the section below.

Scalar arguments, and results that depend only on scalar arguments, will be returned as scalars in the dict.  Array-like arguments, and results that depend on them, will all be broadcast to the same consistent shape.

The keys ending with `_out` are only available if at least one of the `temperature_out` or `pressure_out` arguments was provided.

!!! abstract "`PyCO2SYS.CO2SYS_nd` results dict"

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
    * `"peng_correction"`: the **"Peng correction"** for alkalinity (applies only for `opt_k_carbonic = 7`) in μmol·kg<sup>−1</sup>.

    #### pH and water

    * `"pH"`/`"pH_out"`: **pH** at input/output conditions on the scale specified by input `opt_pH_scale`.
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
    * `"isocapnic_quotient"`/`"isocapnic_quotient_out"`: **isocapnic quotient** of [HDW18](../refs/#h) at input/output conditions.
    * `"isocapnic_quotient_approx"`/`"isocapnic_quotient_approx_out"`: **isocapnic quotient approximation** of [HDW18](../refs/#h) at input/output conditions.

    #### Biological properties

    Seawater properties related to the marine carbonate system that have a primarily biological application.

    * `"substrate_inhibitor_ratio"`/`"substrate_inhibitor_ratio_out"`: **substrate:inhibitor ratio** of [B15](../refs/#b) at input/output conditions in mol(HCO<sub>3</sub><sup>−</sup>)·μmol(H<sup>+</sup>)<sup>−1</sup>.

    #### Totals estimated from salinity

    * `"total_borate"`: **total borate** in μmol·kg<sup>−1</sup>.
    * `"total_fluoride"`: **total fluoride** μmol·kg<sup>−1</sup>.
    * `"total_sulfate"`: **total sulfate** in μmol·kg<sup>−1</sup>.
    * `"total_calcium"`: **total calcium** in μmol·kg<sup>−1</sup>.

    #### Equilibrium constants

    All equilibrium constants are returned on the pH scale of input `pHSCALEIN` except for `"KFinput"`/`"KFoutput"` and `"KSO4input"`/`"KSO4output"`, which are always on the Free scale.

    * `"k_CO2"`/`"k_CO2_out"`: **Henry's constant for CO<sub>2</sub>** at input/output conditions.
    * `"k_carbonic_1"`/`"k_carbonic_1_out"`: **first carbonic acid** dissociation constant at input/output conditions.
    * `"k_carbonic_2"`/`"k_carbonic_2_out"`: **second carbonic acid** dissociation constant at input/output conditions.
    * `"k_water"`/`"k_water_out"`: **water** dissociation constant at input/output conditions.
    * `"k_borate"`/`"k_borate_out"`: **boric acid** dissociation constant at input/output conditions.
    * `"k_fluoride"`/`"k_fluoride_out"`: **hydrogen fluoride** dissociation constant at input/output conditions.
    * `"k_bisulfate"`/`"k_bisulfate_out"`: **bisulfate** dissociation constant at input/output conditions.
    * `"k_phosphoric_1"`/`"k_phosphoric_1_out"`: **first phosphoric acid** dissociation constant at input/output conditions.
    * `"k_phosphoric_2"`/`"k_phosphoric_2_out"`: **second phosphoric acid** dissociation constant at input/output conditions.
    * `"k_phosphoric_3"`/`"k_phosphoric_3_out"`: **third phosphoric acid** dissociation constant at input/output conditions.
    * `"k_silicate"`/`"k_silicate_out"`: **silicic acid** dissociation constant at input/output conditions.
    * `"k_ammonia"`/`"k_ammonia_out"`: **ammonia** equilibrium constant at input/output conditions.
    * `"k_sulfide"`/`"k_sulfide_out"`: **hydrogen sulfide** equilibrium constant at input/output conditions.

    The ideal gas constant used in the calculations is also returned.  Note the unusual unit:

    * `"gas_constant"`: **ideal gas constant** in ml·bar<sup>−1</sup>·mol<sup>−1</sup>·K<sup>−1</sup>.

    #### Function arguments

    All the function arguments not already mentioned here are also returned as results with the same keys.

[^1]: See [ZW01](../refs/#z) for definitions of the different pH scales.

[^2]: In `buffers_mode='explicit'`, the Revelle factor is calculated using a simple finite difference scheme, just like the MATLAB version of CO2SYS.

[^3]: Equations for the buffer factors of [ESM10](../refs/#e) in `buffers_mode='explicit'` have all been corrected for typos following [RAH18](../refs/#r) and [OEDG18](../refs/#o).
