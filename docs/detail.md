# Arguments and results

This page provides a comprehensive overview of all possible arguments to PyCO2SYS and the results it can compute.  It assumes familiarity with the basic syntax of `pyco2.sys` (see the [Quick-start guide](quick.md)).

!!! info "Content, not concentration"
    For all arguments and results in μmol&nbsp;kg<sup>–1</sup>, the "kg" refers to the total solution, not H<sub>2</sub>O.  These are therefore accurately termed *substance content* or *molinity* values (as opposed to *concentration* or *molality*).

## Keyword arguments

Each argument to `pyco2.sys` can be either a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.  A combination of different multidimensional array shapes and sizes is allowed as long as they can all be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) with each other.

### Carbonate system parameters

Up to two carbonate system parameters can be provided.

!!! inputs "Carbonate system parameters"

    If two parameters are provided, these can be any pair of:

    * `alkalinity`: **total alkalinity** in μmol&nbsp;kg<sup>–1</sup>.
    * `dic`: **dissolved inorganic carbon** in μmol&nbsp;kg<sup>–1</sup>.
    * `pH`: **pH** on the total, seawater, free or NBS scale.  Which scale is given by `opt_pH_scale`.
    * `HCO3`: **bicarbonate ion** in μmol&nbsp;kg<sup>–1</sup>.
    * Any one of:
        * `pCO2`: **partial pressure of CO<sub>2</sub>** in μatm,
        * `fCO2`: **fugacity of CO<sub>2</sub>** in μatm,
        * `CO2`: **aqueous CO<sub>2</sub>** in μmol&nbsp;kg<sup>–1</sup>, or
        * `xCO2`: **dry-air mole fraction of CO<sub>2</sub>** in ppm.
    * Any one of:
        * `CO3`: **carbonate ion** in μmol&nbsp;kg<sup>–1</sup>,
        * `saturation_calcite`: **saturation state with respect to calcite**, or
        * `saturation_aragonite`: **saturation state with respect to aragonite**.

If one parameter is provided, then the full marine carbonate system cannot be solved, but some results can be calculated.  The single parameter can be any of:

  * `pCO2`, `fCO2`, `CO2` or `xCO2`: the others in this group of parameters can be calculated and adjusted to different temperatures.
  * `pH`, which can be converted to different scales.

If no carbonate system parameters are provided, then all equilibrium constants and total salt contents can still be calculated.

### Hydrographic conditions

If not provided, these revert to default values.

!!! inputs "Hydrographic conditions"

    * `salinity`: **practical salinity** (default 35).
    * `temperature`: **temperature** in °C (default 25 °C) at which the carbonate system parameters are provided.
    * `pressure`: **hydrostatic pressure** in dbar (default 0 dbar) at which the carbonate system parameters are provided.
    * `pressure_atmosphere`: **atmospheric pressure** in atm (default 1 atm).

### Nutrients and other solutes

Nutrients default to zero if not provided, while other solutes are calculated from salinity.

!!! inputs "Nutrients and other solutes"

    Some default to zero if not provided:

    * `total_silicate`: **total silicate** in μmol&nbsp;kg<sup>–1</sup> (default 0 μmol&nbsp;kg<sup>–1</sup>) ($[\mathrm{Si(OH)}_4] + [\mathrm{SiO(OH)_3}^-]$).
    * `total_phosphate`: **total phosphate** in μmol&nbsp;kg<sup>–1</sup> (default 0 μmol&nbsp;kg<sup>–1</sup>) ($[\mathrm{H}_3\mathrm{PO}_4] + [\mathrm{H}_2\mathrm{PO}_4^-] + [\mathrm{HPO}_4^{2-}] + [\mathrm{PO}_4^{3-}]$).
    * `total_ammonia`: **total ammonia** in μmol&nbsp;kg<sup>–1</sup> (default 0 μmol&nbsp;kg<sup>–1</sup>) ($[\mathrm{NH}_3] + [\mathrm{NH}_4^+]$).
    * `total_sulfide`: **total hydrogen sulfide** in μmol&nbsp;kg<sup>–1</sup> (default 0 μmol&nbsp;kg<sup>–1</sup>) ($[\mathrm{H}_2\mathrm{S}] + [\mathrm{HS}^-]$).
    * `total_nitrite`: **total nitrite** in μmol&nbsp;kg<sup>–1</sup> (default 0 μmol&nbsp;kg<sup>–1</sup>) ($[\mathrm{HNO}_2] + [\mathrm{NO}_2^-]$).

    Others are calculated from salinity if not provided:

    * `total_borate`: **total borate** in μmol&nbsp;kg<sup>–1</sup> ($[\mathrm{B(OH)}_3] + [\mathrm{B(OH)}_4^-]$).
    * `total_fluoride`: **total fluoride** in μmol&nbsp;kg<sup>–1</sup> ($[\mathrm{HF}] + [\mathrm{F}^-]$).
    * `total_sulfate`: **total sulfate** in μmol&nbsp;kg<sup>–1</sup> ($[\mathrm{HSO}_4^-] + [\mathrm{SO}_4^2-]$).
    * `Ca`: **dissolved calcium** in μmol&nbsp;kg<sup>–1</sup> ($[\mathrm{Ca}^{2+}]$).

    If these are provided then their [parameterisation settings](#settings) are ignored.

### Settings

#### pH scale

If `pH` is provided as an known marine carbonate system parameter, the pH scale that it is reported on can be specified with `opt_pH_scale`.  All equilibrium constants will also be calculated on the same scale.

!!! inputs "pH scale"

    * `opt_pH_scale`: which **pH scale** was used for `pH`, as defined by [ZW01](refs.md/#z):
        * `1`: total (default), i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$.
        * `2`: seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$.
        * `3`: free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$.
        * `4`: NBS, i.e. relative to [NBS/NIST](https://www.nist.gov/history/nist-100-foundations-progress/nbs-nist) reference standards.
    
    * `opt_fH`: how the **hydrogen ion activity coefficient** is calculated, for conversions to/from the NBS scale:
        * **`1`: [TWB82](refs.md/#t) (default).**
        * `2`: [PTBO87](refs.md/#p), for GEOSECS compatibility.
        * `3`: the coefficient is set to 1, for freshwater.

#### Carbonic acid dissociation

!!! inputs "Carbonic acid dissociation"

    * `opt_k_carbonic`: which set of equilibrium constant parameterisations to use for **carbonic acid dissociation**.  The valid temperature (*T*) and salinity (*S*) ranges, original pH scale, and type of material measured to derive each set of constants are shown.
        * `1`: [RRV93](refs.md/#r) (0 < *T* < 45 °C, 5 < *S* < 45, total scale, artificial seawater).
        * `2`: [GP89](refs.md/#g) (−1 < *T* < 40 °C, 10 < *S* < 50, seawater scale, artificial seawater).
        * `3`: [H73a](refs.md/#h) and [H73b](refs.md/#h) refit by [DM87](refs.md/#d) (2 < *T* < 35 °C, 20 < *S* < 40, seawater scale, artificial seawater).
        * `4`: [MCHP73](refs.md/#m) refit by [DM87](refs.md/#d) (2 < *T* < 35 °C, 20 < *S* < 40, seawater scale, real seawater).
        * `5`: [H73a](refs.md/#h), [H73b](refs.md/#h) and [MCHP73](refs.md/#m) refit by [DM87](refs.md/#d) (2 < *T* < 35 °C, 20 < *S* < 40, seawater scale, artificial seawater).
        * `6`: [MCHP73](refs.md/#m) aka "GEOSECS" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `7`: [MCHP73](refs.md/#m) without certain species aka "Peng" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `8`: [M79](refs.md/#m) (0 < *T* < 50 °C, *S* = 0, freshwater only).
        * `9`: [CW98](refs.md/#c) (2 < *T* < 30 °C, 0 < *S* < 40, NBS scale, real estuarine seawater).
        * **`10`: [LDK00](refs.md/#l) (default) (2 < *T* < 35 °C, 19 < *S* < 43, total scale, real seawater).**
        * `11`: [MM02](refs.md/#m) (0 < *T* < 45 °C, 5 < *S* < 42, seawater scale, real seawater).
        * `12`: [MPL02](refs.md/#m) (−1.6 < *T* < 35 °C, 34 < *S* < 37, seawater scale, field measurements).
        * `13`: [MGH06](refs.md/#m) (0 < *T* < 50 °C, 1 < *S* < 50, seawater scale, real seawater).
        * `14`: [M10](refs.md/#m) (0 < *T* < 50 °C, 1 < *S* < 50, seawater scale, real seawater).
        * `15`: [WMW14](refs.md/#w) (0 < *T* < 45 °C, 0 < *S* < 45, seawater scale, real seawater).
        * `16`: [SLH20](refs.md/#s)  (−1.67 < *T* < 31.80 °C, 30.73 < *S* < 37.57, total scale, field measurements).
        * `17`: [SB21](refs.md/#s) (15 < *T* < 35 °C, 19.6 < *S* < 41, total scale, real seawater).
        * `18`: [PLR18](refs.md/#p) (–6 < *T* < 25 °C, 33 < *S* < 100, total scale, real seawater).

    * `opt_factor_k_H2CO3`: **first carbonic acid** dissociation constant **pressure correction**:
        * **`1`: [M95](refs.md/#m) (default).**
        * `2`: [EG70](refs.md/#e), for GEOSECS compatibility.
        * `3`: [M83](refs.md/#m), for freshwater.

    * `opt_factor_k_HCO3`: **second carbonic acid** dissociation constant** pressure correction**:
        * **`1`: [M95](refs.md/#m) (default).**
        * `2`: [EG70](refs.md/#e), for GEOSECS compatibility.
        * `3`: [M83](refs.md/#m), for freshwater.

#### Other dissociation constants

!!! inputs "Other dissociation constants"

    * `opt_k_HSO4`: which parameterisation to use to model **bisulfate dissociation**:

        * **`1`: [D90a](refs.md/#d) (default)**.
        * `2`: [KRCB77](refs.md/#k).
        * `3`: [WM13](refs.md/#w) with the corrections of [WMW14](refs.md/#w).

    * `opt_k_HF`: which parameterisation to use for **hydrogen fluoride dissociation:**
        * **`1`: [DR79](refs.md/#d) (default)**.
        * `2`: [PF87](refs.md/#p).

    * `opt_k_BOH3`: which parameterisation to use for **boric acid dissociation**:
        * **`1`: [D90b](refs.md/#d) (default).**
        * `2`: [LTB69](refs.md/#l), for GEOSECS compatibility.

    * `opt_k_phosphate`: which parameterisation to use for **phosphoric acid dissociation**:
        * **`1`: [YM95](refs.md/#y) (default).**
        * `2`: [KP67](refs.md/#k), for GEOSECS compatibility.

    * `opt_k_NH3`: which parameterisation to use for **ammonium dissociation**:
        * **`1`: [CW95](refs.md/#c) (default).**
        * `2`: [YM95](refs.md/#y).

    * `opt_k_Si`: which parameterisation to use for **silicate dissociation**:
        * **`1`: [YM95](refs.md/#y) (default).**
        * `2`: [SMB64](refs.md/#s), for GEOSECS compatibility.

    * `opt_k_calcite`: which parameterisation to use for the **saturation state with respect to **:
        * **`1`: [M83](refs.md/#m) (default).**
        * `2`: [I75](refs.md/#i), for GEOSECS compatibility.

    * `opt_k_aragonite`: which parameterisation to use for the **saturation state with respect to **:
        * **`1`: [M83](refs.md/#m) (default).**
        * `2`: [ICHP73](refs.md/#i), for GEOSECS compatibility.

    * `opt_k_H2O`: which parameterisation to use for **water dissociation**:
        * **`1`: [M95](refs.md/#m) (default).**
        * `2`: [M79](refs.md/#m), for GEOSECS compatibility.
        * `3`: [HO58](refs.md/#h) refit by [M79](refs.md/#m), for freshwater.
    
    * `opt_k_HNO2`: which parameterisation to use for **nitrous acid dissociation**:
        * **`1`: [BBWB24](refs.md/#b) for seawater (default).** 
        * `2`: [BBWB24](refs.md/#b) for freshwater.

#### Other dissociation constant pressure corrections

!!! inputs "Other dissociation constant pressure corrections"

    * `opt_factor_k_BOH3`: **boric acid** dissociation constant **pressure correction**:
        * **`1`: [M79](refs.md/#m) (default).**
        * `2`: [EG70](refs.md/#e), for GEOSECS compatibility.

    * `opt_factor_k_H2O`: **water** dissociation constant **pressure correction**:
        *  **`1`: [M95](refs.md/#m) (default).**
        *  `2`: [M83](refs.md/#m), for freshwater.

#### Total salt contents

These settings are ignored if [their values are provided as arguments](#nutrients-and-other-solutes).

!!! inputs "Total salt contents"

    * `opt_total_borate`: which **boron:salinity** relationship is used to calculate total borate (ignored if the `total_borate` argument is provided):
        * **`1`: [U74](refs.md/#u) (default)**.
        * `2`: [LKB10](refs.md/#l).
        * `3`: [KSK18](refs.md/#k), for the Baltic Sea.

    * `opt_Ca`: which **calcium:salinity** relationship is used to calculate dissolved calcium (ignored if the `Ca` argument is provided):
        * **`1`: [RT67](refs.md/#r) (default)**.
        * `2`: [C65](refs.md/#c), for GEOSECS compatibility.

#### Other settings

!!! inputs "Other settings"

    * `opt_gas_constant`: what value to use for the **universal gas constant** (`gas_constant`):
        * `1`: DOEv2 (consistent with other CO2SYS software before July 2020).
        * `2`: DOEv3.
        * **`3`: [2018 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?r) (default)**.

    * `opt_fugacity_factor`: how to convert between partial pressure and fugacity of CO<sub>2</sub> (`pCO2` and `fCO2`):
        * **`1`: using a fugacity factor (default)**.
        * `2`: assuming that partial pressure and fugacity are equal, for compatibility with GEOSECS.

    * `opt_HCO3_root`: if **DIC and bicarbonate ion** are the known carbonate system parameter pair, then there are two possible valid solutions (e.g., [HLSP22](refs.md/#h)):
        * `1`: find the low-pH solution.
        * **`2`: find the high-pH solution (default)**. 

    * `opt_fCO2_temperature`: how to calculate the **temperature-sensitivity of fCO<sub>2</sub>** (`upsilon`) when only one marine carbonate system parameter is known:
        * **`1`: [H24](refs.md/#h) parameterisation (default).**
        * `2`: [TOG93](refs.md/#t) linear fit.
        * `3`: [TOG93](refs.md/#t) quadratic fit.

### Equilibrium constants

All of the equilibrium constants needed by PyCO2SYS are calculated internally from temperature, salinity and pressure, and returned in the results.  However, values for any of these constants can be provided instead when calling `pyco2.sys`.  They should be provided on the pH scale indicated by [`opt_pH_scale`](#ph-scale).

To do this, the arguments should have the same keywords as the corresponding [results dict keys](#equilibrium-constants).  For example, to provide a custom water dissociation constant value of p<i>K</i> 14, use `k_H2O=1e-14`.

<!--If non-zero using `total_alpha` and/or `total_beta`, you should also supply the corresponding stoichiometric dissociation constant values as `k_alpha`/`k_alpha_out` and/or `k_beta`/`k_beta_out`.  If not provided, these default to p*K* = 7.-->

## Results

See [Advanced results access](results.md) for a more detailed overview of the different ways that the results keys in the sections below can be solved for and accessed from a `CO2System`.

### `pyco2.sys` arguments

All [keyword arguments](#keyword-arguments) that can be provided to `pyco2.sys` and are not settings (i.e., do not begin with `opt_`) are also available as results with the same keyword.

Settings arguments can be found at `co2s.opts`.  They should not be modified there - doing so will have unpredictable consequences for future calculations.

### pH

!!! Outputs "pH"

    * `pH`: **pH** on the scale specified by `opt_pH_scale`.
    * `pH_total`: pH on the **total scale**.
    * `pH_sws`: pH on the **seawater scale**.
    * `pH_free`: pH on the **free scale**.
    * `pH_nbs`: pH on the **NBS scale**.
    * `fH`: **activity coefficient of H<sup>+</sup>** for conversions to and from the NBS scale.

### Chemical speciation

!!! Outputs "Chemical speciation"

    * `H_free`: **"free" proton** in μmol&nbsp;kg<sup>–1</sup>.
    * `OH`: **hydroxide ion** in μmol&nbsp;kg<sup>–1</sup>.
    * `CO3`: **carbonate ion** in μmol&nbsp;kg<sup>–1</sup>.
    * `HCO3`: **bicarbonate ion** in μmol&nbsp;kg<sup>–1</sup>.
    * `CO2`: **aqueous CO<sub>2</sub>** in μmol&nbsp;kg<sup>–1</sup>.
    * `BOH4`: **tetrahydroxyborate** $[\text{B(OH)}_4^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `BOH3`: **boric acid** $[\text{B(OH)}_3]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `H3PO4`: **phosphoric acid** $[\text{H}_3\text{PO}_4]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `H2PO4`: **dihydrogen phosphate** $[\text{H}_2\text{PO}_4^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `HPO4`: **monohydrogen phosphate** $[\text{HPO}_4^{2-}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `PO4`: **phosphate** $[\text{PO}_4^{3-}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `H4SiO4`: **orthosilicic acid** $[\text{Si(OH)}_4]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `H3SiO4`: **trihydrogen orthosilicate** $[\text{SiO(OH)}_3^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `NH3`: **ammonia** $[\text{NH}_3]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `NH4`: **ammonium** $[\text{NH}_4^+]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `HS`: **bisulfide** $[\text{HS}^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `H2S`: **hydrogen sulfide** $[\text{H}_2\text{S}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `HSO4`: **bisulfate** $[\text{HSO}_4^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `SO4`: **sulfate** $[\text{SO}_4^{2-}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `HF`: **hydrofluoric acid** $[\text{HF}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `F`: **fluoride** $[\text{F}^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `HNO2`: **nitrous acid** $[\text{HNO}_2]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `NO2`: **nitrite** $[\text{NO}_2^-]$ in μmol&nbsp;kg<sup>–1</sup>.


### Carbonate mineral saturation

!!! outputs "Carbonate mineral saturation"

    * `saturation_calcite`: **saturation state** with respect to **calcite**.
    * `saturation_aragonite`: **saturation state** with respect to **aragonite**.

### Chemical buffer factors

Buffer factors are evaluated using automatic differentiation of the complete alkalinity equation.

!!! outputs "Buffer factors"

    * `revelle_factor`: **Revelle factor**.
    * `psi`: *ψ* of [FCG94](refs.md/#f).
    * `gamma_dic`: **buffer factor *γ*<sub>DIC</sub>** of [ESM10](refs.md/#e).
    * `beta_dic`: **buffer factor *β*<sub>DIC</sub>** of [ESM10](refs.md/#e).
    * `omega_dic`: **buffer factor *ω*<sub>DIC</sub>** of [ESM10](refs.md/#e).
    * `gamma_alkalinity`: **buffer factor *γ*<sub>TA</sub>** of [ESM10](refs.md/#e).
    * `beta_alkalinity`: **buffer factor *β*<sub>TA</sub>** of [ESM10](refs.md/#e).
    * `omega_alkalinity`: **buffer factor *ω*<sub>TA</sub>** of [ESM10](refs.md/#e).
    * `Q_isocap`: **isocapnic quotient** of [HDW18](refs.md/#h).
    * `Q_isocap_approx`: **isocapnic quotient approximation** of [HDW18](refs.md/#h).
    * `dlnfCO2_dT`: **temperature derivative** of **ln(fCO<sub>2</sub>)**.
    * `dlnpCO2_dT`: **temperature derivative** of **ln(fCO<sub>2</sub>)**.
    * `substrate_inhibitor_ratio`: **substrate:inhibitor ratio** of [B15](refs.md/#b) in mol(HCO<sub>3</sub><sup>−</sup>)·μmol(H<sup>+</sup>)<sup>−1</sup>.
     
### Equilibrium constants

All equilibrium constants are returned on the pH scale of `opt_pH_scale` except for `k_HF_free` and `k_HSO4_free`, which are always on the free scale.  They are all stoichiometric constants, i.e., defined in terms of reactant contents rather than activities, with the exception of `k_CO2`, which is a hybrid constant.

!!! outputs "Equilibrium constants"

    * `k_CO2`: **Henry's constant for CO<sub>2</sub>**.
    * `k_H2CO3`: **first carbonic acid** dissociation constant.
    * `k_HCO3`: **second carbonic acid** dissociation constant.
    * `k_H2O`: **water** dissociation constant.
    * `k_BOH3`: **boric acid** dissociation constant.
    * `k_HF_free`: **hydrogen fluoride** dissociation constant.
    * `k_HSO4_free`: **bisulfate** dissociation constant.
    * `k_H3PO4`: **first phosphoric acid** dissociation constant.
    * `k_H2PO4`: **second phosphoric acid** dissociation constant.
    * `k_HPO4`: **third phosphoric acid** dissociation constant.
    * `k_Si`: **silicic acid** dissociation constant.
    * `k_NH3`: **ammonia** equilibrium constant.
    * `k_H2S`: **hydrogen sulfide** dissociation constant.
    * `k_HNO2`: **nitrous acid** dissociation constant.
    <!--* `k_alpha`: **HA** equilibrium constant.-->
    <!--* `k_beta`: **HB** equilibrium constant.-->

### Other results

!!! outputs "Other results"

    * `fugacity_factor`: **fugacity factor** for converting between CO<sub>2</sub> partial pressure and fugacity.
    * `vp_factor`: **vapour pressure factor** for converting between <i>x</i>CO<sub>2</sub> and <i>p</i>CO<sub>2</sub>.    
    * `gas_constant`: **ideal gas constant** in ml bar<sup>−1</sup> mol<sup>−1</sup> K<sup>−1</sup> (note the unusual unit).
