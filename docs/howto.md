# How to use PyCO2SYS

## Syntax

### Define the `CO2System`

```python
import PyCO2SYS as pyco2

co2s = pyco2.sys(**kwargs)
```

The result `co2s` is a `CO2System` object.  At this point, it contains only the input parameters given as the `kwargs`.  The following set of default values is assumed:

```python
# Default kwargs for pyco2.sys
kwargs = {
    # Parameter values
    "temperature": 25.0,  # °C
    "total_ammonia": 0.0,  # µmol/kg-sw
    "total_phosphate": 0.0,  # µmol/kg-sw
    "total_silicate": 0.0,  # µmol/kg-sw
    "total_sulfide": 0.0,  # µmol/kg-sw
    "salinity": 35.0,
    "pressure": 0.0,  # dbar
    "pressure_atmosphere": 1.0,  # atm
    # Settings
    "opt_gas_constant": 3,
    "opt_factor_k_BOH3": 1,
    "opt_factor_k_H2CO3": 1,
    "opt_factor_k_HCO3": 1,
    "opt_factor_k_H2O": 1,
    "opt_fH": 1,
    "opt_k_carbonic": 1,
    "opt_k_phosphate": 1,
    "opt_k_BOH3": 1,
    "opt_k_H2O": 1,
    "opt_k_HF": 1,
    "opt_k_HSO4": 1,
    "opt_k_NH3": 1,
    "opt_k_Si": 1,
    "opt_pH_scale": 1,
    "opt_total_borate": 1,
    "opt_Ca": 1,
    "opt_fugacity_factor": 1,
    "opt_HCO3_root": 2,
    "opt_k_calcite": 1,
    "opt_k_aragonite": 1,
    "opt_adjust_temperature": 1,
}
```

### Calculate new parameters

Attempting to access parameters with square brackets will cause them to be calculated and returned.  For example:

```python
import PyCO2SYS as pyco2

# Set up the CO2System
co2s = pyco2.sys(
    alkalinity=2250,
    pH=8.1,
    temperature=12.5,
    salinity=32.4,
    opt_k_carbonic=10,
)
# (If necessary) solve for and retrieve a calculated parameter
dic = co2s["dic"]
# Solve for and retrieve multiple parameters as a dict
params = co2s[["fCO2", "k_H2CO3"]]
```

All intermediate parameters used in calculating the requested parameter will also be stored in the `co2s`, so they can be accessed more quickly for future calculations.

### Propagate uncertainties

To propagate independent uncertainties through the calculations, use `propagate`:

```python
# Propagate uncertainties
co2s.propagate("dic", {"alkalinity": 2, "pH": 0.02})

# Access propagated uncertainties
dic_uncertainty = co2s.uncertainty["dic"]["total"]
dic_uncertainty_from_pH = co2s.uncertainty["dic"]["pH"]
```

In the example above, independent uncertainties in alkalinity of 2 µmol&nbsp;kg<sup>–1</sup> and pH of 0.02 were propagated through to DIC (`dic_uncertainty` in the standard units of DIC, i.e., µmol&nbsp;kg<sup>–1</sup>).  The individual components of uncertainty deriving from each source can also be accessed.

### Adjust conditions

To adjust the system to a different set of temperature and/or pressure conditions, use `adjust`:

```python
co2s_adj = co2s.adjust(temperature=25, pressure=1000)
```

The result `co2s_adj` is a new `CO2System` with all values at the new conditions (above, temperature of 25 °C and hydrostatic pressure of 1000 dbar).

The `adjust` method can be used if any two carbonate system parameters are known, but also if only one of pCO<sub>2</sub>, fCO<sub>2</sub>, [CO<sub>2</sub>(aq)] or *x*CO<sub>2</sub> is known.  In this case, `adjust` can take additional kwargs:

  * `method_fCO2`: how to do the temperature conversion.
    * `1`: using the parameterised <i>υ<sub>h</sub></i> equation of [H24](../refs/#h) (**default**). 
    * `2`: using the constant <i>υ<sub>h</sub></i> fitted to the [TOG93](../refs/#t) dataset by [H24](../refs/#h).
    * `3`: using the constant theoretical <i>υ<sub>x</sub></i> of [H24](../refs/#h).
    * `4`: following the [H24](../refs/#h) approach but using a user-provided $b_h$ value (given with the additional kwarg `bh_upsilon`).
    * `5`: using the linear fit of [TOG93](../refs/#t).
    * `6`: using the quadratic fit of [TOG93](../refs/#t) (default before v1.8.3).
  * `opt_which_fCO2_insitu`: whether the input (`1`, **default**) or output (`2`) condition pCO<sub>2</sub>, fCO<sub>2</sub>, [CO<sub>2</sub>(aq)] and/or <i>x</i>CO<sub>2</sub> values are at in situ conditions, for determining <i>b<sub>h</sub></i> with the parameterisation of [H24](../refs/#h).  Only applies when `method_fCO2` is `1`.

## Keyword arguments

Each argument to `pyco2.sys` can be either a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.  A combination of different multidimensional array shapes and sizes is allowed as long as they can all be [broadcasted](https://numpy.org/doc/stable/user/basics.broadcasting.html) with each other.

!!! inputs "`pyco2.sys` arguments (`kwargs`)"

    For all arguments and results in μmol&nbsp;kg<sup>–1</sup>, the "kg" refers to the total solution, not H<sub>2</sub>O.  These are therefore most accurately termed *substance content* or *molinity* values (as opposed to *concentration* or *molality*).

    #### Carbonate system parameters

    Up to two carbonate system parameters can be provided.

    If two parameters are provided, these can be any pair of:

    * `alkalinity`: **total alkalinity** in μmol&nbsp;kg<sup>–1</sup>.
    * `dic`: **dissolved inorganic carbon** in μmol&nbsp;kg<sup>–1</sup>.
    * `pH`: **pH** on the total, seawater, free or NBS scale[^1].  Which scale is given by `opt_pH_scale`.
    * Any one of:
        * `pCO2`: **partial pressure of CO<sub>2</sub>** in μatm,
        * `fCO2`: **fugacity of CO<sub>2</sub>** in μatm,
        * `CO2`: **aqueous CO<sub>2</sub>** in μmol&nbsp;kg<sup>–1</sup>, or
        * `xCO2`: **dry mole fraction of CO<sub>2</sub>** in ppm.
    * Any one of:
        * `CO3`: **carbonate ion** in μmol&nbsp;kg<sup>–1</sup>,
        * `saturation_calcite`: **saturation state with respect to calcite**, or
        * `saturation_aragonite`: **saturation state with respect to aragonite**.
    * `HCO3`: **bicarbonate ion** in μmol&nbsp;kg<sup>–1</sup>.

    If one parameter is provided, then the full marine carbonate system cannot be solved, but some results can be calculated.  The single parameter can be any of:

    * `pH`.  In this case, the pH values can be converted to all the other pH scales.
    * Any one of `pCO2`, `fCO2`, `CO2` or `xCO2`.  In this case, the others in this group of parameters can all be calculated.

    If no carbonate system parameters are provided, then all the equilibrium constants and total salt contents can be calculated.

    #### Hydrographic conditions

    * `salinity`: **practical salinity** (default 35).
    * `temperature`: **temperature** at which the carbonate system parameters are provided in °C (default 25 °C).
    * `pressure`: **hydrostatic pressure** at which the carbonate system parameters are provided in dbar (default 0 dbar).

    #### Nutrients and other solutes

    Some default to zero if not provided:

    * `total_silicate`: **total silicate** in μmol&nbsp;kg<sup>–1</sup>.
    * `total_phosphate`: **total phosphate** in μmol&nbsp;kg<sup>–1</sup>.
    * `total_ammonia`: **total ammonia** in μmol&nbsp;kg<sup>–1</sup>.
    * `total_sulfide`: **total hydrogen sulfide** in μmol&nbsp;kg<sup>–1</sup>.
    <!--* `total_alpha`: **total Hα** (a user-defined extra contributor to alkalinity) in μmol&nbsp;kg<sup>–1</sup>.-->
    <!--* `total_beta`: **total Hβ** (a user-defined extra contributor to alkalinity) in μmol&nbsp;kg<sup>–1</sup>.-->

    <!--If using non-zero `total_alpha` and/or `total_beta`, then you should also provide the corresponding equilibrium constant values `k_alpha` and/or `k_beta`.-->

    Others, PyCO2SYS estimates from salinity if not provided:

    * `total_borate`: **total borate** in μmol&nbsp;kg<sup>–1</sup>.
    * `Ca`: **total calcium** in μmol&nbsp;kg<sup>–1</sup>.
    * `total_fluoride`: **total fluoride** in μmol&nbsp;kg<sup>–1</sup>.
    * `total_sulfate`: **total sulfate** in μmol&nbsp;kg<sup>–1</sup>.

    If `total_borate` is provided, then the `opt_total_borate` argument is ignored.

    #### Atmospheric pressure

    * `pressure_atmosphere`: atmospheric pressure in atm (default 1 atm).

    This is used for conversions between *p*CO<sub>2</sub>, *f*CO<sub>2</sub> and *x*CO<sub>2</sub>.

    #### Settings

    * `opt_pH_scale`: which **pH scale** was used for `pH`, as defined by [ZW01](../refs/#z):
        * `1`: Total, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-])$ **(default)**.
        * `2`: Seawater, i.e. $\mathrm{pH} = -\log_{10} ([\mathrm{H}^+] + [\mathrm{HSO}_4^-] + [\mathrm{HF}])$.
        * `3`: Free, i.e. $\mathrm{pH} = -\log_{10} [\mathrm{H}^+]$.
        * `4`: NBS, i.e. relative to [NBS/NIST](https://www.nist.gov/history/nist-100-foundations-progress/nbs-nist) reference standards.

    * `opt_k_carbonic`: which set of equilibrium constant parameterisations to use to model **carbonic acid dissociation:**
        * `1`: [RRV93](../refs/#r) (0 < *T* < 45 °C, 5 < *S* < 45, Total scale, artificial seawater).
        * `2`: [GP89](../refs/#g) (−1 < *T* < 40 °C, 10 < *S* < 50, Seawater scale, artificial seawater).
        * `3`: [H73a](../refs/#h) and [H73b](../refs/#h) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `4`: [MCHP73](../refs/#m) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, real seawater).
        * `5`: [H73a](../refs/#h), [H73b](../refs/#h) and [MCHP73](../refs/#m) refit by [DM87](../refs/#d) (2 < *T* < 35 °C, 20 < *S* < 40, Seawater scale, artificial seawater).
        * `6`: [MCHP73](../refs/#m) aka "GEOSECS" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `7`: [MCHP73](../refs/#m) without certain species aka "Peng" (2 < *T* < 35 °C, 19 < *S* < 43, NBS scale, real seawater).
        * `8`: [M79](../refs/#m) (0 < *T* < 50 °C, *S* = 0, freshwater only).
        * `9`: [CW98](../refs/#c) (2 < *T* < 30 °C, 0 < *S* < 40, NBS scale, real estuarine seawater).
        * `10`: [LDK00](../refs/#l) (2 < *T* < 35 °C, 19 < *S* < 43, Total scale, real seawater) **(default)**.
        * `11`: [MM02](../refs/#m) (0 < *T* < 45 °C, 5 < *S* < 42, Seawater scale, real seawater).
        * `12`: [MPL02](../refs/#m) (−1.6 < *T* < 35 °C, 34 < *S* < 37, Seawater scale, field measurements).
        * `13`: [MGH06](../refs/#m) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).
        * `14`: [M10](../refs/#m) (0 < *T* < 50 °C, 1 < *S* < 50, Seawater scale, real seawater).
        * `15`: [WMW14](../refs/#w) (0 < *T* < 45 °C, 0 < *S* < 45, Seawater scale, real seawater).
        * `16`: [SLH20](../refs/#s)  (−1.67 < *T* < 31.80 °C, 30.73 < *S* < 37.57, Total scale, field measurements).
        * `17`: [SB21](../refs/#s) (15 < *T* < 35 °C, 19.6 < *S* < 41, Total scale, real seawater).
        * `18`: [PLR18](../refs/#p) (–6 < *T* < 25 °C, 33 < *S* < 100, Total scale, real seawater).

    The brackets above show the valid temperature (*T*) and salinity (*S*) ranges, original pH scale, and type of material measured to derive each set of constants.

    * `opt_k_bisulfate`: which equilibrium constant parameterisation to use to model **bisulfate ion dissociation**:

        * `1`: [D90a](../refs/#d) **(default)**.
        * `2`: [KRCB77](../refs/#k).
        * `3`: [WM13](../refs/#w)/[WMW14](../refs/#w).

    * `opt_total_borate`: which **boron:salinity** relationship to use to estimate total borate (ignored if the `total_borate` argument is provided):

        * `1`: [U74](../refs/#u) **(default)**.
        * `2`: [LKB10](../refs/#l).
        * `3`: [KSK18](../refs/#k).

    * `opt_k_fluoride`: which equilibrium constant parameterisation to use for **hydrogen fluoride dissociation:**
        * `1`: [DR79](../refs/#d) **(default)**.
        * `2`: [PF87](../refs/#p).

    * `opt_gas_constant`: what value to use for the **gas constant** (*R*):
        * `1`: DOEv2 (consistent with other CO2SYS software before July 2020).
        * `2`: DOEv3.
        * `3`: [2018 CODATA](https://physics.nist.gov/cgi-bin/cuu/Value?r) **(default)**.

    #### Override equilibrium constants

    All the equilibrium constants needed by PyCO2SYS are estimated internally from temperature, salinity and pressure, and returned in the results.  However, you can also directly provide your own values for any of these constants instead.

    To do this, the arguments have the same keywords as the corresponding [results dict keys](#equilibrium-constants).  For example, to provide your own water dissociation constant value at input conditions of $10^{-14}$, use `k_H2O=1e-14`.

    <!--If non-zero using `total_alpha` and/or `total_beta`, you should also supply the corresponding stoichiometric dissociation constant values as `k_alpha`/`k_alpha_out` and/or `k_beta`/`k_beta_out`.  If not provided, these default to p*K* = 7.-->

## Results

!!! outputs "`pyco2.sys` results"

    #### Dissolved inorganic carbon

    * `"dic"`: **dissolved inorganic carbon** in μmol&nbsp;kg<sup>–1</sup>.
    * `"pCO2"`: **seawater partial pressure of CO<sub>2</sub>** in μatm.
    * `"fCO2"`: **seawater fugacity of CO<sub>2</sub>** in μatm.
    * `"xCO2"`: **seawater dry mole fraction of CO<sub>2</sub>** in ppm.
    * `"fugacity_factor"`: **fugacity factor** for converting between CO<sub>2</sub> partial pressure and fugacity.
    * `"vp_factor"`: **vapour pressure factor** for converting between <i>x</i>CO<sub>2</sub> and <i>p</i>CO<sub>2</sub>.

    #### Alkalinity and chemical speciation

    * `"alkalinity"`: **total alkalinity** in μmol&nbsp;kg<sup>–1</sup>.
    * `"H_free"`: **"free" proton** in μmol&nbsp;kg<sup>–1</sup>.
    * `"OH"`: **hydroxide ion** in μmol&nbsp;kg<sup>–1</sup>.
    * `"CO3"`: **carbonate ion** in μmol&nbsp;kg<sup>–1</sup>.
    * `"HCO3"`: **bicarbonate ion** in μmol&nbsp;kg<sup>–1</sup>.
    * `"CO2"`: **aqueous CO<sub>2</sub>** in μmol&nbsp;kg<sup>–1</sup>.
    * `"BOH4"`: **tetrahydroxyborate** $[\text{B(OH)}_4^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"BOH3"`: **boric acid** $[\text{B(OH)}_3]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"H3PO4"`: **phosphoric acid** $[\text{H}_3\text{PO}_4]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"H2PO4"`: **dihydrogen phosphate** $[\text{H}_2\text{PO}_4^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"HPO4"`: **monohydrogen phosphate** $[\text{HPO}_4^{2-}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"PO4"`: **phosphate** $[\text{PO}_4^{3-}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"H4SiO4"`: **orthosilicic acid** $[\text{Si(OH)}_4]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"H3SiO4"`: **trihydrogen orthosilicate** $[\text{SiO(OH)}_3^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"NH3"`: **ammonia** $[\text{NH}_3]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"NH4"`: **ammonium** $[\text{NH}_4^+]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"HS"`: **bisulfide** $[\text{HS}^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"H2S"`: **hydrogen sulfide** $[\text{H}_2\text{S}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"HSO4"`: **bisulfate** $[\text{HSO}_4^-]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"SO4"`: **sulfate** $[\text{SO}_4^{2-}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"HF"`: **hydrofluoric acid** $[\text{HF}]$ in μmol&nbsp;kg<sup>–1</sup>.
    * `"F"`: **fluoride** $[\text{F}^-]$ in μmol&nbsp;kg<sup>–1</sup>.

    #### pH

    * `"pH"`: **pH** on the scale specified by `opt_pH_scale`.
    * `"pH_total"`: **pH** on the **total scale**.
    * `"pH_sws"`: **pH** on the **seawater scale**.
    * `"pH_free"`: **pH** on the **free scale**.
    * `"pH_nbs"`: **pH** on the **NBS scale**.
    * `"fH"`: **activity coefficient of H<sup>+</sup>** for conversions to and from the NBS scale.

    #### Carbonate mineral saturation

    * `"saturation_calcite"`: **saturation state** with respect to **calcite**.
    * `"saturation_aragonite"`: **saturation state** with respect to **aragonite**.

    #### Buffer factors

    These are all evaluated using automatic differentiation of the complete alkalinity equation.

    * `"revelle_factor"`: **Revelle factor**.
    * `"psi"`: *ψ* of [FCG94](../refs/#f).
    * `"gamma_dic"`: **buffer factor *γ*<sub>DIC</sub>** of [ESM10](../refs/#e).
    * `"beta_dic"`: **buffer factor *β*<sub>DIC</sub>** of [ESM10](../refs/#e).
    * `"omega_dic"`: **buffer factor *ω*<sub>DIC</sub>** of [ESM10](../refs/#e).
    * `"gamma_alkalinity"`: **buffer factor *γ*<sub>TA</sub>** of [ESM10](../refs/#e).
    * `"beta_alkalinity"`: **buffer factor *β*<sub>TA</sub>** of [ESM10](../refs/#e).
    * `"omega_alkalinity"`: **buffer factor *ω*<sub>TA</sub>** of [ESM10](../refs/#e).
    * `"Q_isocap"`: **isocapnic quotient** of [HDW18](../refs/#h).
    * `"Q_isocap_approx"`: **isocapnic quotient approximation** of [HDW18](../refs/#h).
    * `"dlnfCO2_dT"`: **temperature derivative of ln(<i>ƒ</i>CO<sub>2</sub>)** (see [TOG93](../refs/#t)).
    * `"dlnpCO2_dT"`: **temperature derivative of ln(<i>p</i>CO<sub>2</sub>)** (see [TOG93](../refs/#t)).

    #### Biological properties

    Seawater properties related to the marine carbonate system that have a primarily biological application.

    * `"substrate_inhibitor_ratio"`: **substrate:inhibitor ratio** of [B15](../refs/#b) in mol(HCO<sub>3</sub><sup>−</sup>)·μmol(H<sup>+</sup>)<sup>−1</sup>.

    #### Total salts

    * `"total_borate"`: **total borate** in μmol&nbsp;kg<sup>–1</sup>.
    * `"total_fluoride"`: **total fluoride** μmol&nbsp;kg<sup>–1</sup>.
    * `"total_sulfate"`: **total sulfate** in μmol&nbsp;kg<sup>–1</sup>.
    * `"Ca"`: **total calcium** in μmol&nbsp;kg<sup>–1</sup>.

    #### Equilibrium constants

    All equilibrium constants are returned on the pH scale of `opt_pH_scale` except for `"k_HF_free"` and `"k_HSO4_free"`, which are always on the free scale.

    * `"k_CO2"`: **Henry's constant for CO<sub>2</sub>**.
    * `"k_H2CO3"`: **first carbonic acid** dissociation constant.
    * `"k_HCO3"`: **second carbonic acid** dissociation constant.
    * `"k_H2O"`: **water** dissociation constant.
    * `"k_BOH3"`: **boric acid** dissociation constant.
    * `"k_HF_free"`: **hydrogen fluoride** dissociation constant.
    * `"k_HSO4_free"`: **bisulfate** dissociation constant.
    * `"k_H3PO4"`: **first phosphoric acid** dissociation constant.
    * `"k_H2PO4"`: **second phosphoric acid** dissociation constant.
    * `"k_HPO4"`: **third phosphoric acid** dissociation constant.
    * `"k_Si"`: **silicic acid** dissociation constant.
    * `"k_NH3"`: **ammonia** equilibrium constant.
    * `"k_H2S"`: **hydrogen sulfide** equilibrium constant.
    <!--* `"k_alpha"`: **HA** equilibrium constant.-->
    <!--* `"k_beta"`: **HB** equilibrium constant.-->

    The ideal gas constant used in the calculations is also returned.  Note the unusual unit:

    * `"gas_constant"`: **ideal gas constant** in ml·bar<sup>−1</sup>·mol<sup>−1</sup>·K<sup>−1</sup>.

    #### Function arguments

    All the function arguments not already mentioned here are also returned as results with the same keys.

[^1]: See [ZW01](../refs/#z) for definitions of the different pH scales.
