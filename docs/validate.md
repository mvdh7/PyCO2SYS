# Should you trust PyCO2SYS?

There are no "certified" results of marine carbonate system calculations against which software like PyCO2SYS can be unambiguously validated.  But we can evaluate its performance by testing its internal consistency and by comparing its outputs with those from other programs.

## Internal consistency

### Round robin

PyCO2SYS can solve the marine carbonate system from any valid pair of total alkalinity, dissolved inorganic carbon, pH, partial pressure or fugacity or molinity of aqueous CO<sub>2</sub>, bicarbonate ion molinity, and carbonate ion molinity.  In a "round robin" test, we first determine all of these core variables from one given input pair, and then solve the system again from the results using every possible combination of pairs as the input.  We expect to find exactly the same results from every input pair combination.

#### Do it yourself

We can conveniently run a round-robin test with PyCO2SYS for any given set of input conditions using `PyCO2SYS.test.roundrobin` ([script file available here](https://github.com/mvdh7/PyCO2SYS/blob/master/examples/round-robin.py)):

    :::python
    # Import PyCO2SYS
    import PyCO2SYS as pyco2

    # Define test conditions
    par1 = 2300  # parameter 1, here total alkalinity in μmol/kg-sw
    par2 = 8.1  # parameter 2, here pH on the Total scale
    par1type = 1  # "parameter 1 is total alkalinity"
    par2type = 3  # "parameter 2 is pH"
    sal = 33  # practical salinity
    temp = 22  # temperature in °C
    pres = 1000  # pressure in dbar
    si = 10  # total silicate in μmol/kg-sw
    phos = 1  # total phosphate in μmol/kg-sw
    nh3 = 2  # total ammonia in μmol/kg-sw
    h2s = 3  # total sulfide in μmol/kg-sw
    pHscale = 1  # "input pH is on the Total scale"
    k1k2c = 10  # "use LDK00 constants for carbonic acid dissociation"
    kso4c = 3  # "use D90a for bisulfate dissociation & LKB10 for borate:sal"

    # Run the test
    res, diff = pyco2.test.roundrobin(par1, par2, par1type, par2type,
        sal, temp, pres, si, phos, pHscale, k1k2c, kso4c, NH3=nh3, H2S=h2s,
        buffers_mode="none")

#### Results

With PyCO2SYS v1.3.0, running the round-robin test with the inputs in the example code above gave the following maximum absolute differences across all input pair combinations:

<div style="text-align:center">
<!-- HTML for table generated with examples/round-robin.py -->
<table>
<tr><th style="text-align:right">Carbonate system parameter</th><th style="text-align:center">Mean result</th><th style="text-align:center">Max. abs. diff.</th></tr>
<tr><td style="text-align:right">Total alkalinity / μmol/kg-sw</td><td style="text-align:center">2300.0</td><td style="text-align:center">9.09·10<sup>−13</sup></td></tr>
<tr><td style="text-align:right">Dissolved inorganic carbon / μmol/kg-sw</td><td style="text-align:center">1982.2</td><td style="text-align:center">4.55·10<sup>−13</sup></td></tr>
<tr><td style="text-align:right">pH (Total scale)</td><td style="text-align:center">8.1</td><td style="text-align:center">1.78·10<sup>−15</sup></td></tr>
<tr><td style="text-align:right"><i>p</i>CO<sub>2</sub> / μatm</td><td style="text-align:center">312.0</td><td style="text-align:center">1.53·10<sup>−12</sup></td></tr>
<tr><td style="text-align:right"><i>f</i>CO<sub>2</sub> / μatm</td><td style="text-align:center">311.0</td><td style="text-align:center">1.48·10<sup>−12</sup></td></tr>
<tr><td style="text-align:right">Carbonate ion / μmol/kg-sw</td><td style="text-align:center">218.1</td><td style="text-align:center">8.53·10<sup>−14</sup></td></tr>
<tr><td style="text-align:right">Bicarbonate ion / μmol/kg-sw</td><td style="text-align:center">1754.5</td><td style="text-align:center">6.82·10<sup>−13</sup></td></tr>
<tr><td style="text-align:right">Aqueous CO<sub>2</sub> / μmol/kg-sw</td><td style="text-align:center">9.6</td><td style="text-align:center">4.80·10<sup>−14</sup></td></tr>
</table>
</div>

The maximum absolute differences across all the different input pair combinations are negligible in this example, all at least ten orders of magnitude smaller than the accuracy with which any of these variables can be measured.  The differences are not exactly zero because the iterative pH solvers stop once a certain tolerance threshold is reached.  By default, this threshold is set at 10<sup>−8</sup> (in pH units) in PyCO2SYS.

### Buffer factors

**Automatic vs explicit!**

## External comparisons

### CO<sub>2</sub>SYS for MATLAB

The code for PyCO2SYS was originally based on [CO<sub>2</sub>SYS for MATLAB, version 2.0.5](https://github.com/jamesorr/CO2SYS-MATLAB/releases/tag/v2.0.5).  We should therefore expect that the results of these two programs will agree with each other perfectly, or that differences should be negligible for calculations where PyCO2SYS has since adjusted its calculation approach.

One advantage of comparing with this version of CO<sub>2</sub>SYS is that that MATLAB program has itself been rigorously compared with a suite of similar software packages that have been implemented in several different coding languages by [OEG15](../refs/#o).  In this intercomparison, CO<sub>2</sub>SYS was highly regarded:

!!! quote "[OEG15](../refs/#o) on CO<sub>2</sub>SYS"
    To compare packages, it was necessary to define a common reference.  Although check values exist for most of the equilibrium constants (Dickson et al., 2007), none are available for  computed variables.  Hence we chose CO2SYS as a relative reference for three reasons: (1) it was the first publicly available package; (2) its core routines already serve as the base code for two other packages (CO2calc and ODV); and (3) its documentation and code reveal the intense effort that its developers have put into ferreting out the right coefficients from the literature and the most appropriate version of formulations for the constants.

If the differences between PyCO2SYS and this CO<sub>2</sub>SYS for MATLAB version can be shown to be negligible, then the conclusions drawn about CO<sub>2</sub>SYS for MATLAB by [OEG15](../refs/#o) could arguably be considered to apply to PyCO2SYS too.

However, PyCO2SYS now calculates a wider array of properties than CO<sub>2</sub>SYS for MATLAB, and it has more inputs and options, so not everything can be tested this way.
