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
<!-- HTML for table generated with examples/round-robin.py --><table>
<tr><th style="text-align:right">Variable</th><th style="text-align:center">Mean result</th><th style="text-align:center">Max. abs. diff.</th></tr>
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

## External comparisons

### CO<sub>2</sub>SYS for MATLAB
