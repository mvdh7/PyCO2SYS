# Should you trust PyCO2SYS?

There are no "certified" results of marine carbonate system calculations against which software like PyCO2SYS can be unambiguously validated.  But we can evaluate its performance by testing its internal consistency and by comparing its outputs with those from other programs.

## Internal consistency

### Round robin

PyCO2SYS can solve the marine carbonate system from any valid pair of total alkalinity, dissolved inorganic carbon, pH, partial pressure or fugacity or molinity of aqueous CO<sub>2</sub>, bicarbonate ion molinity, and carbonate ion molinity.  In a "round robin" test, we first determine all of these core variables from one given input pair, and then solve the system again from the results using every possible combination of pairs as the input.  We expect to find exactly the same results from every input pair combination.

#### Do it yourself

We can conveniently run a round-robin test with PyCO2SYS for any given set of input conditions using `PyCO2SYS.test.roundrobin`:

    :::python
    # Import PyCO2SYS
    import PyCO2SYS as pyco2

    # Define test conditions
    par1 = 2300
    par2 = 2150
    par1type = 1
    par2type = 2
    sal = 33
    temp = 22
    pres = 1000
    si = 10
    phos = 1
    pHscale = 1
    k1k2 = 10
    kso4 = 4
    nh3 = 2
    h2s = 3

    # Run the test
    res, diff = pyco2.test.roundrobin(par1, par2, par1type, par2type,
        sal, temp, pres, si, phos, pHscale, k1k2, kso4, NH3=nh3, H2S=h2s)

#### Results

## External comparisons

### CO<sub>2</sub>SYS for MATLAB
