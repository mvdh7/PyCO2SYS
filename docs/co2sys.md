# Calculate everything with `CO2SYS`

The simplest way to use PyCO2SYS is to follow the approach of previous versions of CO<sub>2</sub>SYS and calculate every possible variable of interest at once. We can do this using the top-level `CO2SYS` function:

    :::python
    import PyCO2SYS as pyco2
    CO2dict = pyco2.CO2SYS(PAR1, PAR2, PAR1TYPE, PAR2TYPE, SAL, TEMPIN, TEMPOUT,
        PRESIN, PRESOUT, SI, PO4, pHSCALEIN, K1K2CONSTANTS, KSO4CONSTANTS,
        NH3=0.0, H2S=0.0, KFCONSTANT=1)

Most of the inputs should be familiar to previous users of CO<sub>2</sub>SYS for MATLAB, and they work exactly the same here. Each input can either be a single scalar value, or a [NumPy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) containing a series of values.

!!! info "`pyco2.CO2SYS` inputs"
    * `PAR1` and `PAR2`: values of two different carbonate system parameters.

    These

    * `PAR1TYPE` and `PAR2TYPE`: the types
