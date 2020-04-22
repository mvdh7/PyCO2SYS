# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
from autograd.numpy import array, full, isin, nan, size, unique, where
from autograd.numpy import all as np_all
from autograd.numpy import any as np_any
from autograd.numpy import min as np_min
from autograd.numpy import max as np_max
from . import solve

def inputs(input_locals):
    """Condition inputs for use with CO2SYS (sub)functions."""
    # Determine and check lengths of input vectors
    veclengths = array([size(v) for v in input_locals.values()])
    assert size(unique(veclengths[veclengths != 1])) <= 1, \
        'CO2SYS function inputs must all be of same length, or of length 1.'
    # Make vectors of all inputs
    ntps = max(veclengths)
    args = {k: full(ntps, v) if size(v)==1 else v.ravel()
            for k, v in input_locals.items()}
    # Convert to float where appropriate
    float_vars = ['SAL', 'TEMPIN', 'TEMPOUT', 'PRESIN', 'PRESOUT', 'SI', 'PO4',
                  'NH3', 'H2S', 'PAR1', 'PAR2']
    for k in args.keys():
        if k in float_vars:
            args[k] = args[k].astype('float64')
    return args, ntps

def pair2core(par1, par2, par1type, par2type):
    """Expand `par1` and `par2` inputs into one array per core variable of the marine 
    carbonate system.
    """
    assert (
        size(par1) == size(par2) == size(par1type) == size(par2type)
    ), "`par1`, `par2`, `par1type` and `par2type` must all be the same size."
    ntps = size(par1)
    # Generate empty vectors for...
    TA = full(ntps, nan)  # total alkalinity
    TC = full(ntps, nan)  # DIC
    PH = full(ntps, nan)  # pH
    PC = full(ntps, nan)  # CO2 partial pressure
    FC = full(ntps, nan)  # CO2 fugacity
    CARB = full(ntps, nan)  # carbonate ions
    HCO3 = full(ntps, nan)  # bicarbonate ions
    CO2 = full(ntps, nan)  # aqueous CO2
    # Assign values to empty vectors and convert micro[mol|atm] to [mol|atm]
    TA = where(par1type == 1, par1 * 1e-6, TA)
    TC = where(par1type == 2, par1 * 1e-6, TC)
    PH = where(par1type == 3, par1, PH)
    PC = where(par1type == 4, par1 * 1e-6, PC)
    FC = where(par1type == 5, par1 * 1e-6, FC)
    CARB = where(par1type == 6, par1 * 1e-6, CARB)
    HCO3 = where(par1type == 7, par1 * 1e-6, HCO3)
    CO2 = where(par1type == 8, par1 * 1e-6, CO2)
    TA = where(par2type == 1, par2 * 1e-6, TA)
    TC = where(par2type == 2, par2 * 1e-6, TC)
    PH = where(par2type == 3, par2, PH)
    PC = where(par2type == 4, par2 * 1e-6, PC)
    FC = where(par2type == 5, par2 * 1e-6, FC)
    CARB = where(par2type == 6, par2 * 1e-6, CARB)
    HCO3 = where(par2type == 7, par2 * 1e-6, HCO3)
    CO2 = where(par2type == 8, par2 * 1e-6, CO2)
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2


def getIcase(par1type, par2type):
    """Generate vector describing the combination of input parameters.

    Options for `par1type` and `par2type`:

      * `1` = total alkalinity
      * `2` = dissolved inorganic carbon
      * `3` = pH
      * `4` = partial pressure of CO2
      * `5` = fugacity of CO2
      * `6` = carbonate ion
      * `7` = bicarbonate ion
      * `8` = aqueous CO2

    `Icase` is `10*parXtype + parYtype` where `parXtype` is whichever of `par1type` or 
    `par2type` is greater.

    Noting that a pair of any two from pCO2, fCO2 and CO2(aq) is not allowed, the valid 
    `Icase` options are therefore:

        12, 13, 14, 15, 16, 17, 18,
            23, 24, 25, 26, 27, 28,
                34, 35, 36, 37, 38,
                        46, 47,
                        56, 57,
                            67, 68,
                                78.
    """
    # Check validity of separate `par1type` and `par2type` inputs
    Iarr = array([par1type, par2type])
    assert np_all(
        isin(Iarr, [1, 2, 3, 4, 5, 6, 7, 8])
    ), "All `par1type` and `par2type` values must be integers from 1 to 8."
    assert ~np_any(
        par1type == par2type
    ), "`par1type` and `par2type` must be different from each other."
    # Combine inputs into `Icase` and check its validity
    Icase = 10 * np_min(Iarr, axis=0) + np_max(Iarr, axis=0)
    assert ~np_any(
        isin(Icase, [45, 48, 58])
    ), "Combinations of pCO2, fCO2 and CO2(aq) are not valid input pairs."
    return Icase


def solvecore(par1, par2, par1type, par2type, PengCx, totals, K0, FugFac, Ks):
    """Solve the core marine carbonate system (MCS) from any 2 of its variables.
    
    The core MCS outputs and associated `par1type`/`par2type` inputs are:
        
      * Type `1`, `TA`: total alkalinity in mol/kg-sw.
      * Type `2`, `TC`: dissolved inorganic carbon in mol/kg-sw.
      * Type `3`, `PH`: pH on whichever scale(s) the constants in `Ks` are provided.
      * Type `4`, `PC`: partial pressure of CO2 in atm.
      * Type `5`, `FC`: fugacity of CO2 in atm.
      * Type `6`, `CARB`: carbonate ion in mol/kg-sw.
      * Type `7`, `HCO3`: bicarbonate ion in mol/kg-sw.
      * Type `8`, `CO2`: aqueous CO2 in mol/kg-sw.
    """
    # Expand inputs `par1` and `par2` into one array per core MCS variable
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = pair2core(par1, par2, par1type, par2type)
    # Generate vector describing the combination(s) of input parameters
    Icase = getIcase(par1type, par2type)
    # Solve the core marine carbonate system
    TA, TC, PH, PC, FC, CARB, HCO3, CO2 = solve.core(
        Icase, K0, TA, TC, PH, PC, FC, CARB, HCO3, CO2, PengCx, FugFac, Ks, totals
    )
    return TA, TC, PH, PC, FC, CARB, HCO3, CO2