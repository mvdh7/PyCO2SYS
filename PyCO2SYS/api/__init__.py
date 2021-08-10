# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2021  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Alternative APIs for executing the main CO2SYS function."""

from . import ezio
from ..engine import CO2SYS, _CO2SYS


def CO2SYS_wrap(
    dic=None,
    alk=None,
    pco2=None,
    fco2=None,
    pH=None,
    carb=None,
    bicarb=None,
    co2aq=None,
    sal=35,
    temp_in=25,
    temp_out=25,
    pres_in=0,
    pres_out=0,
    si=0,
    po4=0,
    nh3=0,
    h2s=0,
    K1K2_constants=4,
    KSO4_constants=1,
    KF_constant=1,
    pHscale_in=1,
    buffers_mode="auto",
    verbose=True,
):
    """
    A Pythonic API for PyCO2SYS that contains defaults and accepts/returns
    numpy.ndarrays, pandas.Series, xarray.DataArrays as inputs.

    Parameters
    ----------
    dic : array-like
        dissolved inorganic carbon in umol/kg
    alk : array-like
        total alkalinity in umol/kg
    pco2 : array-like
        partial pressure of carbon dioxide in uatm
    fco2 : array-like
        fugacity of carbon dioxide in uatm
    pH : array-like
        pH of seawater in total-scale
    carb : array-like
        carbonate ion in umol/kg
    bicarb : array-like
        bicarbonate ion in umol/kg
    co2aq : array-like
        aqueous CO2 in umol/kg
    sal : array-like
        salinity in parts per thousand
    temp : array-like
        temperature in degrees C, both input and output temperature can be given,
        distinguished with _in / _out suffix
    pres : array-like
        pressure in dbar (similar to depth in m), both input and output pressure
        can be given, distinguished with _in / _out suffix
    si : array-like
        total silicate in umol/kg
    po4 : array-like
        total phosphate in umol/kg
    nh3 : array-like
        total ammonia in umol/kg
    h2s : array-like
        total sulfide in umol/kg
    K1K2_constants : int
        The constants used to solve the marine carbonate system
        1   =  Roy, 1993
        2   =  Goyet & Poisson
        3   =  Hansson refit by Dickson AND Millero
        4   =  Mehrbach refit by Dickson AND Millero (DEFAULT)
        5   =  Hansson and Mehrbach refit by Dickson AND Millero
        6   =  GEOSECS (i.e., original Mehrbach)
        7   =  Peng    (i.e., original Mehrbach but without XXX)
        8   =  Millero, 1979, FOR PURE WATER ONLY (i.e., Sal=0)
        9   =  Cai and Wang, 1998
        10  =  Lueker et al, 2000
        11  =  Mojica Prieto and Millero, 2002.
        12  =  Millero et al, 2002
        13  =  Millero et al, 2006
        14  =  Millero, 2010
        15  =  Waters, Millero, & Woosley, 2014
    KSO4_constants : int
        The constants used for bisulfate dissociation and boron:salinity
        1  =  KSO4 of Dickson 1990a   & TB of Uppstrom 1974  (DEFAULT)
        2  =  KSO4 of Khoo et al 1977 & TB of Uppstrom 1974
        3  =  KSO4 of Dickson 1990a   & TB of Lee 2010
        4  =  KSO4 of Khoo et al 1977 & TB of Lee 2010
    KF_constant : int
        The constant used for hydrogen fluoride dissociation
        1  =  KF of Dickson & Riley 1979  (DEFAULT)
        2  =  KF of Perez & Fraga 1987
    pHscale : int
        The scale on which the input pH was determined
        1  =  Total scale  (DEFAULT)
        2  =  Seawater scale
        3  =  Free scale
        4  =  NBS scale
    buffers_mode : str
        Which method to use to evaluate buffer factors.
        'auto'      =  automatic differentiation (DEFAULT)
        'explicit'  =  explicit equations but without nutrient effects
        'none'      =  do not calculate buffers, return NaNs for them

    Returns
    -------
    pd.DataFrame or xr.Dataset containing the fully solved marine carbonate
    system parameters. Note that output variables are labelled as the original
    CO2SYS output names, and not the wrapper inputs.
    """
    from autograd import numpy as np
    import inspect
    import pandas as pd

    try:
        import xarray as xr

        hasxr = True
    except ImportError:
        hasxr = False

    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # MAKING ALL DATA ARRAYS
    # using a little bit of trickery to get all inputs
    # and make them arrays if they were floats or ints
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    cube = None
    params = {}  # will be used for data throughout the function
    for k in args:
        v = values[k]
        # XARRAY INSPECTION
        # if the input variables are xarray dataarrays, then
        # the structure of the dataarray will be saved and used
        # later to convert the data back into the same format
        if hasxr:
            if (cube is None) & (type(v) == xr.core.dataarray.DataArray):
                printv("Input is xarray.DataArray - output will be xr.Dataset")
                cube = v.copy() * np.nan
            if type(v) == xr.core.dataarray.DataArray:
                v = v.load()
        params[k] = np.array(v, ndmin=1).reshape(-1)

    # CO2 PARAMETERS - CHECK ONLY 2 PARAMS
    # we go through the carbon parameters to find the arrays that are None
    # these arrays will be dropped at the end of the code block
    # if there are more or less than 2 carbon params, an error will be raised
    keys = ["dic", "alk", "pco2", "fco2", "pH", "carb", "bicarb", "co2aq"]
    # find inputs that are None
    to_drop = [k for k in keys if (params[k] == None).all()]
    if len(set(keys) - set(to_drop)) != 2:
        raise KeyError(
            f"You must have two inputs for the marine carbonate system: "
            f"{', '.join(keys)}"
        )
    for k in to_drop:
        params.pop(k)  # drop the empty data

    # CONVERT DATA TO PANDAS DATAFRAME
    printv("Convert data to pandas.DataFrame")
    # first find the sizes of all the arrays in the params then
    # convert to a dataframe with the largest of the sizes providing the index
    # an error will be raised with information about the sizes if mismatched
    sizes = pd.Series({k: v.size for k, v in params.items()})
    if len(params["buffers_mode"]) < max(sizes):
        params["buffers_mode"] = np.full(max(sizes), params["buffers_mode"][0])
    try:
        df = pd.DataFrame(params, index=np.arange(max(sizes)))
    except ValueError:
        raise UserWarning(
            "Your inputs must be length of 1 or n (sizes shown below)"
            ":\n {}".format(str(sizes))
        )

    # DEFINE PARAMETER TYPES
    # use a dictionary to find the input parameters based on
    # the names of the CO2 parameters in the params dictionary
    parnum = dict(alk=1, dic=2, pH=3, pco2=4, fco2=5, carb=6, bicarb=7, co2aq=8)
    df["PAR1TYPE"] = parnum[df.columns[0]]
    df["PAR2TYPE"] = parnum[df.columns[1]]

    # set the correct order for the dataframe for CO2SYS
    df = df.loc[
        :,
        [
            df.columns.values[0],
            df.columns.values[1],
            "PAR1TYPE",
            "PAR2TYPE",
            "sal",
            "temp_in",
            "temp_out",
            "pres_in",
            "pres_out",
            "si",
            "po4",
            "pHscale_in",
            "K1K2_constants",
            "KSO4_constants",
            "nh3",
            "h2s",
            "KF_constant",
            "buffers_mode",
        ],
    ]
    df.columns = [
        "PAR1",
        "PAR2",
        "PAR1TYPE",
        "PAR2TYPE",
        "SAL",
        "TEMPIN",
        "TEMPOUT",
        "PRESIN",
        "PRESOUT",
        "SI",
        "PO4",
        "pHSCALEIN",
        "K1K2CONSTANTS",
        "KSO4CONSTANTS",
        "NH3",
        "H2S",
        "KFCONSTANT",
        "buffers_mode",
    ]

    # REMOVE NANS FOR EFFICIENCY
    printv("Removing nans for efficiency")
    # in order to save processing time on the MATLAB side, we remove all
    # the nans in the data. This will speed up things quite a bit if there
    # are a large number of nans, which may often be the case if you're
    # giving xarray datasets to the function.
    df_nonan = df.dropna(subset=[df.columns[0], df.columns[1]])

    # COMPUTING CO2sys PARAMETERS
    printv("Computing CO2 parameters")
    dict_out = CO2SYS(*df_nonan.values.T)

    # CONVERTING DICTIONARY TO PANDAS.DATAFRAME
    # here we convert to a pandas.DataFrame and we use the index
    # from the data without nans (that was passed to CO2SYS).
    df_out = pd.DataFrame(dict_out, index=df_nonan.index.values)
    # In the following step we reindex to the dataframe that has nans
    # these will be reinserted into the dataframe
    printv("Insert nans back into DataFrame")
    dfo = df_out.reindex(df.index)

    # MAKING XARRAY IF INPUT MATCHES XARRAY
    # if any of the inputs were an xr.DataArray, we now convert
    # all the output to an xr.Dataset with the names derived from
    # the headers.
    if cube is not None:
        printv("Converting data to xarray.Dataset")
        xds = xr.Dataset()
        coords = cube.coords
        dims = tuple(cube.dims)
        for key in dfo:
            xds[key] = xr.DataArray(
                dfo[key].values.reshape(*cube.shape), dims=dims, coords=coords
            )
        return xds
    else:
        return dfo


def CO2SYS_MATLABv3(
    PAR1,
    PAR2,
    PAR1TYPE,
    PAR2TYPE,
    SAL,
    TEMPIN,
    TEMPOUT,
    PRESIN,
    PRESOUT,
    SI,
    PO4,
    NH3,
    H2S,
    pHSCALEIN,
    K1K2CONSTANTS,
    KSO4CONSTANT,
    KFCONSTANT,
    BORON,
    buffers_mode="auto",
    WhichR=3,
    totals=None,
    equilibria_in=None,
    equilibria_out=None,
):
    """Run CO2SYS with the new MATLAB v3 syntax and updated gas constant value."""
    return _CO2SYS(
        PAR1,
        PAR2,
        PAR1TYPE,
        PAR2TYPE,
        SAL,
        TEMPIN,
        TEMPOUT,
        PRESIN,
        PRESOUT,
        SI,
        PO4,
        NH3,
        H2S,
        pHSCALEIN,
        K1K2CONSTANTS,
        KSO4CONSTANT,
        KFCONSTANT,
        BORON,
        buffers_mode,
        WhichR,
        KSO4CONSTANTS=0,
        totals=totals,
        equilibria_in=equilibria_in,
        equilibria_out=equilibria_out,
    )
