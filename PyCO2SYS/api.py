

def CO2SYS_wrap(dic=None, alk=None, pco2=None, fco2=None, pH=None,
                sal=34.5, temp_in=25, temp_out=25, pres_in=0, pres_out=0,
                si=5.5, po4=0.5,
                K1K2_constants=4, KSO4_constants=1, pHscale_in=1):
    """
    A pythonic api for PyCO2SYS that contains defaults and accepts/returns
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
    sal : array-like
        salinity in parts per thousand
    temp : array-like
        temperature in degress C, both input and output temeprature can be given
    pres : array-like
        pressure in dbar (akin to depth), both input and output pressure can be given
    si : array-like
        silicate concentration in umol/kg
    po4 : array-like
        phosphate concentration in umol/kg
    K1K2_constants : int
        The constants used to solve the marine carbonate system
        1  =  Roy, 1993                                          T:    0-45  S:  5-45. Total scale. Artificial seawater.
        2  =  Goyet & Poisson                                    T:   -1-40  S: 10-50. Seaw. scale. Artificial seawater.
        3  =  Hansson refit by Dickson AND Millero               T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
        4  =  Mehrbach refit by Dickson AND Millero              T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
        5  =  Hansson and Mehrbach refit by Dickson AND Millero  T:    2-35  S: 20-40. Seaw. scale. Artificial seawater.
        6  =  GEOSECS (i.e., original Mehrbach)                  T:    2-35  S: 19-43. NBS scale. Real seawater.
        7  =  Peng    (i.e., originam Mehrbach but without XXX)  T:    2-35  S: 19-43. NBS scale. Real seawater.
        8  =  Millero, 1979, FOR PURE WATER ONLY (i.e., Sal=0)   T:    0-50  S:     0.
        9  =  Cai and Wang, 1998                                 T:    2-35  S:  0-49. NBS scale. Real and artificial seawater.
        10  =  Lueker et al, 2000                                 T:    2-35  S: 19-43. Total scale. Real seawater.
        11  =  Mojica Prieto and Millero, 2002.                   T:    0-45  S:  5-42. Seaw. scale. Real seawater
        12  =  Millero et al, 2002                                T: -1.6-35  S: 34-37. Seaw. scale. Field measurements.
        13  =  Millero et al, 2006                                T:    0-50  S:  1-50. Seaw. scale. Real seawater.
        14  =  Millero, 2010                                      T:    0-50  S:  1-50. Seaw. scale. Real seawater.
        15  =  Waters, Millero, & Woosley, 2014")
    KO4_constants : int
        The constants used for phosphate calculations
        1  =  KSO4 of Dickson 1990a   & TB of Uppstrom 1974  (PREFERRED)
        2  =  KSO4 of Khoo et al 1977 & TB of Uppstrom 1974
        3  =  KSO4 of Dickson 1990a   & TB of Lee 2010
        4  =  KSO4 of Khoo et al 1977 & TB of Lee 2010
    pHscale : int
        The scale on which the input pH was determined
        1  =  Total scale
        2  =  Seawater scale
        3  =  Free scale
        4  =  NBS scale
    """
    import numpy as np
    import inspect
    import pandas as pd
    from . import CO2SYS

    try:
        import xarray as xr
        hasxr = True
    except ImportError:
        hasxr = False

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
                print('input is xarray.DataArray - output will be xr.Dataset')
                cube = v.copy() * np.nan
            if type(v) == xr.core.dataarray.DataArray:
                v = v.load()
        params[k] = np.array(v, ndmin=1).reshape(-1)

    # CO2 PARAMETERS - CHECK ONLY 2 PARAMS
    # we go through the carbon parameters to find the arrays that are None
    # these arrays will be dropped at the end of the code block
    # if there are more or less than 2 carbon params, an error will be raised
    keys = 'dic  alk  pco2  fco2  pH'.split()  # lazy way to make co2 keys
    to_drop = [k for k in keys if (params[k] == None).all()]  # find inputs that are None
    assert len(to_drop) != 2, 'You must only have two inputs for the marine carbonate system.'
    for k in to_drop:
        params.pop(k)  # drop the empty data

    # CONVERT DATA TO PANDAS DATAFRAME
    print('Convert data to pandas.DataFrame')
    # first find the sizes of all the arrays in the params
    # then convert to a dataframe with the largest of the sizes providing the index
    # an error will be raised with information about the sizes if mismatched
    sizes = pd.Series({k: v.size for k, v in params.items()})
    try:
        df = pd.DataFrame(params, index=np.arange(max(sizes)))
    except ValueError:
        raise UserWarning('Your inputs must be length of 1 or n (sizes shown below):\n {}'.format(str(sizes)))

    # DEFINE PARAMETER TYPES
    # use a dictionary to find the input parameters based on
    # the names of the CO2 parameters in the params dictionary
    parnum = dict(alk=1, dic=2, pH=3, pco2=4, fco2=5)
    df['PAR1TYPE'] = parnum[df.columns[0]]
    df['PAR2TYPE'] = parnum[df.columns[1]]

    # set the correct order for the dataframe for CO2SYS
    df = df.loc[:, [df.columns.values[0], df.columns.values[1], "PAR1TYPE", "PAR2TYPE",
                    "sal", "temp_in", "temp_out", "pres_in", "pres_out", "si", "po4",
                    "pHscale_in", "K1K2_constants", "KSO4_constants"]]
    df.columns = ['PAR1', 'PAR2', 'PAR1TYPE', 'PAR2TYPE',
                'SAL', 'TEMPIN', 'TEMPOUT', 'PRESIN', 'PRESOUT',
                'SI', 'PO4', 'pHSCALEIN', 'K1K2CONSTANTS', 'KSO4CONSTANTS']

    # REMOVE NANS FOR EFFICIENCY
    print('Removing nans for efficiency')
    # in oder to save processing time on the MATLAB side, we remove all
    # the nans in the data. This will speed up things quite a bit if there
    # are a large number of nans, which may often be the case if you're
    # giving xarray datasets to the function.
    df_nonan = df.dropna(subset=[df.columns[0], df.columns[1]]).astype(float)

    # COMPUTING CO2sys PARAMETERS
    print('Computing CO2 parameters')
    dict_out = CO2SYS(*df_nonan.values.T)

    # CONVERTING DICTIONARY TO PADNAS.DATAFRAME
    # here we convert to a pandas.DataFrame and we use the index
    # from the data without nans (that was passed to CO2SYS).
    df_out = pd.DataFrame(dict_out, index=df_nonan.index.values)
    # In the following step we reindex to the dataframe that has nans
    # these will be reinserted into the dataframe
    print('Insert nans back into DataFrame')
    dfo = df_out.reindex(df.index)

    # MAKING XARRAY IF INPUT MATCHES XARRAY
    # if any of the inputs were an xr.DataArray, we now convert
    # all the output to an xr.Dataset with the names derived from
    # the headers.
    if cube is not None:
        print('Converting data to xarray.Dataset')
        xds = xr.Dataset()
        coords = cube.coords
        dims = tuple(cube.dims)
        for key in dfo:
            xds[key] = xr.DataArray(
                dfo[key].values.reshape(*cube.shape),
                dims=dims, coords=coords)
        return xds
    else:
        return dfo
