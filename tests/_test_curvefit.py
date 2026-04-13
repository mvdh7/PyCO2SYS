def fitting_function(t_s, a, b, c, d, e, f):
    """Carbonic acid dissociation constants with K2 following MMB25.
    K1 should come from WMW14.
    Used when opt_k_carbonic = 19.
    """
    temperature, salinity = t_s
    TempK = temperature + 273.15
    Sal = salinity
    pK2 = (
        a
        + b / TempK
        + c / TempK**2
        + d * np.sqrt(Sal) / (1 + 1.11 * np.sqrt(Sal))
        + e * Sal / np.log(TempK)
        + f * np.sqrt(Sal) * TempK
    )
    return pK2


pk_ff = fitting_function(
    [cv.Temp_C.values, cv.Salinity.values],
    5.1703,
    2136.77,
    -177788,
    -0.4457,
    0.0674,
    -0.0008238,
)
cf = curve_fit(
    fitting_function,
    [cv.Temp_C.values, cv.Salinity.values],
    cv.param_pK2.values,
)
