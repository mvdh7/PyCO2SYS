import numpy as np, matplotlib as mpl
from matplotlib import pyplot as plt
import PyCO2SYS as pyco2

# Matplotlib settings
mpl.rcParams["font.family"] = "Open Sans"
mpl.rcParams["font.size"] = mpl.rcParams["axes.titlesize"] = 7

# Use these!  k_alpha and k_beta need automating
dx_values = {
    "salinity": 1e-3,
    "temperature": 1e-3,
    "pressure": 1e-3,
    "par1": 1e-4,  # higher to cope with pH input
    "par2": 1e-4,  # higher to cope with pH input
    "total_silicate": 1e-3,
    "total_phosphate": 1e-3,
    "total_ammonia": 1e-3,
    "total_sulfide": 1e-3,
    "total_borate": 1e-3,
    "total_sulfate": 1e-3,
    "total_fluoride": 1e-3,
    "total_alpha": 1e-3,
    "total_beta": 1e-3,
    "total_calcium": 1e-3,
    "k_CO2": 1e-8,
    "k_carbonic_1": 1e-12,
    "k_carbonic_2": 1e-15,
    "k_water": 1e-20,
    "k_borate": 1e-15,
    "k_bisulfate": 1e-7,
    "k_fluoride": 1e-9,
    "k_phosphoric_1": 1e-4,  # needs to be weirdly low, not sure why
    "k_phosphoric_2": 1e-12,
    "k_phosphoric_3": 1e-15,
    "k_silicate": 1e-16,
    "k_ammonia": 1e-16,
    "k_sulfide": 1e-13,
    "k_calcite": 1e-13,
    "k_aragonite": 1e-13,
}

as_pk = False
log10_dx_for_pks = 4  # this works as a standard for pk adjustments if needed (not?)

for test_arg in ["k_carbonic_1"]:

    if test_arg in ["k_calcite", "total_calcium"]:
        target_arg = "saturation_calcite"
    elif test_arg == "k_aragonite":
        target_arg = "saturation_aragonite"
    else:
        target_arg = "pH"
        # target_arg = "dic"
    
    kwargs_raw = {
        "par1": 2250,
        "par2": 2150,
        "par1_type": 1,
        "par2_type": 2,
        "total_phosphate": 10,
        "total_silicate": 10,
        "total_ammonia": 10,
        "total_sulfide": 10,
        "temperature": 0,
    }
    results_raw = pyco2.sys(**kwargs_raw)
    
    pks = {k: -np.log10(v) for k, v in results_raw.items() if k.startswith("k")}
    
    if test_arg.startswith("k_") and not as_pk:
        k_floor = np.floor(pks[test_arg])
        log10_delta = np.arange(k_floor + 2, k_floor + 13, 0.01)
    else:
        log10_delta = np.arange(0, 11, 0.01)
    delta = 10.0 ** -log10_delta
    kwargs = kwargs_raw.copy()
    if test_arg not in kwargs:
        kwargs.update({test_arg: results_raw[test_arg]})
    do_pk = test_arg.startswith("k_") and as_pk
    if do_pk:
        pk_raw = -np.log10(kwargs[test_arg]) + delta
        kwargs.update({test_arg: 10.0 ** -pk_raw})
    else:
        kwargs.update({test_arg: kwargs[test_arg] + delta})
    results = pyco2.sys(**kwargs)
    target_grad = (results[target_arg] - results_raw[target_arg]) / delta
    
    fig, ax = plt.subplots(dpi=300, figsize=(3.5, 2.5))
    ax.plot(log10_delta, target_grad, lw=0.8, alpha=0.8, c="xkcd:bright red")
    if do_pk:
        extra = "p"
    else:
        extra = ""
    ax.set_xlabel("$-$log$_{10}(\Delta$" + "{}[{}])".format(extra, test_arg))
    ax.set_ylabel("d[{}] / d{}[{}]".format(target_arg, extra, test_arg))
    if test_arg in dx_values:
        if do_pk:
            nearest_ix = np.argmin(np.abs(delta - 10 ** -log10_dx_for_pks))
            vlinex = log10_dx_for_pks
        else:
            nearest_ix = np.argmin(np.abs(delta - dx_values[test_arg]))
            vlinex = -np.log10(dx_values[test_arg])
        ax.axvline(
            vlinex, zorder=-1, lw=1, c="xkcd:ocean blue", alpha=0.8
        )
        ax.axhline(target_grad[nearest_ix], zorder=-1, lw=1, c="xkcd:ocean blue", alpha=0.8)
    ax.grid(alpha=0.3)
