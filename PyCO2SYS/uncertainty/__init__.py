# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Propagate uncertainties through marine carbonate system calculations."""

from copy import deepcopy
from scipy.misc import derivative
from autograd.numpy import array, isin, size, sqrt
from autograd.numpy import all as np_all
from autograd.numpy import sum as np_sum
from . import automatic
from .. import convert, engine

__all__ = ["automatic"]


def derivatives(
    co2dict,
    grads_of,
    grads_wrt,
    totals=None,
    equilibria_input=None,
    equilibria_output=None,
    dx=1e-8,
    use_explicit=True,
    verbose=True,
):
    """Get derivatives of `co2dict` values w.r.t. the main function inputs.

    `co2dict` is output by `PyCO2SYS.CO2SYS`.
    `grads_of` is a list of keys from `co2dict` that you want to calculate the
    derivatives of, or a single key, or `"all"`.
    `grads_wrt` is a list of `PyCO2SYS.CO2SYS` input variable names that you want to
    calculate the derivatives with respect to, or a single name, or `"all"`.
    """

    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Derivatives can be calculated w.r.t. these inputs only
    inputs_wrt = [
        "PAR1",
        "PAR2",
        "SAL",
        "TEMPIN",
        "TEMPOUT",
        "PRESIN",
        "PRESOUT",
        "SI",
        "PO4",
        "NH3",
        "H2S",
    ]
    totals_wrt = ["TB", "TF", "TSO4", "TCa"]
    Ks_wrt = [
        "KSO4",
        "KF",
        "fH",
        "KB",
        "KW",
        "KP1",
        "KP2",
        "KP3",
        "KSi",
        "K1",
        "K2",
        "KH2S",
        "KNH3",
        "K0",
        "FugFac",
    ]
    Kis_wrt = ["{}input".format(K) for K in Ks_wrt]
    Kos_wrt = ["{}output".format(K) for K in Ks_wrt]
    # If only a single w.r.t. is requested, check it's allowed & convert to list
    groups_wrt = ["all", "measurements", "totals", "equilibria_in", "equilibria_out"]
    all_wrt = groups_wrt + inputs_wrt + totals_wrt + Kis_wrt + Kos_wrt
    if isinstance(grads_wrt, str):
        assert grads_wrt in all_wrt
        if grads_wrt == "all":
            grads_wrt = all_wrt
        elif grads_wrt == "measurements":
            grads_wrt = inputs_wrt
        elif grads_wrt == "totals":
            grads_wrt = totals_wrt
        elif grads_wrt == "equilibria_in":
            grads_wrt = Kis_wrt
        elif grads_wrt == "equilibria_out":
            grads_wrt = Kos_wrt
        else:
            grads_wrt = [grads_wrt]
    # Make sure all requested w.r.t.'s are allowed
    assert np_all(isin(list(grads_wrt), all_wrt)), "Invalid `grads_wrt` requested."
    # If only a single grad of is requested, check it's allowed & convert to list
    if isinstance(grads_of, str):
        assert grads_of in ["all"] + engine.gradables
        if grads_of == "all":
            grads_of = engine.gradables
        else:
            grads_of = [grads_of]
    # Make sure all requested grads of are allowed
    assert np_all(isin(grads_of, engine.gradables)), "Invalid `grads_of` requested."
    # Assemble dict of input arguments for engine._CO2SYS()
    co2args = {
        arg: co2dict[arg]
        for arg in [
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
            "NH3",
            "H2S",
            "pHSCALEIN",
            "K1K2CONSTANTS",
            "KSO4CONSTANT",
            "KFCONSTANT",
            "BORON",
            "buffers_mode",
            "KSO4CONSTANTS",
        ]
    }
    co2args["totals"] = totals
    co2args["equilibria_input"] = equilibria_input
    co2args["equilibria_output"] = equilibria_output
    # Get totals/Ks values from the `co2dict` too
    co2dict_totals = engine.dict2totals_umol(co2dict)
    co2dict_Kis, co2dict_Kos = engine.dict2Ks(co2dict)
    # Preallocate output dict to store the gradients
    co2derivs = {of: {wrt: None for wrt in grads_wrt} for of in grads_of}
    # Define gradients that we have explicit methods for, if not unrequested
    if use_explicit:
        # Use automatic derivatives for PAR1/PAR2 propagation into core MCS
        pars_requested = [wrt for wrt in grads_wrt if wrt in ["PAR1", "PAR2"]]
        p1p2u = automatic.pars2core(co2dict, pars_requested)
        for wrt in pars_requested:
            for of, v in p1p2u[wrt].items():
                if of in grads_of:
                    co2derivs[of][wrt] = v
    # Get central difference derivatives for the rest
    for of in grads_of:
        printv("Computing derivatives of {}...".format(of))
        for wrt in grads_wrt:
            if co2derivs[of][wrt] is None:
                if wrt in inputs_wrt:

                    def kfunc(v, co2args):
                        co2args[wrt] = v
                        return engine._CO2SYS(**co2args)[of]

                    co2derivs[of][wrt] = derivative(
                        kfunc, co2args[wrt], dx=dx, args=[co2args]
                    )
                elif wrt in totals_wrt:
                    tco2args = deepcopy(co2args)
                    if totals is None:
                        tco2args["totals"] = {}
                    if wrt not in tco2args["totals"]:
                        tco2args["totals"][wrt] = co2dict_totals[wrt]

                    def kfunc(v, tco2args):
                        tco2args["totals"][wrt] = v
                        return engine._CO2SYS(**tco2args)[of]

                    co2derivs[of][wrt] = derivative(
                        kfunc, tco2args["totals"][wrt], dx=dx, args=[tco2args]
                    )
                elif wrt in Kis_wrt:
                    tco2args = deepcopy(co2args)
                    twrt = wrt.replace("input", "")
                    if equilibria_input is None:
                        tco2args["equilibria_input"] = {}
                    if wrt not in tco2args["equilibria_input"]:
                        tco2args["equilibria_input"][twrt] = co2dict_Kis[twrt]

                    def kfunc(v, tco2args):
                        tco2args["equilibria_input"][twrt] = v
                        return engine._CO2SYS(**tco2args)[of]

                    co2derivs[of][wrt] = derivative(
                        kfunc,
                        tco2args["equilibria_input"][twrt],
                        dx=dx,
                        args=[tco2args],
                    )
                elif wrt in Kos_wrt:
                    tco2args = deepcopy(co2args)
                    twrt = wrt.replace("output", "")
                    if equilibria_input is None:
                        tco2args["equilibria_output"] = {}
                    if wrt not in tco2args["equilibria_output"]:
                        tco2args["equilibria_output"][twrt] = co2dict_Kos[twrt]

                    def kfunc(v, tco2args):
                        tco2args["equilibria_output"][twrt] = v
                        return engine._CO2SYS(**tco2args)[of]

                    co2derivs[of][wrt] = derivative(
                        kfunc,
                        tco2args["equilibria_output"][twrt],
                        dx=dx,
                        args=[tco2args],
                    )
    return co2derivs


def propagate(
    co2dict,
    uncertainties_into,
    uncertainties_from,
    totals=None,
    equilibria_input=None,
    equilibria_output=None,
    dx=1e-8,
    use_explicit=True,
    verbose=True,
):
    """Propagate uncertainties from requested inputs to outputs."""
    co2derivs = derivatives(
        co2dict,
        uncertainties_into,
        uncertainties_from,
        totals=totals,
        equilibria_input=equilibria_input,
        equilibria_output=equilibria_output,
        dx=dx,
        use_explicit=use_explicit,
        verbose=verbose,
    )
    npts = size(co2dict["PAR1"])
    uncertainties_from = engine.condition(uncertainties_from, npts=npts)[0]
    components = {
        u_into: {
            u_from: co2derivs[u_into][u_from] * v_from
            for u_from, v_from in uncertainties_from.items()
        }
        for u_into in uncertainties_into
    }
    uncertainties = {
        u_into: sqrt(
            np_sum(
                array([component for component in components[u_into].values()]) ** 2,
                axis=0,
            )
        )
        for u_into in uncertainties_into
    }
    return uncertainties, components
