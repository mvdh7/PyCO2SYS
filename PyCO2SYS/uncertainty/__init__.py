# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Propagate uncertainties through marine carbonate system calculations."""

from scipy.misc import derivative
from autograd.numpy import array, isin, size, sqrt
from autograd.numpy import all as np_all
from autograd.numpy import sum as np_sum
from . import automatic
from .. import convert, engine

__all__ = ["automatic"]


def derivatives(co2dict, grads_of, grads_wrt, dx=1e-8, use_explicit=True, verbose=True):
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
    if isinstance(grads_wrt, str):
        assert (grads_wrt == "all") or grads_of in inputs_wrt
        if grads_wrt == "all":
            grads_wrt = inputs_wrt
        else:
            grads_wrt = [grads_wrt]
    assert np_all(isin(list(grads_wrt), inputs_wrt,)), "Invalid `grads_wrt` requested."
    if isinstance(grads_of, str):
        assert (grads_of == "all") or grads_of in engine.gradables
        if grads_of == "all":
            grads_of = engine.gradables
        else:
            grads_of = [grads_of]
    assert np_all(isin(grads_of, engine.gradables,)), "Invalid `grads_of` requested."
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
    # Get the gradients
    co2derivs = {of: {wrt: None for wrt in grads_wrt} for of in grads_of}
    # Get gradients w.r.t. internal variables

    # Define gradients that we have explicit methods for, if requested
    if use_explicit:
        # Automatic derivatives for PAR1/PAR2 propagation into core MCS
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

                def kfunc(v, co2args):
                    co2args[wrt] = v
                    return engine._CO2SYS(**co2args)[of]

                co2derivs[of][wrt] = derivative(
                    kfunc, co2args[wrt], dx=dx, args=[co2args]
                )
    return co2derivs


def propagate(
    co2dict,
    uncertainties_from,
    uncertainties_into,
    dx=1e-8,
    use_explicit=True,
    verbose=True,
):
    """Propagate uncertainties from requested inputs to outputs."""
    co2derivs = derivatives(
        co2dict,
        uncertainties_into,
        uncertainties_from,
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
