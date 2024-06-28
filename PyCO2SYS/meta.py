# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2024  Matthew P. Humphreys et al.  (GNU GPLv3)
"""Define metadata about PyCO2SYS."""
import jax
from jax import numpy as np

version = "1.8.3.1"
version_xyz = "1.8.3"
authorlist = [
    "Humphreys, Matthew P.",
    "Schiller, Abigail J.",
    "Sandborn, Daniel E.",
    "Gregor, Luke",
    "Pierrot, Denis",
    "van Heuven, Steven M. A. C.",
    "Lewis, Ernie R.",
    "Wallace, Douglas W. R.",
]
authors = " and ".join(authorlist)


def hello():
    print(
        """
               M.P. Humphreys, A.J. Schiller, D.E. Sandborn,
                L. Gregor, D. Pierrot, S.M.A.C. van Heuven,
                        E.R. Lewis & D.W.R. Wallace

                             ~~~ present ~~~

        PyCO2SYS: marine carbonate system calculations in Python.
               Version {} :: doi:10.5281/zenodo.3744275


  Py  CCCC       OOOOO        222        SSS      YY      YY      SSS
     CC   C     OO   OO     22   22    SS   SS     YY    YY     SS   SS
    CC         OO     OO         22    SS           YY  YY      SS
    CC         OO     OO        22        S           YY          SSS
    CC         OO     OO       22           SS        YY             SS
     CC   C     OO   OO      22        SS   SS        YY        SS   SS
      CCCC       OOOOO      2222222      SSS          YY          SSS


   Lasciate ogni speranza, voi ch' entrate!
                                    Dante, Inferno iii, 9
                                    sign on the entrance gates of hell
""".format(
            version_xyz
        )
    )  # (All hope abandon, ye who enter here!)


def egrad(g):
    # https://github.com/google/jax/issues/3556#issuecomment-649779759
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped
