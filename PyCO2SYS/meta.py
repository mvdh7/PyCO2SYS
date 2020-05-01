# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020  Matthew Paul Humphreys et al.  (GNU GPLv3)
"""Define metadata about the PyCO2SYS package."""

version = "1.3.0"
authorlist = [
    "Humphreys, Matthew P.",
    "Gregor, Luke",
    "Pierrot, Denis",
    "van Heuven, Steven M. A. C.",
    "Lewis, Ernie",
    "Wallace, Douglas W. R.",
]
authors = " and ".join(authorlist)


def say_hello():
    print(
        """
 MP Humphreys, L Gregor, D Pierrot, SMAC van Heuven, E Lewis & D Wallace

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
                                    sign on the entrance gates of hell""".format(
            version
        )
    )  # (All hope abandon, ye who enter here!)
