# PyCO2SYS: marine carbonate system calculations in Python.
# Copyright (C) 2020--2026  Matthew P. Humphreys et al.  (GNU GPLv3)
import importlib

import numpy as np


def bh_H24():
    covmx_path = importlib.resources.files(
        "PyCO2SYS.uncertainty.covmx"
    ).joinpath("bh_H24.txt")
    return np.genfromtxt(covmx_path)
