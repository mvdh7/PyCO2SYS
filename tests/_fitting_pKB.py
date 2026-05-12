#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:06:12 2026

@author: mmartinmayor
"""

# #uncertainty matrix for pKB?
#     pKB = (
#         -60.0439
#         + 88.29443 * np.sqrt(Sal)
#         + 3709.084 / TempK
#         + 2982.70 * np.sqrt(Sal) / TempK
#         - 0.078359 * np.sqrt(Sal)/(1 + 1.11 * np.sqrt(Sal))
#         + 9.9762 * np.log(TempK)
#         - 524.5915 * Sal/np.log(TempK)
#         - 0.0213143 * np.sqrt(Sal)* np.log(TempK)
#         + 0.00001828 * Sal* TempK
#     )

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


cv = pd.read_excel("tests/data/combined_pKB_data.xlsx")


def fitting_function(t_s, a, b, c, d, e, f, g, h, i):
    Temp_K, Sal = t_s
    pKB = (
        +a
        + b * np.sqrt(Sal)
        + c / Temp_K
        + d * np.sqrt(Sal) / Temp_K
        + e * np.sqrt(Sal) / (1 + np.sqrt(Sal))
        + f * np.log(Temp_K)
        + g * np.sqrt(Sal) / np.log(Temp_K)
        + h * np.sqrt(Sal) * Temp_K
        + i * Sal * Temp_K
    )
    return pKB


# pk_ff = fitting_function(
#     [cv.Temp_C.values, cv.Salinity.values],
#     5.1703,
#     2136.77,
#     -177788,
#     -0.4457,
#     0.0674,
#     -0.0008238,
# )

cf = curve_fit(
    fitting_function,
    [cv.Temp_K.values, cv.Sal.values],
    cv.pKB_combined.values,
    method="lm",
)
# Above does not give the exact same coefficients than the parameterization in R
# lm() in R - QR decomposition and designed for linear models and it is an exact least squares solution
# curve_fit in Python is designed for non-linear fitting and it is iterative
# To fit the data like R is doing you need to run:

pKB = np.column_stack(
    [
        np.ones(len(cv)),  # intercept term - in R ~ 1
        np.sqrt(cv.Sal),
        1 / cv.Temp_K,
        np.sqrt(cv.Sal) / cv.Temp_K,
        np.sqrt(cv.Sal) / (1 + np.sqrt(cv.Sal)),
        np.log(cv.Temp_K),
        np.sqrt(cv.Sal) / np.log(cv.Temp_K),
        np.sqrt(cv.Sal) * cv.Temp_K,
        cv.Sal * cv.Temp_K,
    ]
)

exp_pKB = cv.pKB_combined.values

coef_pKB = np.linalg.lstsq(pKB, exp_pKB, rcond=None)[0]


# residuals
residuals = exp_pKB - pKB @ coef_pKB
n, p = pKB.shape

# estimate of error variance (sigma^2)
sigma2 = np.sum(residuals**2) / (n - p)

# covariance matrix of coefficients (9x9)
cov_coef_pKB = sigma2 * np.linalg.inv(pKB.T @ pKB)

# std error of every term in the eq.
stderror = np.sqrt(np.diag(cov_coef_pKB))
