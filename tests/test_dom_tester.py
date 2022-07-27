import numpy as np
from matplotlib import pyplot as plt

# from test_dom import nica_log10_chi, nd_fulvic


def get_donnan(log10_chi, a, b):
    chi = 10.0**log10_chi
    return a * chi**2 + b * chi


def get_nica(log10_chi, q1, q2, k1cH, k2cH, m1, m2):
    chi = 10.0**log10_chi
    return (
        q1 * (k1cH * chi) ** m1 / (1 + (k1cH * chi) ** m1)
        + q2 * (k2cH * chi) ** m2 / (1 + (k2cH * chi) ** m2)
        - q1
        - q2
    )


log10_chi = np.linspace(-10, 10, num=1000)
donnan = get_donnan(log10_chi, 0, 0)
nica = get_nica(log10_chi, 6, 1, 1e-5 * 1e2, 1e-5 * 1e8, 0.5, 0.5)
# nica2 = nica_log10_chi(log10_chi, 1e-8, nd_fulvic)

fig, ax = plt.subplots(dpi=300)
ax.plot(log10_chi, donnan, label="Donnan")
ax.plot(log10_chi, nica, label="NICA")
# ax.plot(log10_chi, nica2 - 7)
# ax.set_ylim(np.array([-1, 1]) * 10)
