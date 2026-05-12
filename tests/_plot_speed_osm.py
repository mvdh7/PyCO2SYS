# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

v1 = {}
v1["sidelength"] = np.array(
    [
        1,
        # 10,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1200,  # NOTE the 1200 run here is just one run not an average of 10!
    ]
)
v1["peak"] = np.array(
    [
        4.1,
        # 5.0,
        117.4,
        456.5,
        1022.0,
        1813.4,
        2830.9,
        4731.0,
        6437.5,
        8406.6,
        11376.3,
        13132.5,
        22844.0,  #
    ]
)
v1["peak_pm"] = np.array(
    [
        0.048,
        # 0.048,
        0.050,
        0.050,
        0.050,
        0.050,
        0.050,
        0.046,
        0.046,
        0.046,
        0.047,
        0.007,
        0,  #
    ]
)
v1["final"] = np.array(
    [
        0.3,
        # 0.4,
        7.0,
        27.2,
        60.8,
        107.7,
        168.2,
        242.0,
        329.3,
        430.0,
        544.2,
        671.8,
        967.2,  #
    ]
)
v1["final_pm"] = np.array(
    [
        0.048,
        # 0.046,
        0.047,
        0.047,
        0.047,
        0.047,
        0.047,
        0.044,
        0.044,
        0.044,
        0.042,
        0.006,
        0,  #
    ]
)
v1["runtime"] = np.array(
    [
        0.53,
        # 0.39,
        0.66,
        1.05,
        1.73,
        2.74,
        4.05,
        6.64,
        8.83,
        12.84,
        30.70,
        75.47,
        181.73,  #
    ]
)
v1["runtime_pm"] = np.array(
    [
        0.01,
        # 0.01,
        0.02,
        0.02,
        0.02,
        0.03,
        0.09,
        0.12,
        0.11,
        0.83,
        4.72,
        8.17,
        0,  #
    ]
)
v1 = pd.DataFrame(v1)

v2_pH = {}
v2_pH["sidelength"] = [10, 200, 400, 600, 800, 1000, 1200]
v2_pH["peak"] = [3.6, 5.3, 15.1, 32.6, 59.2, 91.2, 133.4]
v2_pH["peak_pm"] = [0.021, 0.390, 2.028, 4.351, 6.188, 6.125, 9.153]
v2_pH["final"] = [3.0, 3.6, 5.4, 8.5, 12.7, 18.2, 25.0]
v2_pH["final_pm"] = [0.004, 0.001, 0.001, 0.001, 0.026, 0.001, 0.001]
v2_pH["runtime1"] = [1.75, 1.95, 1.89, 2.09, 2.26, 2.55, 3.12]
v2_pH["runtime1_pm"] = [0.03, 0.04, 0.03, 0.04, 0.03, 0.05, 0.05]
v2_pH["runtime2"] = [0.017, 0.05, 0.13, 0.31, 0.55, 0.83, 1.45]
v2_pH["runtime2_pm"] = [0.002, 0.01, 0.01, 0.03, 0.02, 0.04, 0.03]
v2_pH = pd.DataFrame(v2_pH)

v2_all = {}
v2_all["sidelength"] = [1, 200, 400, 600, 800, 1000, 1200]
v2_all["peak"] = [5.3, 5.9, 16.1, 36.3, 66.3, 101.8, 145.5]
v2_all["peak_pm"] = [0.002, 0.002, 2.121, 3.584, 0.079, 0.212, 0.001]
v2_all["final"] = [4.7, 5.3, 7.2, 10.2, 14.5, 20.0, 26.7]
v2_all["final_pm"] = [0.002, 0.002, 0.004, 0.002, 0.002, 0.004, 0.002]
v2_all["runtime1"] = [3.76, 4.48, 4.55, 4.95, 5.35, 5.99, 7.43]
v2_all["runtime1_pm"] = [0.06, 0.07, 0.14, 0.09, 0.07, 0.15, 0.19]
v2_all["runtime2"] = [0.156, 0.260, 0.420, 0.906, 1.675, 2.498, 4.403]
v2_all["runtime2_pm"] = [0.005, 0.005, 0.011, 0.046, 0.118, 0.042, 0.156]
v2_all = pd.DataFrame(v2_all)

c_v1 = "xkcd:grapefruit"
c_v2_pH = "xkcd:sky"
c_v2_all = "xkcd:cerulean blue"

fig, axs = plt.subplots(nrows=2, figsize=(5, 7))

markers = dict(
    marker="o",
    markersize=2,
)

ax = axs[0]
ax.plot(
    v1.sidelength,
    v1.runtime,
    c=c_v1,
    **markers,
    label="v1",
)
ax.plot(
    v2_pH.sidelength,
    v2_pH.runtime1,
    c=c_v2_pH,
    **markers,
    ls=":",
    lw=1.2,
    label="v2 (pH + JIT)",
)
ax.plot(
    v2_pH.sidelength,
    v2_pH.runtime2,
    c=c_v2_pH,
    **markers,
    label="v2 (pH)",
)
ax.plot(
    v2_all.sidelength,
    v2_all.runtime1,
    c=c_v2_all,
    **markers,
    ls=":",
    lw=1.2,
    label="v2 (all + JIT)",
)
ax.plot(
    v2_all.sidelength,
    v2_all.runtime2,
    c=c_v2_all,
    **markers,
    label="v2 (all)",
)
oo = np.array([1, 1])
# for i, row in v1.iterrows():
#     ax.plot(
#         row.sidelength * oo,
#         [row.runtime - row.runtime_pm, row.runtime + row.runtime_pm],
#         c="k",
#     )
# for i, row in v2_pH.iterrows():
#     ax.plot(
#         row.sidelength * oo,
#         [row.runtime1 - row.runtime1_pm, row.runtime1 + row.runtime1_pm],
#         c="k",
#     )
#     ax.plot(
#         row.sidelength * oo,
#         [row.runtime2 - row.runtime2_pm, row.runtime2 + row.runtime2_pm],
#         c="k",
#     )
ax.set_yscale("log")
ax.set_ylabel("Run time / s")
ax.set_yticks([0.1, 1, 10, 100])
ax.set_yticklabels([0.1, 1, 10, 100])
ax.legend(ncols=2, fontsize=9)
ax.text(0, 1.03, "(a)", transform=ax.transAxes)

ax = axs[1]
dstyle = dict(
    dashes=(4, 1.5),
    lw=1.2,
)
ax.plot(
    v1.sidelength,
    v1.peak,
    c=c_v1,
    **markers,
    **dstyle,
    label="v1 (peak)",
)
ax.plot(
    v2_pH.sidelength,
    v2_pH.peak,
    c=c_v2_pH,
    **markers,
    **dstyle,
    label="v2 (pH, peak)",
)
ax.plot(
    v2_all.sidelength,
    v2_all.peak,
    c=c_v2_all,
    **markers,
    **dstyle,
    label="v2 (all, peak)",
)
ax.plot(
    v1.sidelength,
    v1.final,
    c=c_v1,
    **markers,
    label="v1 (final)",
)
ax.plot(
    v2_pH.sidelength,
    v2_pH.final,
    c=c_v2_pH,
    **markers,
    label="v2 (pH, final)",
)
ax.plot(
    v2_all.sidelength,
    v2_all.final,
    c=c_v2_all,
    **markers,
    label="v2 (all, final)",
)
ax.legend(ncols=2, loc="lower right", fontsize=9)
ax.text(0, 1.03, "(b)", transform=ax.transAxes)

# oo = np.array([1, 1])
# for i, row in v1.iterrows():
#     ax.plot(
#         row.sidelength * oo,
#         [row.runtime - row.runtime_pm, row.runtime + row.runtime_pm],
#         c="k",
#     )
# for i, row in v2_pH.iterrows():
#     ax.plot(
#         row.sidelength * oo,
#         [row.runtime1 - row.runtime1_pm, row.runtime1 + row.runtime1_pm],
#         c="k",
#     )
#     ax.plot(
#         row.sidelength * oo,
#         [row.runtime2 - row.runtime2_pm, row.runtime2 + row.runtime2_pm],
#         c="k",
#     )
ax.set_yscale("log")
ax.set_ylabel("Memory allocated / MB")
ax.set_yticks([1, 10, 100, 1000, 10000])
ax.set_yticklabels([1, 10, "10$^2$", "10$^3$", "10$^4$"])

for ax in axs:
    ax.set_xlim(0, 1200)
    ax.set_xlabel("√$N$")
    ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("tests/_plot_speed_osm.png")
