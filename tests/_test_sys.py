# %%
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

import PyCO2SYS as pyco2
from PyCO2SYS import convert

co2s = (
    pyco2.sys(dic=2100, alkalinity=2250)
    .solve(["pH", "saturation_calcite", "saturation_aragonite"])
    .set_uncertainty(**pyco2.uncertainty.pKs_OEDG18, dic=2)
    .propagate("pH")
).set_uncertainty(total_borate__f=0.02)
co2s.uncertainty

# %%
data = pd.DataFrame({"dic": [2000, 2100], "pH": 8.1, "temperature_2": "1"})
co2s = pyco2.sys(data=data[data.dic == 2000]).adjust(
    temperature=data.temperature_2[data.dic == 2000]
)

# %%

dic = xr.DataArray(np.ones((30, 5)) * 2000, dims=("lat", "lon"))
alkalinity = xr.DataArray(np.ones((30, 5)) * 2200, dims=("lat", "lon"))
temperature = xr.DataArray(np.ones((30, 5)) * 25, dims=("lat", "lon"))
temperature_2 = xr.DataArray(np.ones((5, 30)) * 0, dims=("lon", "lat"))
data = xr.Dataset(
    {
        "dic": dic,
        "alkalinity": alkalinity,
        "temperature": temperature,
        "temperature_2": temperature_2,
    }
)

co2s = pyco2.sys(data=data).solve("pH")
co2sa = co2s.adjust(temperature=temperature_2)

# %%

co2s = pyco2.sys(
    data=pd.DataFrame(
        {
            "dich": [2100, 2200],
            "pHx": 8.1,
        }
    ),
    dic="dich",
    pH="pHx",
)
co2s.solve(["CO3"])
print(co2s.CO3)

# %%
co2s = pyco2.sys(
    dic=2100,
    ph=8.1,
)

# %%
co2s.solve(["CO3"], store_steps=1)
node_size = 2100
co2s.plot_graph(
    show_unknown=False,
    keep_unknown="HCO3",
    show_isolated=False,
    show_tsp=True,
    prog_graphviz="dot",
    # root_graphviz="dic",
    # args_graphviz="-Granksep=1.0 -Gnodesep=1.0",
    # nx_layout=nx.arf_layout,
    # nx_args=("salinity",),
    skip_nodes=["gas_constant", "pressure"],
    node_kwargs={"alpha": 0.7, "node_size": node_size},
    edge_kwargs={"arrowstyle": "-|>", "alpha": 1, "node_size": node_size},
)
# (
#     osage,
#     gc,
#     sccmap,
#     gvpr,
#     ccomps,
#     sfdp,
#     patchwork,
#     twopi,
#     nop,
#     circo,
#     dot,
#     gvcolor,
#     acyclic,
#     neato,
#     unflatten,
#     tred,
#     fdp,
# )

# %%

# %% Draw the graph
nx.draw(
    co2s.graph,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=500,
    font_size=10,
)
# plt.show()

# %%

temperature = np.arange(10, 20)
salinity = np.vstack(np.arange(31, 35))

co2s = pyco2.sys(
    alkalinity=2250,
    dic=2100,
    temperature=temperature,
    salinity=salinity,
)
co2s.solve("pH")

# %%
co2s = pyco2.sys(
    alkalinity=np.arange(2250, 2260),
    pCO2=400,
    temperature=np.arange(10, 20),
    salinity=np.arange(30, 40),
    pressure=10,
    total_silicate=100,
)
co2s.solve("pH", store_steps=0)

# TODO what if pH and pCO2 were measured discretely but at different temperatures
# from each other?!?!?!??!! --- you need to first convert pCO2 to the pH temperature?

co2adj = co2s.adjust(temperature=30, pressure=1000)

# %% One parameter
co2s = pyco2.sys(pH=8.1)
co2s.solve()

# %%
co2s = pyco2.sys(pH=8.1)
co2s["pH_free"]
co2s.solve(["pH_total", "pH_free"])

# %%
co2s = pyco2.sys(
    fCO2=400,
    temperature=10,
)
co2sa = co2s.adjust(temperature=25)

# Uncertainties
co2s = pyco2.sys(
    alkalinity=np.arange(2250, 2260),
    pCO2=400,
    temperature=np.arange(10, 20),
    salinity=np.arange(30, 40),
    pressure=10,
    total_silicate=100,
    opt_k_carbonic=10,
)
co2s.propagate("pH", {"alkalinity": 2, "pCO2": 1})

# %%
co2s.plot_graph(show_unknown=False)

# %%
kwargs = {}
a = np.array([[0, 1, 2], [3, 4, 5]])
b = a.transpose()
xa = xr.DataArray(a, dims=("a_row", "a_col"))
xb = xr.DataArray(b, dims=("b_row", "b_col"))
data = xr.Dataset({"xa": xa, "xb": xb})
xr_dims = list(data.sizes.keys())
xr_shape = list(data.sizes.values())
for k, v in data.items():
    # # Version 1: rubbish
    # ndims = []
    # for d in xr_dims:
    #     if d in v.sizes:
    #         ndims.append(v.sizes[d])
    #     else:
    #         ndims.append(1)
    # # Version 2: works if all dims present but not with extras
    # vdata = np.expand_dims(v.data, list(range(len(v.sizes), len(xr_dims))))
    # order = []
    # for d in v.sizes:
    #     order.append(xr_dims.index(d))
    # for x in range(len(v.sizes), len(xr_dims)):
    #     order.append(x + 1)
    # kwargs[k] = np.transpose(vdata, order)
    # Version 3: works for tests so far; need to try more complicated shapes
    # (mostly, more incompatible axes)
    v_dims = list(v.sizes)
    v_data = v.data
    move_from = []
    extra_dims = 0
    for i, d in enumerate(xr_dims):
        if d in v_dims:
            move_from.append(v_dims.index(d))
        else:
            move_from.append(len(v_dims) + extra_dims)
            v_data = np.expand_dims(v_data, -1)
            extra_dims += 1
    kwargs[k] = np.moveaxis(v_data, move_from, range(len(xr_dims)))

# %%
from PyCO2SYS.engine import da_to_array

na = da_to_array(xa, xr_dims)
nb = da_to_array(xb, xr_dims)
