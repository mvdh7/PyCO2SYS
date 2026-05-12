# %%
import numpy as np
import plotly.graph_objects as go

import PyCO2SYS as pyco2

ta0 = np.vstack([2150, 2450])
acid = np.linspace(0, 4000, num=500)
ta = ta0 - acid

co2s = pyco2.sys(
    ta=ta,
    dic=2200 * np.exp(-acid / 10000),
    t=25,
    s=32.5,
)


# Convert pH into EMF
R = 8.3145
F = 96.4853321
emf0 = 600  # mV
emf = emf0 + np.log(10 ** -co2s["ph"]) * R * (co2s.t + 273.15) / F

tt = go.Scatter(
    x=acid[1:],
    # y=co2s["pH"][0],
    # y=10 ** -co2s["pH"][0],
    y=np.diff(emf[0]),
    mode="lines",
)
tt1 = go.Scatter(
    x=acid[1:],
    # y=co2s["pH"][1],
    # y=10 ** -co2s["pH"][1],
    y=np.diff(emf[1]),
    mode="lines",
)
fig = go.Figure([tt, tt1])
fig.show()

# %% Go back to alkalinity now
emf0 = 600

pH = -np.log10(np.exp((emf - emf0) * F / (R * (co2s.t + 273.15))))
co2r = pyco2.sys(
    ph=pH,
    dic=2200,
    s=32.5,
).solve("ta")
ta_all = co2r.ta + acid

tt2 = go.Scatter(
    x=acid,
    y=ta_all[0],
    mode="lines",
)
fig = go.Figure(tt2)
fig.show()
