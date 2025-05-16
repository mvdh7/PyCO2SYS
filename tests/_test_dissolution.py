# %%
import numpy as np
import plotly.graph_objects as go

x = np.linspace(0, 10, num=1000)
offset = 5
y1 = offset + 2 * np.exp(-x * 0.5)
y2 = offset + 2 * np.exp(-x)
y3 = offset + 2 * np.exp(-x * 2)

s = []
si = []
se = []
for y in [y1, y2, y3]:
    s.append(go.Scatter(x=x, y=y, mode="lines"))
    si.append(go.Scatter(x=1 / np.sqrt(x), y=y, mode="lines"))
    se.append(go.Scatter(x=x, y=np.log((y - 5) / 2), mode="lines"))
fig = go.Figure(se)
fig.show()
