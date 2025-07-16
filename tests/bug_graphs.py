# %%
import PyCO2SYS as pyco2

co2s = (
    pyco2.sys(
        dic=2100,
        ph=8.1,
        salinity=500,
        tsi=5,
        pk1=8,
        total_borate=300,
    )
    .set_u(dic=2, ph=0.005, **pyco2.uncertainty_OEDG18)
    # .solve("ta")
    .prop("ta")
)
co2s.plot_graph(prog_graphviz="dot", show_unknown=False, mode="state")
# BUG ^ pKs coming up orange in state graph?
# co2s.plot_graph(prog_graphviz="dot", show_unknown=False, mode="valid")
# # BUG ^ ionic_strength KeyError
