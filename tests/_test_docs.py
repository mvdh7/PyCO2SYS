# %%
import PyCO2SYS as pyco2


co2s = pyco2.sys(t=[5, 10, 15]).set_u(t=0.2)
co2s = pyco2.sys(t=[5, 10, 15]).set_u(t=[0.1, 0.2, 0.3]).prop("pk1")
co2s = (
    pyco2.sys(t=[5, 10, 15])
    .set_u(
        t=[
            [0.1, 0.05, 0.05],
            [0.05, 0.2, 0.05],
            [0.05, 0.05, 0.3],
        ]
    )
    .prop("pk1")
)
co2s = (
    pyco2.sys(
        dic=2100,
        ta=2250,
        t=[5, 10, 15],
    )
    .set_u(t=0.1)
    .set_u_OEDG18()
).prop("pk1")

# Propagate uncertainties that were set with set_u
co2s.prop(["pH", "fCO2"])

# Access uncertainty results
uncert_fCO2 = co2s.u["fCO2"]
uncert_pH_due_to_dic = co2s.u.parts["pH"]["t"]

# You can also use dot notation and shortcuts here
uncert_fCO2 = co2s.u.fCO2
uncert_pH_due_to_dic = co2s.u.parts.pH.t
