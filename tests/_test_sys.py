# %%
from collections import UserDict

from PyCO2SYS import CO2System
from PyCO2SYS.engine import CO2System_ud

# class CO2SystemResult(UserDict):
#     def __init__(self, sys):
#         super().__init__()
#         self.sys = sys

#     def __getitem__(self, key):
#         return self.sys.solve(key)

#     # def keys(self):
#     #     return self.funcs.keys()


# def system(**kwargs):
#     if "data" in kwargs:
#         values = kwargs["data"].copy()
#     else:
#         values = {}
#     opts = {}
#     for k, v in kwargs.items():
#         if k.startswith("opt_"):
#             opts.update({k: v})
#         elif k == "data":
#             pass
#         else:
#             values.update({k: v})
#     sys = CO2System(values=values, opts=opts)
#     return CO2SystemResult(sys)


sys = CO2System_ud(
    dic=2100,
    alkalinity=2300,
    opt_k_carbonic=10,
)
# sys.solve(["pH", "fCO2"])
pH = CO2System_ud(
    dic=2100,
    alkalinity=2300,
    opt_k_carbonic=10,
)["pH"]
results = CO2System_ud(
    dic=2100,
    alkalinity=2300,
    opt_k_carbonic=10,
)[["pH", "fCO2"]]
