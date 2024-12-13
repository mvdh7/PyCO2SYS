# %%
from collections import UserDict


class CO2SystemResult(UserDict):
    def __init__(self):
        super().__init__()
        self.funcs = {
            "a": lambda: 1,
            "b": lambda: 2,
            "c": lambda: 3,
        }

    # def __missing__(self, key):
    #     return "hi"

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.funcs[key]()
        return self.data[key]

    def keys(self):
        return self.funcs.keys()

    # def values(self):
    #     ValuesView

    # def items(self):
    #     return {key: self[key] for key in self.keys()}


c = CO2SystemResult()
print(c)
print(c["a"])
print(c)
