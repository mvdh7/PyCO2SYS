# %%
from collections import UserDict


class DotDict(UserDict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            try:
                return self.data[attr]
            except KeyError:
                raise AttributeError(attr)


class Uncertainty(DotDict):
    def __init__(self):
        super().__init__()
        self.assigned = DotDict()
        self.components = DotDict()

    def assign(self, **uncertainties):
        for k, v in uncertainties.items():
            self.assigned[k] = v
        # then re-propagate


uncert = Uncertainty()
uncert.assign(dic=2, temperature=0.1)

# Stuff below would be done by propagate()
uncert.components["pH"] = DotDict(dic=0.03, temperature=0.04)
uncert["pH"] = 0.05
