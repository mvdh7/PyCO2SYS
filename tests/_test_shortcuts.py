# %%
from collections import UserDict


class Shortcuts(UserDict):
    def __init__(self, keys, suffixes=None, no_suffix=None):
        if isinstance(keys, str):
            keys = [keys]
        super().__init__(**{k.lower(): k for k in keys})
        if suffixes is not None:
            if isinstance(suffixes, str):
                suffixes = [suffixes]
            if no_suffix is None or isinstance(no_suffix, str):
                no_suffix = [no_suffix]
            for k in keys:
                if k not in no_suffix:
                    kl = k.lower()
                    for s in suffixes:
                        self.data[kl + s] = self.data[kl] + s

    def __getitem__(self, key):
        return self.data[key.lower()]


shortcuts = Shortcuts(["pH", "dic"], suffixes=["__pre"], no_suffix="dic")

print(shortcuts["ph"], shortcuts["pH"], shortcuts["PH"])
print(shortcuts)
