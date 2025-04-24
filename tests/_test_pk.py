# %%
import numpy as np

pk = 0.1
k = 10**-pk
lnk = np.log(k)
print(
    pk,
    k,
    lnk,
    -lnk / np.log(10),
)

# %%
pk = 8
a = 1.5
k = 10**-pk * a
print(
    pk,
    k,
    10 ** -(pk - np.log10(a)),
)
