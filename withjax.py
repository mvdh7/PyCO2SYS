import jax
import jax.numpy as np
import numpy as onp
from autograd import elementwise_grad


def egrad(g):
    def wrapped(x, *rest):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
        (x_bar,) = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped


x = onp.full((2, 5), 3.5)


def f(x):
    return x**2


df_jax = egrad(f)
df_ag = elementwise_grad(f)
df_vmap = jax.vmap(jax.vmap(jax.grad(f), 0, 0), 1, 1)
df_vmap_jit = jax.jit(jax.vmap(jax.vmap(jax.grad(f), 0, 0), 1, 1))

print(df_jax(x))
print(df_ag(x))
print(df_vmap(x))
print(df_vmap_jit(x))

# %timeit df_jax(x)
# 356 μs ± 4.14 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# %timeit df_ag(x)
# 16.8 μs ± 76.6 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

# %timeit df_vmap(x)
# 769 μs ± 23.2 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# %timeit df_vmap_jit(x)
# 2.76 μs ± 34.8 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
