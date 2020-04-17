from autograd import elementwise_grad as egrad
from jax import numpy as np
from jax import grad, vmap

def whoop(varin):
    varsum = np.sum(np.sqrt(varin))
    i = 0
    while i < 10:
        varsum = varsum*varin[1] + varin[0] + 0.5
        i += 1
    return varsum

varin = np.arange(10.0)    
varout = whoop(varin)
# gradout = egrad(whoop)(varin)
jaxout = grad(whoop)(np.array([2.0, 2.5]))

# def sum_logistic(x):
#   return np.sum(1.0 / (1.0 + np.exp(-x)))

# x_small = np.arange(3.)
# derivative_fn = grad(sum_logistic)
# print(derivative_fn(x_small))
