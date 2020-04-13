from autograd import elementwise_grad as egrad
from autograd import numpy as np
from jax import grad, vmap

def whoop(varin):
    varsum = np.zeros(np.size(varin))
    i = 0
    while i < 10:
        varsum = varsum + varin + 0.5
        i += 1
    return varsum

varin = np.arange(10.0)    
varout = whoop(varin)
gradout = egrad(whoop)(varin)
jaxout = vmap(grad(whoop), in_axes=(0,))(varin)
