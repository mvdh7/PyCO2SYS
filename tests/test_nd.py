import PyCO2SYS as pyco2
from autograd import numpy as np


par1 = np.random.normal(size=(5, 4, 3))
par2 = [10, 11, 12]
par3 = np.random.normal(size=(5, 4, 3))
par4 = np.random.normal(size=(3, 4, 5))
par5 = 12.2
par6 = "test"
par7 = np.random.normal(size=(5, 1, 3))
par8 = ["test"]

inputs = {
    "PAR1": par1,
    "PAR2": par2,
    "par3": par3,
    # "par4": par4,  # this one should make it fail
    "par5": par5,
    "par6": par6,
    "par7": par7,
    "par8": par8,
}

cond = pyco2.engine.nd.condition(inputs, to_shape=(10, 5, 4, 3))
