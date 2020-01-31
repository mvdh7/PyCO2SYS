import numpy as np
from PyCO2SYS import CO2SYS

ta = np.vstack(np.ones(10)*2200)
dic = np.ones(10)*2050
DICT, DATA, HEADERS, NICEHEADERS = CO2SYS(ta, dic, 1, 2, 35, 10, 10, 0, 0, 0, 0, 3, 10, 3)
# print(test)
