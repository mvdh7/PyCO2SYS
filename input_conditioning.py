import PyCO2SYS
import numpy as np

loc = {
    'a': 123,
    'b': 383,
    'TEMPIN': np.vstack([25, 26, 27]),
}
args, ntps = PyCO2SYS.assemble.inputs(loc)
