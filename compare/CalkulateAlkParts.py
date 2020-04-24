import PyCO2SYS as pyc

HCO3, CO3, BAlk, OH, PAlk, SiAlk, NH3Alk, H2SAlk, Hfree, HSO4, HF = \
    pyc.solve.AlkParts(8.0, 2100e-6, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0,
                          400e-6, 10e-6, 2000e-6, 10e-6, 50e-6, 0, 0)
