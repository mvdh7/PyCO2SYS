import PyCO2SYS as pyco2

args = (2100, 2450, 2, 1)
kwargs = dict(total_alpha=100.0, k_alpha=1e-10, total_beta=500, k_beta=1e-7)
res = pyco2.sys(*args, **kwargs)
print(res['pH_total'])
print(res['beta_dic'] * 1e6)
pyco2.solve.get.TAfromTCpH = lambda a, b, c, d: a + b
res = pyco2.sys(*args, **kwargs)
print(res['pH_total'])
print(res['beta_dic'] * 1e6)
pyco2.solve.get.TAfromTCpH = pyco2.solve.get.TAfromTCpH_fixed
res = pyco2.sys(*args, **kwargs)
print(res['pH_total'])
print(res['beta_dic'] * 1e6)
