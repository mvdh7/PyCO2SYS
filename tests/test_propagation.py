# Test the propagation calculations against Monte-Carlo simulations
import PyCO2SYS as pyco2
import numpy as np

# First just par1, par2 and their combination
pars_true = np.array([2350, 2100, 8.1, 400, 400, 350, 1900, 12])
paru_true = np.array([2, 3, 0.001, 2, 2, 2.5, 4, 0.1])
partypes = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Other conditions
sal = 31.4
tempin = 22.2
tempout = 19.2
presin = 44.4
presout = 1234.5
si = 3.6
po4 = 1.2
nh3 = 0.4
h2s = 0.6
phscale = 3
k1k2c = 12
kso4c = 3
kfc = 2
buffers_mode = "auto"
args = (sal, tempin, tempout, presin, presout, si, po4, phscale, k1k2c, kso4c)
kwargs = {"NH3": nh3, "H2S": h2s, "KFCONSTANT": kfc, "buffers_mode": buffers_mode}


def get_compare(montecarlo, direct):
    """Robustly get percentage differences between Monte-Carlo and direct methods."""
    if direct == 0:
        assert montecarlo < 1e-12
        compare = np.array([0.0])
    else:
        compare = np.abs(100 * (montecarlo - direct) / direct)
    return compare


def compare_Kunc(p1, p2, Kstr, io):
    """Compare uncertainty propagation from the equilibrium constant internal overrides
    vs Monte-Carlo simulations.
    """
    mcsize = 10000
    Kunc_pct = 0.05
    co2d = pyco2.CO2SYS(
        np.full(mcsize, pars_true[p1]),
        pars_true[p2],
        partypes[p1],
        partypes[p2],
        *args,
        **kwargs
    )
    if io == "in":
        equilibria_in = {
            Kstr: np.random.normal(
                loc=co2d["{}{}put".format(Kstr, io)][0],
                scale=co2d["{}{}put".format(Kstr, io)][0] * Kunc_pct,
                size=mcsize,
            )
        }
        co2d_mcsim = pyco2.CO2SYS(
            np.full(mcsize, pars_true[p1]),
            pars_true[p2],
            partypes[p1],
            partypes[p2],
            *args,
            **kwargs,
            equilibria_in=equilibria_in
        )
    elif io == "out":
        equilibria_out = {
            Kstr: np.random.normal(
                loc=co2d["{}{}put".format(Kstr, io)][0],
                scale=co2d["{}{}put".format(Kstr, io)][0] * Kunc_pct,
                size=mcsize,
            )
        }
        co2d_mcsim = pyco2.CO2SYS(
            np.full(mcsize, pars_true[p1]),
            pars_true[p2],
            partypes[p1],
            partypes[p2],
            *args,
            **kwargs,
            equilibria_out=equilibria_out
        )
    testvar = "isoQ{}".format(io)
    testunc_Mcsim = np.std(co2d_mcsim[testvar])
    uncertainties, components = pyco2.uncertainty.propagate(
        co2d,
        [testvar],
        {"{}{}put".format(Kstr, io): Kunc_pct * co2d["{}{}put".format(Kstr, io)][0]},
    )
    comparison = get_compare(testunc_Mcsim, uncertainties[testvar][0])
    print(p1, p2, Kstr)
    if comparison > -1:
        print(testunc_Mcsim)
        print(uncertainties[testvar][0])
        print(comparison)
    assert (comparison < 4) or (uncertainties[testvar][0] < 1e-10)
    # ^ either the comparison is acceptable or the effect is so small that differences
    # due to pH solver tolerance make large percentage errors.  Both are acceptable.


def test_Kunc_in():
    for K in [
        "K1",
        "K2",
        "KW",
        "KP1",
        "KP2",
        "KP3",
        "KB",
        "KSi",
        "KSO4",
        "FugFac",
        "KNH3",
        "KH2S",
        "KF",
    ]:
        for p1p2 in ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)):
            compare_Kunc(*p1p2, K, "in")


def test_Kunc_out():
    for K in [
        "K1",
        "K2",
        "KW",
        "KP1",
        "KP2",
        "KP3",
        "KB",
        "KSi",
        "KSO4",
        "FugFac",
        "KNH3",
        "KH2S",
        "KF",
    ]:
        for p1p2 in ((1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)):
            compare_Kunc(*p1p2, K, "out")


def compare_par1par2(i, fixedpartype, uncertainties_in):
    fixedpar = partypes == fixedpartype
    par1s_true = pars_true[~fixedpar]
    par1u_true = paru_true[~fixedpar]
    par1types = partypes[~fixedpar]
    par2s_true = np.full_like(par1s_true, pars_true[fixedpar][0])
    par2u_true = np.full_like(par1u_true, paru_true[fixedpar][0])
    par2types = np.full_like(par1types, partypes[fixedpar][0])
    par1 = par1s_true[i]
    par1type = par1types[i]
    par2 = par2s_true[i]
    par2type = par2types[i]
    co2d = pyco2.CO2SYS(par1, par2, par1type, par2type, *args, **kwargs)
    # Propagate directly
    uncertainties_in = [uncertainties_in]
    uncertainties, components = pyco2.uncertainty.propagate(
        co2d, uncertainties_in, {"PAR1": par1u_true[i], "PAR2": par2u_true[i]}
    )
    # Estimate the same with Monte-Carlo simulation
    mcsize = (10000,)
    par1sim = np.random.normal(size=mcsize, loc=par1, scale=par1u_true[i])
    par2sim = np.random.normal(size=mcsize, loc=par2, scale=par2u_true[i])
    co2d_par1sim = pyco2.CO2SYS(par1sim, par2, par1type, par2type, *args, **kwargs)
    co2d_par2sim = pyco2.CO2SYS(par1, par2sim, par1type, par2type, *args, **kwargs)
    co2d_bothsim = pyco2.CO2SYS(par1sim, par2sim, par1type, par2type, *args, **kwargs)
    umc1 = np.std(co2d_par1sim[uncertainties_in[0]])
    umc2 = np.std(co2d_par2sim[uncertainties_in[0]])
    umcBoth = np.std(co2d_bothsim[uncertainties_in[0]])
    compare1 = get_compare(umc1, components[uncertainties_in[0]]["PAR1"])
    compare2 = get_compare(umc2, components[uncertainties_in[0]]["PAR2"])
    compareBoth = get_compare(umcBoth, uncertainties[uncertainties_in[0]])
    return compare1, compare2, compareBoth


# Check they're within tolerance
checktol = 3  # %


def check_par1par2(uncertainties_in):
    for j in range(1, 9):
        fixedpar = partypes == j
        par1types = partypes[~fixedpar]
        for i in range(7):
            ijcase = pyco2.solve.getIcase(par1types[i], j, checks=False)
            if ijcase not in [45, 48, 58]:
                print(ijcase)
                compare1, compare2, compareBoth = compare_par1par2(
                    i, j, uncertainties_in
                )
                if compare1 > 1 or compare2 > 1:
                    print(compare1)
                    print(compare2)
                    print(compareBoth)
                assert compare1 < checktol
                assert compare2 < checktol
                assert compareBoth < checktol


def test_par1par2():
    check_par1par2("pHoutNBS")
    check_par1par2("isoQout")


# test_par1par2()
