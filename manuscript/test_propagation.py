"""Test the propagation calculations against Monte-Carlo simulations."""
import copy, itertools
import PyCO2SYS as pyco2, numpy as np

# Initialise random number generator
rng = np.random.default_rng(29)  # seeded for reproducibility

# First just par1, par2 and their combination
pars_true = np.array([2350, 2100, 8.1, 400, 400, 350, 1900, 12])
paru_true = np.array([2, 3, 0.001, 2, 2, 2.5, 4, 0.1])
partypes = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Other conditions
kwargs = {
    "salinity": 31.4,
    "temperature": 22.2,
    "temperature_out": 19.2,
    "pressure": 44.4,
    "pressure_out": 1234.5,
    "total_silicate": 3.6,
    "total_phosphate": 1.2,
    "total_ammonia": 0.4,
    "total_sulfide": 0.6,
    "opt_pH_scale": 3,
    "opt_k_carbonic": 12,
    "opt_k_bisulfate": 1,
    "opt_total_borate": 1,
    "opt_k_fluoride": 2,
    "opts_buffers_mode": 1,
    "total_alpha": 5.0,
    "total_beta": 10.0,
}


def get_compare(montecarlo, direct):
    """Robustly get percentage differences between Monte-Carlo and direct methods."""
    if direct < 1e-10:
        assert montecarlo < 1e-10
        compare = 0.0
    else:
        compare = np.abs(100 * (montecarlo - direct) / direct)
    return compare


def compare_k_propagation(
    p1, p2, k_constant, test_vars, n_simulations=10000, k_uncertainty_percent=0.1
):
    """Compare uncertainty propagation from the equilibrium constant internal overrides
    vs Monte-Carlo simulations.
    """
    # Solve MCS
    par_args = dict(
        par1=pars_true[p1],
        par2=pars_true[p2],
        par1_type=partypes[p1],
        par2_type=partypes[p2],
    )
    results = pyco2.sys(**par_args, **kwargs)
    # Simulate uncertainty in the selected k_constant
    k_value = results[k_constant]
    k_uncertainty = k_value * k_uncertainty_percent / 100
    kwargs_u = copy.deepcopy(kwargs)
    kwargs_u.update(
        {k_constant: rng.normal(loc=k_value, scale=k_uncertainty, size=n_simulations)}
    )
    # Solve MCS with uncertain k_constant
    results_u = pyco2.sys(**par_args, **kwargs_u)
    k_uncertainty__simulated = {k: np.std(results_u[k]) for k in test_vars}
    # Propagate uncertainty
    k_uncertainty__direct = pyco2.uncertainty.propagate_nd(
        results, test_vars, {k_constant: k_uncertainty}, dx=1e-6, **kwargs
    )[0]
    return k_uncertainty__simulated, k_uncertainty__direct


def test_k_constant_single_propagation():
    """Do uncertainties in individual equilibrium constants propagate equivalently using
    the direct calculation approach and in Monte-Carlo simulations?
    """
    k_constants = [k for k in pyco2.engine.nd.input_floats if k.startswith("k_")]
    test_vars = [
        "isocapnic_quotient",
        "pH",
        "saturation_calcite",
        "revelle_factor",
        "fCO2",
    ]
    test_vars += [v + "_out" for v in test_vars]
    test_vars.append("alkalinity")
    p1 = 0
    for k_constant, p2 in itertools.product(k_constants, range(1, 8)):
        k_uncertainty__simulated, k_uncertainty__direct = compare_k_propagation(
            p1, p2, k_constant, test_vars, k_uncertainty_percent=0.01
        )
        for v in test_vars:
            compare = get_compare(k_uncertainty__simulated[v], k_uncertainty__direct[v])
            print(k_constant, compare, v)
            assert compare < 2
        # ^ either the comparison is acceptable or the effect is so small that differences
        # due to pH solver tolerance make large percentage errors.  Both are acceptable.


# test_k_constant_single_propagation()


def compare_par1par2(i, fixedpartype, uncertainties_in):
    """Do uncertainties in par1, par2, and both together, propagate equivalently between
    the direct calculation and Monte-Carlo simulations?
    """
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
    co2d = pyco2.sys(par1, par2, par1type, par2type, **kwargs)
    # Propagate directly
    uncertainties_in = [uncertainties_in]
    uncertainties, components = pyco2.uncertainty.propagate_nd(
        co2d, uncertainties_in, {"par1": par1u_true[i], "par2": par2u_true[i]}, **kwargs
    )
    # Estimate the same with Monte-Carlo simulation
    mcsize = (10000,)
    par1sim = rng.normal(size=mcsize, loc=par1, scale=par1u_true[i])
    par2sim = rng.normal(size=mcsize, loc=par2, scale=par2u_true[i])
    co2d_par1sim = pyco2.sys(par1sim, par2, par1type, par2type, **kwargs)
    co2d_par2sim = pyco2.sys(par1, par2sim, par1type, par2type, **kwargs)
    co2d_bothsim = pyco2.sys(par1sim, par2sim, par1type, par2type, **kwargs)
    umc1 = np.std(co2d_par1sim[uncertainties_in[0]])
    umc2 = np.std(co2d_par2sim[uncertainties_in[0]])
    umcBoth = np.std(co2d_bothsim[uncertainties_in[0]])
    compare1 = get_compare(umc1, components[uncertainties_in[0]]["par1"])
    compare2 = get_compare(umc2, components[uncertainties_in[0]]["par2"])
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
            if ijcase not in [405, 408, 508]:
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
    check_par1par2("pH_nbs_out")
    check_par1par2("isocapnic_quotient_out")


# test_par1par2()
