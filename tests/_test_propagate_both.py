import itertools
import numpy as np
import PyCO2SYS as pyco2

ks = [
    "k_CO2",
    "k_carbonic_1",
    "k_carbonic_2",
    "k_water",
    "k_borate",
    "k_bisulfate",
    "k_fluoride",
    "k_phosphoric_1",
    "k_phosphoric_2",
    "k_phosphoric_3",
    "k_silicate",
    "k_ammonia",
    "k_sulfide",
    "k_calcite",
    "k_aragonite",
]
pks = ["p{}".format(k) for k in ks]


def propagate_both_and_compare(k):
    # Propagate "both" uncertainties through PyCO2SYS and also get derivatives
    u_into = ["isocapnic_quotient_out"]
    u_from = {"{}_both".format(k): 0.5}
    grads_wrt = [k, "{}_out".format(k)]
    kwargs = dict(
        par1=2300,
        par2=8.1,
        par1_type=1,
        par2_type=3,
        temperature_out=0,
        pressure_out=10000,
        grads_of=u_into,
        grads_wrt=grads_wrt,
        uncertainty_into=u_into,
        uncertainty_from=u_from,
    )
    results = pyco2.sys(**kwargs)
    u_result__both = results["u_{}".format(u_into[0])]
    # Assemble uncertainty matrix with complete covariance and Jacobian
    u_mx = np.full((2, 2), u_from["{}_both".format(k)] ** 2)
    jac = np.array(
        [
            [
                results["d_{}__d_{}".format(of, wrt)]
                for of, wrt in itertools.product(u_into, grads_wrt)
            ]
        ]
    )
    u_result__jac = np.sqrt((jac @ u_mx @ jac.T)[0][0])
    # Manual increment
    if k.startswith("p"):
        kk = k[1:]
        da = 1e-6
        kwargs[kk] = 10.0 ** -(-np.log10(results[kk]) + da)
        kwargs["{}_out".format(kk)] = 10.0 ** -(
            -np.log10(results["{}_out".format(kk)]) + da
        )
    else:
        kk = k
        da__f = 1e-6
        da = da__f * results[kk]
        kwargs[kk] = results[kk] + da
        kwargs["{}_out".format(kk)] = results["{}_out".format(kk)] + da
    results_manual = pyco2.sys(**kwargs)
    u_result__manual = np.abs(results_manual[u_into[0]] - results[u_into[0]])
    u_result__manual *= u_from["{}_both".format(k)]
    u_result__manual /= da
    return u_result__both, u_result__jac, u_result__manual


def test_propagate_both():
    """Does the Jacobian method agree with using '_both' for propagation and
    also with manual increments?
    """
    for k in ks:
        b, j, m = propagate_both_and_compare(k)
        assert np.isclose(b, j, rtol=1e-3, atol=1e-6), "Failed on {}".format(k)
        assert np.isclose(b, m, rtol=1e-3, atol=1e-6), "Failed on {}".format(k)
        assert np.isclose(j, m, rtol=1e-3, atol=1e-6), "Failed on {}".format(k)
    for pk in pks:
        b, j, m = propagate_both_and_compare(pk)
        assert np.isclose(b, j, rtol=1e-3, atol=1e-6), "Failed on {}".format(pk)
        assert np.isclose(b, m, rtol=1e-3, atol=1e-6), "Failed on {}".format(pk)
        assert np.isclose(j, m, rtol=1e-3, atol=1e-6), "Failed on {}".format(pk)


# test_propagate_both()
