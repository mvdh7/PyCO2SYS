import PyCO2SYS as pyco2, numpy as np


# Run PyCO2SYS to begin with some silicate
args = (2250, [2100, 8.1, 400, 150, 500, 10000], 1, [2, 3, 4, 5, 2, 2])
opt_pH_scale = 3
r0 = pyco2.sys(*args, total_silicate=100, opt_pH_scale=3)

# Add alpha instead of silicate
r1 = pyco2.sys(
    *args,
    total_silicate=0,
    total_alpha=r0["total_silicate"],
    k_alpha=r0["k_silicate"],
    opt_pH_scale=opt_pH_scale,
)

# Add beta instead of fluoride - note HF used in pK scale conversions so you have to
# provide them all ready-converted
r2 = pyco2.sys(
    *args,
    total_silicate=r0["total_silicate"],
    total_fluoride=0,
    total_beta=r0["total_fluoride"],
    k_beta=r0["k_fluoride"],
    k_carbonic_1=r0["k_carbonic_1"],
    k_carbonic_2=r0["k_carbonic_2"],
    k_water=r0["k_water"],
    k_borate=r0["k_borate"],
    k_bisulfate=r0["k_bisulfate"],
    k_silicate=r0["k_silicate"],
    opt_pH_scale=opt_pH_scale,
)

# Add both alpha (silicate) and beta (borate)
r3 = pyco2.sys(
    *args,
    total_silicate=0,
    total_alpha=r0["total_silicate"],
    k_alpha=r0["k_silicate"],
    total_borate=0,
    total_beta=r0["total_borate"],
    k_beta=r0["k_borate"],
    opt_pH_scale=opt_pH_scale,
)

# Add ammonia and alpha-as-ammonia
total_ammonia = 101.5
r4_ammonia = pyco2.sys(*args, total_ammonia=total_ammonia, opt_pH_scale=opt_pH_scale)
r4_alpha = pyco2.sys(
    *args,
    total_alpha=total_ammonia,
    k_alpha=r4_ammonia["k_ammonia"],
    opt_pH_scale=opt_pH_scale,
)


def test_silicate_alpha():
    assert np.allclose(r0["alkalinity_silicate"], r1["alkalinity_alpha"])
    assert np.allclose(r0["H3SiO4"], r1["alpha"])
    assert np.allclose(r0["H4SiO4"], r1["alphaH"])


def test_fluoride_beta():
    assert np.allclose(r0["HF"], -r2["alkalinity_beta"])


def test_alpha_beta():
    assert np.allclose(r0["alkalinity_silicate"], r3["alkalinity_alpha"])
    assert np.allclose(r0["alkalinity_borate"], r3["alkalinity_beta"])


def test_ammonia_alpha():
    assert np.allclose(r4_ammonia["alkalinity_ammonia"], r4_alpha["alkalinity_alpha"])
    assert np.allclose(r4_ammonia["beta_alk"], r4_alpha["beta_alk"])


# test_silicate_alpha()
# test_fluoride_beta()
# test_alpha_beta()
# test_ammonia_alpha()
