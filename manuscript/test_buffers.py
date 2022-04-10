import copy
import PyCO2SYS as pyco2, numpy as np

# Seed random number generator for reproducibility
rng = np.random.default_rng(7)

# Generate inputs
npts = 1000
alkalinity = rng.normal(loc=2300, scale=500, size=npts)
dic = rng.normal(loc=2100, scale=500, size=npts)
kwargs = dict(par1=alkalinity, par2=dic, par1_type=1, par2_type=2)

# Solve the marine carbonate system
r_explicit = pyco2.sys(**kwargs, buffers_mode=2)
r_automatic = pyco2.sys(**kwargs, buffers_mode=1)

# Compare the buffers
buffers = [
    "beta_alk",
    "beta_dic",
    "gamma_alk",
    "gamma_dic",
    "omega_alk",
    "omega_dic",
    "psi",
    "isocapnic_quotient",
    "revelle_factor",
]
buffers_difference = {}
buffers_percent = {}
for buffer in buffers:
    buffers_difference[buffer] = np.max(
        np.abs(r_explicit[buffer] - r_automatic[buffer])
    )
    buffers_percent[buffer] = np.max(
        np.abs(100 * (r_explicit[buffer] - r_automatic[buffer]) / r_automatic[buffer])
    )


def test_buffers():
    """Do the explicit and automatic calculations agree with non-zero TSO4 and TF?"""
    for buffer in buffers:
        if buffer == "revelle_factor":
            atol = 1e-6
        else:
            atol = 0.02
        assert buffers_percent_allzero[buffer] < atol


# Now set total sulfate and fluoride to zero
kwargs_allzero = copy.deepcopy(kwargs)
kwargs_allzero.update(dict(total_sulfate=0, total_fluoride=0))

# Solve the marine carbonate system
r_explicit_allzero = pyco2.sys(**kwargs_allzero, buffers_mode=2)
r_automatic_allzero = pyco2.sys(**kwargs_allzero, buffers_mode=1)

# Compare the buffers
buffers_difference_allzero = {}
buffers_percent_allzero = {}
for buffer in buffers:
    buffers_difference_allzero[buffer] = np.max(
        np.abs(r_explicit_allzero[buffer] - r_automatic_allzero[buffer])
    )
    buffers_percent_allzero[buffer] = np.max(
        np.abs(
            100
            * (r_explicit_allzero[buffer] - r_automatic_allzero[buffer])
            / r_automatic_allzero[buffer]
        )
    )


def test_buffers_allzero():
    """Do the explicit and automatic calculations agree better with zero TSO4 and TF?"""
    for buffer in buffers:
        if buffer == "revelle_factor":
            atol = 1e-6
        elif buffer in ["psi", "isocapnic_quotient"]:
            atol = 1e-8
        else:
            atol = 1e-11
        assert buffers_percent_allzero[buffer] < atol
