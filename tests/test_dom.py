from autograd import numpy as np
from autograd import elementwise_grad as egrad

# Bold values for fulvic/humic acids from Milne et al. (2001) Table 4
nd_fulvic = {
    "b": 0.57,
    "QmaxH1": 5.88,
    "logKH1": 2.34,
    "m1": 0.38,
    "QmaxH2": 1.86,
    "logKH2": 8.60,
    "m2": 0.53,
}
nd_humic = {
    "b": 0.49,
    "QmaxH1": 3.15,
    "logKH1": 2.93,
    "m1": 0.50,
    "QmaxH2": 2.55,
    "logKH2": 8.00,
    "m2": 0.26,
}

# Test values
temperature_K = 298.15
pH = np.linspace(-2, 14, num=100)
pHstep = np.mean(np.diff(pH))
c_H = 10.0**-pH
c_OH = 1e-14 / c_H
c_Na = np.full_like(c_H, 0.001)
c_Cl = np.full_like(c_H, 0.001)
c_ions = np.array([c_H, c_OH, c_Na, c_Cl]).transpose()
z_ions = np.array([1, -1, 1, -1])
ionic_strength = 0.5 * np.sum(c_ions * z_ions**2, axis=1)


def nica(c_HD, nd):
    """Total amount of protons bound to DOM (Q_H) using the NICA equation, following
    Milne et al. (2001) eq. (1).
    """
    het1 = (10.0 ** nd["logKH1"] * c_HD) ** nd["m1"]
    Q_H1 = nd["QmaxH1"] * het1 / (1 + het1)
    het2 = (10.0 ** nd["logKH2"] * c_HD) ** nd["m2"]
    Q_H2 = nd["QmaxH2"] * het2 / (1 + het2)
    return Q_H1 + Q_H2


def nica_log10_chi(log10_chi, c_H, nd):
    chi = 10.0**log10_chi
    c_HD = chi * c_H
    return nica(c_HD, nd)


def nica_charge(c_HD, nd):
    """Charge on DOM from the NICA equation."""
    return nica(c_HD, nd) - nd["QmaxH1"] - nd["QmaxH2"]


def donnan_volume(ionic_strength, nd):
    """Donnan gel volume in L/kg-DOM, following Milne et al. (2001) eq. (2)."""
    logVD = nd["b"] * (1 - np.log10(ionic_strength)) - 1
    return 10**logVD


def donnan_charge(chi, c_ions, z_ions, ionic_strength, nd):
    """Calculate charge balance of Donnan gel."""
    V_D = donnan_volume(ionic_strength, nd)
    total_charge = np.sum(z_ions * np.vstack(chi) ** z_ions * c_ions, axis=1)
    return V_D * total_charge


def charge_balance(log10_chi, c_ions, z_ions, ionic_strength, nd):
    """Calculate overall charge balance to solve for log10(chi)."""
    chi = 10.0**log10_chi
    c_HD = chi * c_ions[:, 0]  # BRITTLE --- assumes first ion column is H!
    return nica_charge(c_HD, nd) - donnan_charge(
        chi, c_ions, z_ions, ionic_strength, nd
    )


def psi_to_chi(psi, k_boltzmann, temperature_K):
    return np.exp(-psi / (k_boltzmann * temperature_K))


def chi_to_psi(chi, k_boltzmann, temperature_K):
    return -np.log(chi) * k_boltzmann * temperature_K


def charge_balance_psi(psi, c_ions, z_ions, ionic_strength, temperature_K, nd):
    """Calculate overall charge balance to solve for psi."""
    k_boltzmann = 1.380649e-23  # m**2 * kg / (s * K)
    chi = psi_to_chi(psi, k_boltzmann, temperature_K)
    return charge_balance(chi, c_ions, z_ions, ionic_strength, nd)


def charge_balance_grad(log10_chi, c_ions, z_ions, ionic_strength, nd):
    """Calculate derivative of charge balance condition w.r.t. log10(chi)."""
    return egrad(charge_balance)(log10_chi, c_ions, z_ions, ionic_strength, nd)


def quick_solve_chi(c_ions, z_ions, ionic_strength, nd):
    chi_quick = (nd["QmaxH1"] + nd["QmaxH2"]) / (
        nd["QmaxH1"] * 10.0 ** nd["logKH1"] * c_ions[:, 0]
        + nd["QmaxH2"] * 10.0 ** nd["logKH2"] * c_ions[:, 0]
        - donnan_volume(ionic_strength, nd) * np.sum(c_ions * z_ions, axis=1)
    )
    return np.log10(chi_quick)


def solve_chi(c_ions, z_ions, ionic_strength, nd):
    """Solve charge balance for log10(chi) with Newton-Raphson."""
    # log10_chi = np.full_like(c_ions[:, 0], -1.0)  # first guess
    log10_chi = quick_solve_chi(c_ions, z_ions, ionic_strength, nd)  # first guess
    max_step = 0.01
    for i in range(1000):
        log10_chi_delta = -(
            charge_balance(log10_chi, c_ions, z_ions, ionic_strength, nd)
            / charge_balance_grad(log10_chi, c_ions, z_ions, ionic_strength, nd)
        )
        log10_chi_delta = np.where(
            np.abs(log10_chi_delta) > max_step,
            max_step * np.sign(log10_chi_delta),
            log10_chi_delta,
        )
        log10_chi = log10_chi + log10_chi_delta
    return log10_chi


nd = nd_humic
log10_chi = solve_chi(c_ions, z_ions, ionic_strength, nd)
log10_chi_quick = quick_solve_chi(c_ions, z_ions, ionic_strength, nd)
Q_H = nica_log10_chi(log10_chi, c_H, nd)

#%% Quick viz
from matplotlib import pyplot as plt

fix, axs = plt.subplots(nrows=3, ncols=2, dpi=300, figsize=(12, 15))
ax = axs[0, 0]
ax.plot(pH, Q_H)
Q_H_constantchi = nica_log10_chi(1, c_H, nd_fulvic)
ax.plot(pH, Q_H_constantchi)
ax = axs[1, 0]
ax.plot(pH[1:], -np.diff(Q_H) / pHstep)
ax.plot(pH[1:], -np.diff(Q_H_constantchi) / pHstep)
ax = axs[0, 1]
ax.axhline(0, c="k", lw=0.8)
ax.plot(pH, charge_balance(log10_chi, c_ions, z_ions, ionic_strength, nd))
ax = axs[1, 1]
l10 = np.linspace(-3, 3, num=100)
i = 40
ax.plot(l10, charge_balance(l10, np.array([c_ions[i]]), z_ions, ionic_strength[i], nd))
ax.axhline(0, c="k", lw=0.8)
ax.set_title(pH[i])
ax = axs[2, 0]
ax.plot(pH, log10_chi)
ax.plot(pH, log10_chi_quick)

#%% https://www.wolframalpha.com/input?i=f%28x%29%3Dax%5E2+%2B+bx+%2B+c+%2B+sqrt%28dx%29
x = (
    -b / (2 * a)
    - (1 / 2)
    * np.sqrt(
        b**2 / a**2
        - (2(b**2 + 2 * a * c)) / (3 * a**2)
        + (
            2 * b**6
            - 24 * a * c * b**4
            + 18 * a * d * b**3
            + 96 * a**2 * c**2 * b**2
            - 72 * a**2 * c * d * b
            - 128 * a**3 * c**3
            + 27 * a**2 * d**2
            + np.sqrt(
                -6912 * c**3 * d**2 * a**5
                + 729 * d**4 * a**4
                - 3888 * b * c * d**3 * a**4
                + 3456 * b**2 * c**2 * d**2 * a**4
                + 108 * b**3 * d**3 * a**3
                - 432 * b**4 * c * d**2 * a**3
            )
        )
        ** (1 / 3)
        / (3 * 2 ** (1 / 3) * a**2)
        + (
            2 ** (1 / 3)
            * (b**4 - 8 * a * c * b**2 + 6 * a * d * b + 16 * a**2 * c**2)
        )
        / (
            3
            * a**2
            * (
                2 * b**6
                - 24 * a * c * b**4
                + 18 * a * d * b**3
                + 96 * a**2 * c**2 * b**2
                - 72 * a**2 * c * d * b
                - 128 * a**3 * c**3
                + 27 * a**2 * d**2
                + np.sqrt(
                    -6912 * c**3 * d**2 * a**5
                    + 729 * d**4 * a**4
                    - 3888 * b * c * d**3 * a**4
                    + 3456 * b**2 * c**2 * d**2 * a**4
                    + 108 * b**3 * d**3 * a**3
                    - 432 * b**4 * c * d**2 * a**3
                )
            )
            ** (1 / 3)
        )
    )
    - (1 / 2)
    * np.sqrt(
        (2 * b**2) / a**2
        - (4 * (b**2 + 2 * a * c)) / (3 * a**2)
        - (
            2 * b**6
            - 24 * a * c * b**4
            + 18 * a * d * b**3
            + 96 * a**2 * c**2 * b**2
            - 72 * a**2 * c * d * b
            - 128 * a**3 * c**3
            + 27 * a**2 * d**2
            + np.sqrt(
                -6912 * c**3 * d**2 * a**5
                + 729 * d**4 * a**4
                - 3888 * b * c * d**3 * a**4
                + 3456 * b**2 * c**2 * d**2 * a**4
                + 108 * b**3 * d**3 * a**3
                - 432 * b**4 * c * d**2 * a**3
            )
        )
        ** (1 / 3)
        / (3 * 2 ** (1 / 3) * a**2)
        - (
            2 ** (1 / 3)
            * (b**4 - 8 * a * c * b**2 + 6 * a * d * b + 16 * a**2 * c**2)
        )
        / (
            3
            * a**2
            * (
                2 * b**6
                - 24 * a * c * b**4
                + 18 * a * d * b**3
                + 96 * a**2 * c**2 * b**2
                - 72 * a**2 * c * d * b
                - 128 * a**3 * c**3
                + 27 * a**2 * d**2
                + np.sqrt(
                    -6912 * c**3 * d**2 * a**5
                    + 729 * d**4 * a**4
                    - 3888 * b * c * d**3 * a**4
                    + 3456 * b**2 * c**2 * d**2 * a**4
                    + 108 * b**3 * d**3 * a**3
                    - 432 * b**4 * c * d**2 * a**3
                )
            )
            ** (1 / 3)
        )
        - (
            -(8 * b**3) / a**3
            + (8 * (b**2 + 2 * a * c) * b) / a**3
            - (8 * (2 * b * c - d)) / a**2
        )
        / (
            4
            * np.sqrt(
                b**2 / a**2
                - (2 * (b**2 + 2 * a * c)) / (3 * a**2)
                + (
                    2 * b**6
                    - 24 * a * c * b**4
                    + 18 * a * d * b**3
                    + 96 * a**2 * c**2 * b**2
                    - 72 * a**2 * c * d * b
                    - 128 * a**3 * c**3
                    + 27 * a**2 * d**2
                    + np.sqrt(
                        -6912 * c**3 * d**2 * a**5
                        + 729 * d**4 * a**4
                        - 3888 * b * c * d**3 * a**4
                        + 3456 * b**2 * c**2 * d**2 * a**4
                        + 108 * b**3 * d**3 * a**3
                        - 432 * b**4 * c * d**2 * a**3
                    )
                )
                ** (1 / 3)
                / (3 * 2 ** (1 / 3) * a**2)
                + (
                    2 ** (1 / 3)
                    * (
                        b**4
                        - 8 * a * c * b**2
                        + 6 * a * d * b
                        + 16 * a**2 * c**2
                    )
                )
                / (
                    3
                    * a**2
                    * (
                        2 * b**6
                        - 24 * a * c * b**4
                        + 18 * a * d * b**3
                        + 96 * a**2 * c**2 * b**2
                        - 72 * a**2 * c * d * b
                        - 128 * a**3 * c**3
                        + 27 * a**2 * d**2
                        + np.sqrt(
                            -6912 * c**3 * d**2 * a**5
                            + 729 * d**4 * a**4
                            - 3888 * b * c * d**3 * a**4
                            + 3456 * b**2 * c**2 * d**2 * a**4
                            + 108 * b**3 * d**3 * a**3
                            - 432 * b**4 * c * d**2 * a**3
                        )
                    )
                    ** (1 / 3)
                )
            )
        )
    )
)
