import pandas as pd
from PyCO2SYS.api import CO2SYS_wrap as co2sys

# Calculate buffers both ways under standard conditions
testvars = {"alk": 2300, "dic": 2150}
auto = co2sys(**testvars, buffers_mode="auto")
expl = co2sys(**testvars, buffers_mode="explicit")

# Extract calculated buffer factors
buffers = {
    # Revelle factor:
    "RFin": "Revelle factor",
    # FGC94 psi:
    "psi_in": "<i>ψ</i>",
    # ESM10 buffers:
    "gammaTCin": "<i>γ</i><sub>DIC</sub>",
    "gammaTAin": "<i>γ</i><sub>Alk</sub>",
    "betaTCin": "<i>β</i><sub>DIC</sub>",
    "betaTAin": "<i>β</i><sub>Alk</sub>",
    "omegaTCin": "<i>ω</i><sub>DIC</sub>",
    "omegaTAin": "<i>ω</i><sub>Alk</sub>",
    # HDW18 isocapnic quotient:
    "isoQin": "<i>Q</i>",
}
auto = auto[buffers.keys()]
expl = expl[buffers.keys()]

# Multiply certain buffers by 10^3
esm10 = ["gammaTCin", "gammaTAin", "betaTCin", "betaTAin", "omegaTCin", "omegaTAin"]
for b in esm10:
    auto.loc[0][b] = auto[b] * 1e3
    expl.loc[0][b] = expl[b] * 1e3

# Compare automatic vs explicit buffers
diff = auto.subtract(expl)
compare = pd.concat(
    (
        auto.rename(index={0: "Automatic"}),
        expl.rename(index={0: "Explicit"}),
        diff.rename(index={0: "Difference"}),
    )
)

# Generate HTML table for docs, if requested
if False:
    with open("validate/html/buffers_auto_explicit.md", "w") as f:
        f.write(
            "<!-- HTML for table generated with examples/buffers-auto-explicit.py -->\n"
        )
        f.write("<table><tr><th></th>\n")
        for b in compare.columns:
            f.write('<th style="text-align:center">{}</th>\n'.format(buffers[b]))
        for ctype in ["Explicit", "Automatic"]:
            f.write('</tr><tr>\n<th style="text-align:center">{}</th>\n'.format(ctype))
            for b in compare.columns:
                bval = "{:.4f}".format(compare.loc[ctype][b])
                bval = bval.replace("-", "−")
                f.write('<td style="text-align:center">{}</td>\n'.format(bval))
        f.write('</tr><tr>\n<th style="text-align:center">Difference</th>\n')
        for b in compare.columns:
            bdiff = "{:.2e}".format(compare.loc["Difference"][b])
            bdiff = bdiff.replace("e-0", "·10<sup>−").replace("-", "−") + "</sup>"
            f.write('<td style="text-align:center">{}</td>\n'.format(bdiff))
        f.write("</tr></table>\n")


def test_auto_explicit():
    assert compare.loc["Difference"].abs().max() < 1e-6
