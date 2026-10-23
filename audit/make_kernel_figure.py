"""Export diagnostic figure from the saved, unfiltered pilot tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
p = ROOT / "results/psmethod2_151"
a = pd.read_csv(p / "actual_cv.csv")
s = pd.read_csv(p / "actual_support.csv")
a = a[a.model.eq("rf_depth3_leaf8_separate")]
s = s[s.model.eq("rf_depth3_leaf8_separate")]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
ax[0].plot(
    a.bandwidth_factor, 100 * a.improvement_vs_unconditional, "o-", color="#146c94"
)
ax[0].set(
    xlabel="Bandwidth multiplier (actual data)",
    ylabel="OOF kernel MSE reduction (%)",
    title="151 industries: separate RF",
)
b = s.groupby("bandwidth_factor").supported.mean()
ax[1].bar(np.arange(len(b)), 100 * b, color="#146c94")
ax[1].set(
    xticks=np.arange(len(b)),
    xticklabels=[f"{x:g}" for x in b.index],
    ylim=(0, 100),
    xlabel="Bandwidth multiplier (actual data)",
    ylabel="Both industry ESS > 5 (%)",
    title="9 contrasts x 5 splits",
)
z = pd.read_csv(p / "simulation_summary.csv")
for name, color, label in [
    ("rf_separate", "#146c94", "RF"),
    ("oracle_smoothed", "#e68632", "Oracle GPS"),
]:
    g = z[z.model.eq(name)]
    ax[2].errorbar(
        g.bandwidth_factor,
        g.coverage95_among_estimable,
        yerr=1.96 * g.coverage_mcse,
        fmt="o-",
        color=color,
        label=label,
        capsize=3,
    )
ax[2].axhline(0.95, ls="--", color="gray", lw=1)
ax[2].set(
    xlabel="SD-rule factor (simulation)",
    ylabel="95% interval coverage",
    title="Known ATET: 100 simulations",
    ylim=(0.3, 1.02),
)
ax[2].legend(frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", alpha=0.15)
fig.savefig(ROOT / "figures/psmethod2_151.png", dpi=180)
fig.savefig(ROOT / "figures/psmethod2_151.svg")

svg = ROOT / "figures/psmethod2_151.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
