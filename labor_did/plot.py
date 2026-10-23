"""Export figures without retaining open Matplotlib objects."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_event_study(estimates, path, base_year=2022, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, effects in zip(axes, [["total"], ["sme", "large"], ["moderation"]]):
        for effect in effects:
            sub = estimates.loc[estimates.effect.eq(effect)].sort_values("year")
            zero = {c: 0 for c in ["year", "estimate", "se", "ci_lower", "ci_upper"]}
            zero["year"] = base_year
            sub = pd.concat([sub, pd.DataFrame([zero])]).sort_values("year")
            ax.errorbar(
                sub.year,
                sub.estimate,
                yerr=[sub.estimate - sub.ci_lower, sub.ci_upper - sub.estimate],
                marker="o",
                capsize=3,
                label=effect,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(base_year, color="gray", linestyle=":", linewidth=0.8)
        ax.set(xlabel="Year", ylabel="Finite-difference ACR")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_title("Total")
    axes[1].set_title("Own firm-weighted averages")
    axes[2].set_title("Large minus SME")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
