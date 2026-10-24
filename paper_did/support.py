"""Only mathematical computability after R trimming; no ESS threshold or outcome."""

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from labor_did.support import averaging_weights, plan_averages, scopes

from .numerics import gaussian_log_ratio, np_kernel


def computable_grid(frame, grid, h, fitted, c):
    flags = {}
    audits = {}
    for scope, sub in scopes(frame).items():
        if not len(sub):
            flags[scope] = np.zeros(len(grid), bool)
            audits[scope] = dict(grid_points=len(grid), computable_points=0)
            continue
        compressed = (
            sub.assign(
                _mean=fitted["mean"][sub.index], _variance=fitted["variance"][sub.index]
            )
            .groupby("ind_code")
            .agg(
                dose=(c.dose, "first"),
                count=("firm_code", "size"),
                mean=("_mean", "first"),
                variance=("_variance", "first"),
            )
        )
        D = compressed.dose.to_numpy()[None, :]
        n = compressed["count"].to_numpy()[None, :]
        mean = compressed["mean"].to_numpy()[None, :]
        variance = compressed.variance.to_numpy()[None, :]
        values = []
        for start in range(0, len(grid), 256):
            d = grid[start : start + 256, None]
            high = np_kernel(D, d, h)
            low = np_kernel(D, d - c.diff_step, h)
            a = np.divide(
                high,
                np.sum(high * n, axis=1, keepdims=True),
                out=np.zeros_like(high),
                where=np.sum(high * n, axis=1, keepdims=True) > 0,
            )
            loglow = np.full_like(low, -np.inf)
            np.log(low, out=loglow, where=low > 0)
            logb = loglow + gaussian_log_ratio(d, d - c.diff_step, mean, variance)
            mass = logsumexp(logb + np.log(n), axis=1, keepdims=True)
            with np.errstate(invalid="ignore"):
                b = np.where(np.isfinite(mass), np.exp(logb - mass), 0.0)
            keep = (a <= c.trim) & (b <= c.trim)
            values.append(
                (np.sum(a * n * keep, axis=1) > 0) & (np.sum(b * n * keep, axis=1) > 0)
            )
        flags[scope] = np.concatenate(values)
        failed = ~flags[scope]
        edges = np.r_[0, np.flatnonzero(np.diff(failed)) + 1, len(grid)]
        intervals = [
            [float(grid[a]), float(grid[b - 1])]
            for a, b in zip(edges[:-1], edges[1:])
            if failed[a]
        ]
        audits[scope] = dict(
            grid_points=len(grid),
            computable_points=int(flags[scope].sum()),
            undefined_points=int(failed.sum()),
            undefined_intervals=intervals,
        )
    return flags, audits


def averaging_config(c):
    return replace(
        c, gap_policy="strict" if c.gap_policy in ("exclude", "strict") else "bridge"
    )


def averaging_plan(frame, grid, h, fitted, c):
    c = averaging_config(c)
    flags, audit = computable_grid(frame, grid, h, fitted, c)
    supported = {
        k: grid[v] if c.acrt_support == "computable" else grid for k, v in flags.items()
    }
    plans, centers = plan_averages(frame, supported, grid, np.sqrt(5) * h, c)
    if c.acrt_support == "computable":
        # Common-dose averages require both group curves to be defined.
        common = grid[np.logical_and.reduce(list(flags.values()))]
        weights, detail = averaging_weights(
            frame, common, grid, np.sqrt(5) * h, c, common=True
        )
        plans["common"] = dict(
            weights=dict(zip(map(float, common[weights > 0]), weights[weights > 0])),
            audit=detail,
        )
        centers = np.unique([d for plan in plans.values() for d in plan["weights"]])
    return plans, centers, pd.DataFrame(dict(dose=grid, **flags)), audit
