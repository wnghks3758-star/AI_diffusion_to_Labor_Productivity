"""Chunked ESS scanning and exact interpolation weights for actual firms."""

from pathlib import Path

import joblib
import numpy as np

from .io import input_key
from .weights import select_supported_centers


def scopes(frame):
    return {
        "total": frame,
        "sme": frame.loc[frame.large_firm.eq(0)],
        "large": frame.loc[frame.large_firm.eq(1)],
    }


def scan_support(frame, grid, bandwidth, config):
    settings = {
        k: getattr(config, k)
        for k in ["ess", "trim_threshold", "density_floor", "diff_step"]
    }
    settings.update(
        bandwidth=bandwidth,
        grid_start=float(grid[0]),
        grid_end=float(grid[-1]),
        grid_points=len(grid),
    )
    key = input_key(
        frame[
            ["ind_code", "firm_code", config.dose, "large_firm", "D_pred", "sigma_pred"]
        ],
        settings,
    )
    path = Path(config.cache_dir) / "support" / f"{key}.joblib"
    if path.exists():
        return joblib.load(path), {"cache_hit": True, "cache_key": key}
    result = {}
    for scope, sub in scopes(frame).items():
        selected = []
        for start in range(0, len(grid), config.grid_chunk_points):
            values, _ = select_supported_centers(
                sub,
                grid[start : start + config.grid_chunk_points],
                bandwidth,
                config.diff_step,
                config.dose,
                "ind_code",
                config.ess,
                config.trim_threshold,
                config.density_floor,
                options={"support_batch_points": config.support_batch_points},
            )
            selected.append(values)
        result[scope] = np.concatenate(selected) if selected else np.empty(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    return result, {"cache_hit": False, "cache_key": key}


def averaging_weights(frame, supported, full_grid, bandwidth, config, common=False):
    """Firm-count weighting; strict excludes gaps before weights are computed."""
    if common:
        frame = frame.loc[frame.large_firm.isin([0, 1])]
        eligible = (
            frame.groupby("ind_code").large_firm.nunique().loc[lambda x: x.eq(2)].index
        )
        frame = frame.loc[frame.ind_code.isin(eligible)]
    info = frame.groupby("ind_code").agg(
        d=(config.dose, "first"), count=("firm_code", "nunique")
    )
    d = info.d.to_numpy(float)
    counts = info["count"].to_numpy(float)
    grid = np.asarray(supported, float)
    weights = np.zeros(len(grid))
    if not len(grid):
        return weights, {
            "n_target_firms": int(counts.sum()),
            "n_used_firms": 0,
            "coverage": 0.0,
            "n_gap_firms": 0,
            "weight_sum": 0.0,
        }
    inside = (d >= grid[0]) & (d <= grid[-1])
    left = np.clip(np.searchsorted(grid, d, side="right") - 1, 0, len(grid) - 1)
    right = np.minimum(left + 1, len(grid) - 1)
    fraction = np.divide(
        d - grid[left],
        grid[right] - grid[left],
        out=np.zeros(len(d)),
        where=grid[right] > grid[left],
    )
    exact_left = np.isclose(d, grid[left], rtol=0, atol=1e-12)
    exact_right = np.isclose(d, grid[right], rtol=0, atol=1e-12)
    fraction[exact_left] = 0.0
    fraction[exact_right] = 1.0
    positions = np.searchsorted(full_grid, grid)
    gap = (positions[right] - positions[left] > 1) | (
        grid[right] - grid[left] > bandwidth + 1e-12
    )
    bridged = inside & gap & ~exact_left & ~exact_right
    keep = inside & (~bridged if config.gap_policy == "strict" else True)
    mass = counts[keep].sum()
    if mass:
        np.add.at(weights, left[keep], counts[keep] * (1 - fraction[keep]) / mass)
        np.add.at(weights, right[keep], counts[keep] * fraction[keep] / mass)
    return weights, {
        "n_target_firms": int(counts.sum()),
        "n_in_range_firms": int(counts[inside].sum()),
        "n_used_firms": int(mass),
        "n_gap_firms": int(counts[bridged].sum()),
        "coverage": float(mass / counts.sum()) if counts.sum() else 0.0,
        "weight_sum": float(weights.sum()),
        "active_endpoints": int((weights > 0).sum()),
    }


def plan_averages(frame, supported, grid, bandwidth, config):
    plans = {}
    active = []
    for scope, sub in scopes(frame).items():
        weights, audit = averaging_weights(
            sub, supported[scope], grid, bandwidth, config
        )
        plans[scope] = {
            "weights": dict(
                zip(map(float, supported[scope][weights > 0]), weights[weights > 0])
            ),
            "audit": audit,
        }
        active.extend(plans[scope]["weights"])
    common, audit = averaging_weights(
        frame, supported["total"], grid, bandwidth, config, common=True
    )
    plans["common"] = {
        "weights": dict(
            zip(map(float, supported["total"][common > 0]), common[common > 0])
        ),
        "audit": audit,
    }
    active.extend(plans["common"]["weights"])
    return plans, np.unique(active)
