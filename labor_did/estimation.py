"""Annual finite-difference ATET/ACR, streamed over interpolation endpoints."""

import numpy as np
import pandas as pd

from .aggregation import cluster_contribution, draw_joint, inference
from .data import observed_grid
from .models import fit_gps, fit_mu, predict_mu
from .score import normalized_acr_score
from .support import plan_averages, scan_support, scopes
from .weights import get_trimmed_weights


def estimate_year(frame, year, config, on_stage=None):
    notify = on_stage or (lambda *args: None)
    gps_mean, gps_sigma, gps_audit = fit_gps(frame, config)
    frame = frame.copy()
    frame["D_pred"] = gps_mean
    frame["sigma_pred"] = gps_sigma
    notify("gps_done", gps_audit)
    grid, bandwidth = observed_grid(frame, config)
    supported, support_audit = scan_support(frame, grid, bandwidth, config)
    plans, high_centers = plan_averages(frame, supported, grid, bandwidth, config)
    if not len(high_centers):
        raise ValueError(f"{year}: no estimable averaging endpoints")
    notify(
        "support_done",
        {"grid_points": len(grid), "active_endpoints": len(high_centers)},
    )
    folder, manifest, mu_audit = fit_mu(frame, config)
    notify(
        "mu_done",
        {"cache_hit": mu_audit["cache_hit"], "models": len(manifest["models"])},
    )
    cluster_column = "ind_code" if config.bootstrap_unit == "industry" else "firm_code"
    clusters = pd.Index(np.sort(frame[cluster_column].unique()))
    n_clusters = len(clusters)
    scope_frames = scopes(frame)
    indexes = {
        "total": np.arange(len(frame)),
        "sme": np.flatnonzero(frame.large_firm.eq(0)),
        "large": np.flatnonzero(frame.large_firm.eq(1)),
    }
    targets = ["total", "sme", "large", "common_sme", "common_large"]
    points = {key: 0.0 for key in targets}
    scores = {key: np.zeros(n_clusters) for key in targets}
    valid = {
        key: bool(
            plans[
                key.replace("common_sme", "common").replace("common_large", "common")
            ]["weights"]
        )
        for key in targets
    }
    cluster_codes = {
        key: clusters.get_indexer(sub[cluster_column])
        for key, sub in scope_frames.items()
    }
    endpoints = []
    with predict_mu(
        frame, high_centers - config.diff_step, config, folder, manifest
    ) as prediction:
        prediction_audit = prediction.audit
        for start, stop, mu_batch in prediction.batches():
            for local, position in enumerate(range(start, stop)):
                center = float(high_centers[position])
                mu = mu_batch[local]
                for scope, sub in scope_frames.items():
                    own_weight = plans[scope]["weights"].get(center, 0.0)
                    common_weight = (
                        plans["common"]["weights"].get(center, 0.0)
                        if scope != "total"
                        else 0.0
                    )
                    if not (own_weight or common_weight):
                        continue
                    a, b, diagnostic = get_trimmed_weights(
                        sub,
                        center,
                        center - config.diff_step,
                        sub.D_pred.to_numpy(),
                        sub.sigma_pred.to_numpy(),
                        bandwidth,
                        config.dose,
                        "ind_code",
                        config.trim_threshold,
                        config.density_floor,
                        trim_mode=config.trim_mode,
                    )
                    estimable = bool(len(sub) and a.sum() > 0 and b.sum() > 0)
                    targets_here = [(scope, own_weight)]
                    if scope != "total":
                        targets_here.append(("common_" + scope, common_weight))
                    if estimable:
                        residual = sub.Y_diff.to_numpy() - mu[indexes[scope]]
                        point, influence, normalization = normalized_acr_score(
                            residual, a, b, config.diff_step
                        )
                        contribution = cluster_contribution(
                            influence, cluster_codes[scope], n_clusters
                        )
                        for target, weight in targets_here:
                            points[target] += weight * point
                            scores[target] += weight * contribution
                    else:
                        point = np.nan
                        normalization = {}
                        for target, weight in targets_here:
                            if weight:
                                valid[target] = False
                    endpoints.append(
                        {
                            "year": year,
                            "scope": scope,
                            "dose": center,
                            "acr": point,
                            "own_average_weight": own_weight,
                            "common_average_weight": common_weight,
                            "estimable": estimable,
                            **diagnostic,
                            **normalization,
                        }
                    )
    for key in targets:
        if not valid[key]:
            points[key] = np.nan
            scores[key][:] = np.nan
    ordered = targets + ["moderation_means", "moderation_common"]
    points["moderation_means"] = points["large"] - points["sme"]
    scores["moderation_means"] = scores["large"] - scores["sme"]
    points["moderation_common"] = points["common_large"] - points["common_sme"]
    scores["moderation_common"] = scores["common_large"] - scores["common_sme"]
    point_array = np.array([points[key] for key in ordered])
    score_array = np.column_stack([scores[key] for key in ordered])
    draws = draw_joint(point_array, score_array, config)
    rows = []
    for i, key in enumerate(ordered):
        plan = plans[
            "common"
            if key.startswith("common_") or key == "moderation_common"
            else ("total" if key == "moderation_means" else key)
        ]
        row = {
            "year": year,
            "effect": key,
            **inference(points[key], draws[:, i], config.alpha),
            "estimand": "finite_difference_ACR",
            "dose_step": config.diff_step,
            "atet_estimate": float(points[key] * config.diff_step),
            "atet_se": float(np.std(draws[:, i], ddof=1) * config.diff_step),
            "n_clusters": n_clusters,
            "n_firms": len(frame),
            "bandwidth": bandwidth,
            **plan["audit"],
        }
        if key == "moderation_means":
            # Own-group averages use different pools; no common coverage is implied.
            row["coverage"] = min(
                plans["sme"]["audit"]["coverage"], plans["large"]["audit"]["coverage"]
            )
            for field in [
                "n_used_firms",
                "n_target_firms",
                "n_in_range_firms",
                "n_gap_firms",
                "active_endpoints",
            ]:
                row[field] = sum(
                    plans[scope]["audit"].get(field, 0) for scope in ["sme", "large"]
                )
            row["weight_sum"] = (
                np.nan
            )  # A difference of two averages has signed coefficients.
        rows.append(row)
    selected = (
        "moderation_means" if config.size_difference == "means" else "moderation_common"
    )
    selected_row = next(row.copy() for row in rows if row["effect"] == selected)
    selected_row["effect"] = "moderation"
    rows.append(selected_row)
    audit = {
        "kernel_convention": config.kernel_convention,
        "kernel_halfwidth": bandwidth,
        "kernel_rule_bandwidth": bandwidth
        / (np.sqrt(5) if config.kernel_convention == "np" else 1.0),
        "trim_mode": config.trim_mode,
        "year": year,
        "grid_points": len(grid),
        "supported_points": {s: len(v) for s, v in supported.items()},
        "averaging": {s: plan["audit"] for s, plan in plans.items()},
        "gps": gps_audit,
        "mu": mu_audit,
        "support": support_audit,
        "prediction": prediction_audit,
        "stored_bootstrap_matrix_shape": list(draws.shape),
        "avoided_dense_grid_bootstrap_bytes": int(
            len(grid) * config.n_bootstrap * 8 * 3
        ),
    }
    if config.alpha == 0.05:
        table = pd.DataFrame(rows)
        table["ci_lower"] = table.estimate - 1.96 * table.se
        table["ci_upper"] = table.estimate + 1.96 * table.se
        rows = table.to_dict("records")
    # Expose joint cluster scores for additional inference through the Python API.
    return (
        pd.DataFrame(rows),
        pd.DataFrame(endpoints),
        audit,
        {
            "effects": ordered,
            "points": point_array,
            "scores": score_array,
            "clusters": clusters.to_numpy(),
            "draws": draws if config.save_bootstrap_draws else None,
        },
    )
