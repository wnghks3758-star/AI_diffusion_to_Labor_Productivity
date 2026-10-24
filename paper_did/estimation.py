"""Single-pair ATET and the explicitly separate finite-difference ACRT extension."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from labor_did.aggregation import draw_joint
from labor_did.io import write_json
from labor_did.support import scopes

from .models import fit_nuisance, predict_controls
from .numerics import atet_score, gaussian_log_ratio, np_kernel, paper_bandwidth
from .support import averaging_plan

PRIMARY = ["total", "sme", "large", "moderation"]


def summarize(point, contribution, c):
    se = float(np.linalg.norm(contribution))
    z = norm.ppf(1 - c.alpha / 2)
    return dict(
        estimate=float(point),
        se=se,
        p_value=float(2 * norm.sf(abs(point / se))) if se > 0 else np.nan,
        ci_lower=float(point - z * se),
        ci_upper=float(point + z * se),
    )


def pair_scores(frame, dt, dc, h, mu, fitted, c):
    """Fit once, score all firm-size scopes, retain common industry covariance."""
    industries = np.sort(frame.ind_code.unique())
    out = {}
    vectors = {}
    for scope, sub in scopes(frame).items():
        ix = sub.index.to_numpy()
        try:
            lr = gaussian_log_ratio(dt, dc, fitted["mean"][ix], fitted["variance"][ix])
            r = atet_score(
                sub.Y_diff,
                mu[ix],
                np_kernel(sub[c.dose], dt, h),
                np_kernel(sub[c.dose], dc, h),
                lr,
                sub.ind_code,
                c.trim,
            )
            vector = np.zeros(len(industries))
            positions = np.searchsorted(industries, np.sort(sub.ind_code.unique()))
            G = r["n_clusters_retained"]
            vector[positions] = (
                r["cluster_contributions"] * np.sqrt(G / (G - 1)) if G > 1 else np.nan
            )
            vectors[scope] = vector
            scalar = {k: v for k, v in r.items() if np.isscalar(v)}
            out[scope] = {
                **scalar,
                **summarize(r["atet"], vector, c),
                "status": "completed",
                "error": "",
            }
        except ValueError as e:
            vectors[scope] = np.full(len(industries), np.nan)
            out[scope] = dict(
                estimate=np.nan,
                se=np.nan,
                p_value=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                status="failed",
                error=str(e),
                n_firms=len(sub),
                n_industries=sub.ind_code.nunique(),
            )
    v = vectors["large"] - vectors["sme"]
    vectors["moderation"] = v
    point = out["large"]["estimate"] - out["sme"]["estimate"]
    okay = np.isfinite(v).all() and np.isfinite(point)
    out["moderation"] = {
        **summarize(point, v, c),
        "status": "completed" if okay else "failed",
        "error": "" if okay else "A size-group ATET failed",
        "n_firms": int(frame.large_firm.isin([0, 1]).sum()),
        "n_industries": len(industries),
    }
    return out, vectors


def compute_year(frame, year, c, year_dir):
    frame = frame.reset_index(drop=True)
    year_dir = Path(year_dir)
    year_dir.mkdir(parents=True, exist_ok=True)
    ind_dose = frame.groupby("ind_code")[c.dose].first().to_numpy()
    h = paper_bandwidth(ind_dose, c.bandwidth_factor)
    print(f"  {year}: fitting {c.gps_model} GPS / {c.mu_model} Mu", flush=True)
    fitted = fit_nuisance(frame, c)
    print(
        f"  {year}: models ready (cached={fitted['audit']['cache_hit']}), h={h:.6g}",
        flush=True,
    )
    pairs = [
        np.quantile(ind_dose, p).tolist() if c.pair_scale == "quantile" else list(p)
        for p in c.atet_pairs
    ]
    if len({tuple(p) for p in pairs}) != len(pairs) or any(p[0] >= p[1] for p in pairs):
        raise ValueError("ATET pairs collapse to identical observed doses")
    support_audit = {}
    plans = None
    centers = np.empty(0)
    grid = np.empty(0)
    if c.compute_acrt:
        low, high = float(ind_dose.min()), float(ind_dose.max())
        count = int(np.floor((high - low) / c.diff_step))
        if count > 2_000_000:
            raise ValueError("ACRT grid exceeds two million points; increase diff_step")
        grid = low + c.diff_step * np.arange(count + 1)
        if high - grid[-1] > 1e-12:
            grid = np.r_[grid, high]
        plans, centers, computability, support_audit = averaging_plan(
            frame, grid, h, fitted, c
        )
        computability.to_csv(year_dir / "computability.csv", index=False)
    controls = np.unique(np.r_[[p[0] for p in pairs], centers - c.diff_step])
    atet_control = {float(p[0]): [] for p in pairs}
    for i, p in enumerate(pairs):
        atet_control[float(p[0])].append(i)
    acr_control = {float(d - c.diff_step): float(d) for d in centers}
    rows = []
    point_details = []
    vectors_saved = []
    targets = ["total", "sme", "large", "common_sme", "common_large"]
    points = {t: 0.0 for t in targets}
    scores = {t: np.zeros(len(ind_dose)) for t in targets}
    valid = {
        t: bool(
            plans[t.replace("common_sme", "common").replace("common_large", "common")][
                "weights"
            ]
        )
        if plans
        else False
        for t in targets
    }
    partial = year_dir / "acrt_endpoints.partial.csv"
    if partial.exists():
        partial.unlink()
    for start, mu_batch in predict_controls(frame, controls, fitted, c):
        batch_details = []
        for local, mu in enumerate(mu_batch):
            dc = float(controls[start + local])
            for pair_id in atet_control.get(dc, []):
                _, dt = pairs[pair_id]
                results, vs = pair_scores(frame, float(dt), dc, h, mu, fitted, c)
                for effect, r in results.items():
                    rows.append(
                        dict(
                            year=year,
                            estimand="ATET",
                            pair_id=pair_id,
                            effect=effect,
                            dtreat=dt,
                            dcontrol=dc,
                            dose_step=dt - dc,
                            bandwidth=h,
                            kernel_halfwidth=np.sqrt(5) * h,
                            **r,
                        )
                    )
                    vectors_saved.append(vs[effect])
            if dc in acr_control:
                dt = acr_control[dc]
                results, vs = pair_scores(frame, dt, dc, h, mu, fitted, c)
                for scope in ("total", "sme", "large"):
                    own = plans[scope]["weights"].get(dt, 0.0)
                    common = (
                        plans["common"]["weights"].get(dt, 0.0)
                        if scope != "total"
                        else 0.0
                    )
                    if not (own or common):
                        continue
                    r = results[scope]
                    options = [(scope, own)] + (
                        [("common_" + scope, common)] if scope != "total" else []
                    )
                    for target, weight in options:
                        if not weight:
                            continue
                        if r["status"] == "completed":
                            points[target] += weight * r["estimate"] / c.diff_step
                            scores[target] += weight * vs[scope] / c.diff_step
                        else:
                            valid[target] = False
                    diag = {
                        k: r.get(k, np.nan)
                        for k in [
                            "ess_treat",
                            "ess_control",
                            "max_industry_weight",
                            "max_firm_weight",
                            "n_trimmed",
                        ]
                    }
                    batch_details.append(
                        dict(
                            year=year,
                            scope=scope,
                            dose=dt,
                            dcontrol=dc,
                            atet=r["estimate"],
                            acrt=r["estimate"] / c.diff_step,
                            own_average_weight=own,
                            common_average_weight=common,
                            status=r["status"],
                            error=r["error"],
                            **diag,
                        )
                    )
        if batch_details:
            part = pd.DataFrame(batch_details)
            part.to_csv(partial, mode="a", header=not partial.exists(), index=False)
            point_details.extend(batch_details)
        if (start // c.prediction_batch_points) % 10 == 0:
            print(
                f"  {year}: counterfactual points {min(start + len(mu_batch), len(controls))}/{len(controls)}",
                flush=True,
            )
    if c.compute_acrt:
        for t in targets:
            if not valid[t]:
                points[t] = np.nan
                scores[t][:] = np.nan
        for name, left, right in [
            ("moderation_means", "large", "sme"),
            ("moderation_common", "common_large", "common_sme"),
        ]:
            points[name] = points[left] - points[right]
            scores[name] = scores[left] - scores[right]
        chosen = (
            "moderation_means" if c.size_difference == "means" else "moderation_common"
        )
        points["moderation"] = points[chosen]
        scores["moderation"] = scores[chosen]
        for effect, point in points.items():
            scope = (
                "common"
                if effect.startswith("common_") or effect == "moderation_common"
                else ("total" if effect.startswith("moderation") else effect)
            )
            detail = plans[scope]["audit"].copy()
            if effect == "moderation_means" or (
                effect == "moderation" and c.size_difference == "means"
            ):
                detail["coverage"] = min(
                    plans["sme"]["audit"]["coverage"],
                    plans["large"]["audit"]["coverage"],
                )
                for key in [
                    "n_target_firms",
                    "n_used_firms",
                    "n_gap_firms",
                    "n_in_range_firms",
                    "active_endpoints",
                ]:
                    detail[key] = sum(
                        plans[s]["audit"].get(key, 0) for s in ["sme", "large"]
                    )
                detail["weight_sum"] = None
            elif effect == "moderation" and c.size_difference == "common":
                detail = plans["common"]["audit"].copy()
            r = summarize(point, scores[effect], c)
            rows.append(
                dict(
                    year=year,
                    estimand="finite_difference_ACRT",
                    pair_id=-1,
                    effect=effect,
                    dose_step=c.diff_step,
                    bandwidth=h,
                    kernel_halfwidth=np.sqrt(5) * h,
                    status="completed" if np.isfinite(point) else "failed",
                    error=""
                    if np.isfinite(point)
                    else "A required endpoint failed; no silent dropping",
                    **r,
                    n_firms=len(frame),
                    n_industries=len(ind_dose),
                    **detail,
                )
            )
            vectors_saved.append(scores[effect])
    table = pd.DataFrame(rows)
    matrix = np.column_stack(vectors_saved)
    # Shared multipliers retain SME/Large covariance; analytic R-style SE stays primary.
    draws = draw_joint(table.estimate.to_numpy(), matrix, c)
    table["bootstrap_se"] = np.std(draws, axis=0, ddof=1)
    table["percentile_lower"], table["percentile_upper"] = np.quantile(
        draws, [c.alpha / 2, 1 - c.alpha / 2], axis=0
    )
    np.savez_compressed(
        year_dir / "cluster_scores.npz",
        contributions=matrix,
        industries=np.sort(frame.ind_code.unique()),
    )
    table.to_csv(year_dir / "estimates.csv", index=False)
    endpoints = pd.DataFrame(point_details)
    endpoints.to_csv(year_dir / "acrt_endpoints.csv", index=False)
    if partial.exists():
        partial.unlink()
    audit = dict(
        year=year,
        n_firms=len(frame),
        n_industries=len(ind_dose),
        bandwidth=h,
        kernel_halfwidth=np.sqrt(5) * h,
        psmethod=1,
        gps=fitted["audit"],
        atet_pairs=pairs,
        ess_filter=None,
        acrt_support=c.acrt_support,
        computability=support_audit,
        density_floor=None,
        missing_policy=c.missing_policy,
        acrt_grid_points=len(grid),
        acrt_evaluated_centers=len(centers),
        averaging={k: v["audit"] for k, v in plans.items()} if plans else {},
        failed_endpoints=int(endpoints.status.eq("failed").sum())
        if len(endpoints)
        else 0,
    )
    write_json(year_dir / "audit.json", audit)
    return table, endpoints, audit
