"""Fixed-reference ATET and empirical firm-weighted overall ATET."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .numerics import atet_score, gaussian_log_ratio, np_kernel

EFFECTS = ["total", "sme", "large", "moderation"]


def scopes(frame):
    return {"total": frame, "sme": frame[frame.large_firm.eq(0)], "large": frame[frame.large_firm.eq(1)]}


def summarize(point, scores, alpha):
    se = float(np.linalg.norm(scores))
    z = norm.ppf(1 - alpha / 2)
    return dict(estimate=float(point), se=se, p_value=float(2 * norm.sf(abs(point/se))) if se > 0 else np.nan,
                ci_lower=float(point-z*se), ci_upper=float(point+z*se))


def pair_scores(frame, dt, dc, h, mu, fitted, c, kernel=None):
    industries = np.sort(frame.ind_code.unique())
    results, vectors = {}, {}
    for effect, sub in scopes(frame).items():
        vector = np.zeros(len(industries))
        try:
            ix = sub.index.to_numpy()
            high,low = np_kernel(sub[c.dose],dt,h),np_kernel(sub[c.dose],dc,h)
            if c.psmethod == 1:
                ratio = gaussian_log_ratio(dt,dc,fitted["mean"][ix],fitted["variance"][ix])
            else:
                from .kernel_gps import kernel_log_ratio
                if kernel is None:
                    raise ValueError("PS2 requires fitted kernel GPS predictions")
                ratio = kernel_log_ratio(sub,dt,dc,kernel,high,low)
            scored = atet_score(sub.Y_diff,mu[ix],high,low,ratio,sub.ind_code,c.trim)
            G = scored["n_clusters_retained"]
            if G < 2:
                raise ValueError("Fewer than two retained industries")
            pos = np.searchsorted(industries,np.sort(sub.ind_code.unique()))
            vector[pos] = scored["cluster_contributions"] * np.sqrt(G/(G-1))
            result = {k:v for k,v in scored.items() if np.isscalar(v)}
            result.update(summarize(scored["atet"],vector,c.alpha),status="completed",error="")
        except ValueError as exc:
            vector[:] = np.nan
            result = dict(**summarize(np.nan,vector,c.alpha),status="failed",error=str(exc),n_firms=len(sub))
        vectors[effect], results[effect] = vector, result
    vector = vectors["large"]-vectors["sme"]
    point = results["large"]["estimate"]-results["sme"]["estimate"]
    valid = np.isfinite(point) and np.isfinite(vector).all()
    results["moderation"] = dict(**summarize(point,vector,c.alpha),status="completed" if valid else "failed",
                                 error="" if valid else "A size-group ATET failed")
    vectors["moderation"] = vector
    return results, vectors


def aggregate(frame, results, dc, c, year, policy):
    """Aggregate cluster scores before SE; weights/support treated as fixed."""
    industries = np.sort(frame.ind_code.unique())
    rows, vectors, details = [], {}, {}
    for effect, sub in scopes(frame).items():
        target = sub[sub[c.dose] > dc]
        counts = target.groupby(c.dose).firm_code.nunique().sort_index()
        available = [d for d in counts.index if results[float(d)][0][effect]["status"] == "completed"]
        used = counts if policy == "strict" else counts[counts.index.isin(available)]
        valid = bool(len(used)) and all(d in available for d in used.index)
        weights = used/used.sum() if valid else pd.Series(dtype=float)
        point, vector = 0., np.zeros(len(industries))
        for d, w in weights.items():
            r, v = results[float(d)]
            point += w*r[effect]["estimate"]
            vector += w*v[effect]
        if not valid:
            point, vector = np.nan, np.full(len(industries),np.nan)
        total, mass = int(counts.sum()), int(used.sum()) if valid else 0
        eligible_mass = int(counts[counts.index.isin(available)].sum())
        detail = dict(n_target_firms=total,n_used_firms=mass,coverage=mass/total if total else 0.,
                      computable_firms=eligible_mass,computable_coverage=eligible_mass/total if total else 0.,
                      target_dose_points=len(counts),used_dose_points=len(weights),
                      target_industries=target.ind_code.nunique(),excluded_target_industries=sub.ind_code.nunique()-target.ind_code.nunique(),
                      max_aggregation_weight=float(weights.max()) if len(weights) else np.nan)
        details[effect], vectors[effect] = detail, vector
        rows.append(dict(year=year,estimand="ATET_overall",effect=effect,support_policy=policy,
                         **summarize(point,vector,c.alpha),**detail,status="completed" if valid else "failed",
                         error="" if valid else "Missing or failed target ATET"))
    vector = vectors["large"]-vectors["sme"]
    point = rows[2]["estimate"]-rows[1]["estimate"]
    valid = np.isfinite(point) and np.isfinite(vector).all()
    rows.append(dict(year=year,estimand="ATET_overall",effect="moderation",support_policy=policy,
                     **summarize(point,vector,c.alpha),
                     n_target_firms=details["sme"]["n_target_firms"]+details["large"]["n_target_firms"],
                     n_used_firms=details["sme"]["n_used_firms"]+details["large"]["n_used_firms"] if valid else 0,
                     coverage=min(details[s]["coverage"] for s in ["sme","large"]),
                     computable_coverage=min(details[s]["computable_coverage"] for s in ["sme","large"]),
                     status="completed" if valid else "failed",error="" if valid else "A size-group aggregate failed"))
    vectors["moderation"] = vector
    return pd.DataFrame(rows),np.column_stack([vectors[e] for e in EFFECTS])
