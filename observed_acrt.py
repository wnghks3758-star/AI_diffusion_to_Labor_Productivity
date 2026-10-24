"""Direct observed-dose ACRT and separate-point kernel GPS; original paper core stays unchanged."""

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from labor_did.io import input_key
from labor_did.support import averaging_weights, scopes
from paper_did.estimation import pair_scores, summarize
from paper_did.models import columns_for, fit_regression, industry_folds
from paper_did.numerics import atet_score, np_kernel
from paper_did.support import averaging_config

PRIMARY = ["total", "sme", "large", "moderation"]


def kernel_fold(task):
    ind, columns, centers, h, config, fold, train, test = task
    X, D = ind[columns], ind[config.dose].to_numpy()
    pred = np.empty((len(test), len(centers)))
    nonzero, mse, null_mse = [], [], []
    fitted_count = 0
    with threadpool_limits(limits=1):
        for j, center in enumerate(centers):
            target = np_kernel(D, center, h)
            nonzero.append(int((target[train] > 0).sum()))
            if not nonzero[-1]:
                pred[:, j] = 0.
            else:
                model, _ = fit_regression(X.iloc[train], target[train], ind.ind_code.iloc[train],
                                          config.gps_model, config.gps_params, config)
                pred[:, j] = model.predict(X.iloc[test])
                fitted_count += 1
            mse.append(float(np.mean((pred[:, j] - target[test])**2)))
            null_mse.append(float(np.mean((target[train].mean() - target[test])**2)))
    audit = dict(fold=fold, train_industry_positions=train.tolist(), test_industry_positions=test.tolist(),
                 nonzero_training_targets=nonzero, mse=mse, unconditional_mse=null_mse, fitted_models=fitted_count)
    return test, pred, audit


def fit_kernel_oof(frame, centers, h, c, workers=2):
    centers = np.asarray(centers, float)
    if centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        raise ValueError("Kernel GPS centers must be finite, nonempty and strictly increasing")
    columns = columns_for(frame, c.gps_features, c)
    if (frame.groupby("ind_code")[columns + [c.dose]].nunique(dropna=False) > 1).any().any():
        raise ValueError("GPS covariates and dose must be constant within industry")
    ind = frame.groupby("ind_code", as_index=False).first()
    splits, seed = c.fold_settings("gps")
    settings = dict(centers=centers.tolist(), h=h, model=c.gps_model, params=c.gps_params, splits=splits, seed=seed,
                    missing_policy=c.missing_policy, learner_seed=c.seed,
                    implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = input_key(ind[["ind_code", c.dose] + columns], settings)
    file = Path(c.cache_dir) / "kernel_direct" / f"{key}.joblib"
    if file.exists():
        result = joblib.load(file)
        result["audit"]["cache_hit"] = True
        return result
    pred = np.full((len(ind), len(centers)), np.nan)
    tasks = [(ind, columns, centers, h, c, fold, tr, te) for fold, (tr, te) in enumerate(industry_folds(ind.ind_code, splits, seed))]
    audits = []
    if workers == 1:
        results = map(kernel_fold, tasks)
        for test, values, audit in results:
            pred[test] = values
            audits.append(audit)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(kernel_fold, t) for t in tasks]):
                test, values, audit = future.result()
                pred[test] = values
                audits.append(audit)
                print(f"Kernel GPS fold {audit['fold'] + 1}/{len(tasks)} ready ({len(centers)} separate targets)", flush=True)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite direct-kernel OOF predictions")
    result = dict(centers=centers, industries=ind.ind_code.to_numpy(), predictions=pred,
                  audit=dict(cache_key=key, cache_hit=False, columns=columns, folds=sorted(audits, key=lambda a: a["fold"]),
                             model=c.gps_model, params=c.gps_params, target="E[K_h(D-d)|X]", gaussian_assumption=False,
                             density_floor=None, separate_models=True, h=h, kernel_halfwidth=np.sqrt(5)*h,
                             fold_seed=seed, n_splits=len(tasks), evaluated_points=len(centers),
                             fitted_models=sum(a["fitted_models"] for a in audits)))
    file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, file, compress=3)
    return result


def kernel_pair_scores(frame, dt, dc, h, mu, kernel, c):
    qt = kernel["predictions"][:, np.searchsorted(kernel["centers"], dt)]
    qc = kernel["predictions"][:, np.searchsorted(kernel["centers"], dc)]
    mapping = np.searchsorted(kernel["industries"], frame.ind_code.to_numpy())
    qt, qc = qt[mapping], qc[mapping]
    industries = np.sort(frame.ind_code.unique())
    results, vectors = {}, {}
    for scope, sub in scopes(frame).items():
        ix = sub.index.to_numpy()
        high, low = np_kernel(sub[c.dose], dt, h), np_kernel(sub[c.dose], dc, h)
        a, b = qt[ix], qc[ix]
        try:
            if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
                raise ValueError("Nonfinite or negative kernel GPS prediction")
            if ((low > 0) & ((b <= 0) | (a <= 0))).any() or ((high > 0) & (a <= 0)).any():
                raise ValueError("Nonpositive kernel GPS on active support; no clipping/floor")
            ratio = np.zeros(len(ix))
            okay = (a > 0) & (b > 0)
            ratio[okay] = np.log(a[okay]) - np.log(b[okay])
            r = atet_score(sub.Y_diff, mu[ix], high, low, ratio, sub.ind_code, c.trim)
            G = r["n_clusters_retained"]
            v = np.zeros(len(industries))
            v[np.searchsorted(industries, np.sort(sub.ind_code.unique()))] = r["cluster_contributions"] * np.sqrt(G/(G-1)) if G > 1 else np.nan
            vectors[scope] = v
            results[scope] = {**{k: val for k, val in r.items() if np.isscalar(val)},
                              **summarize(r["atet"], v, c), "status": "completed", "error": ""}
        except ValueError as e:
            vectors[scope] = np.full(len(industries), np.nan)
            results[scope] = dict(estimate=np.nan, se=np.nan, p_value=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                                 status="failed", error=str(e))
    v = vectors["large"] - vectors["sme"]
    point = results["large"]["estimate"] - results["sme"]["estimate"]
    vectors["moderation"] = v
    results["moderation"] = dict(**summarize(point, v, c), status="completed" if np.isfinite(point) else "failed", error="")
    return results, vectors


def evaluate_pairs(frame, pairs, fitted, c, h, kernel=None):
    from paper_did.models import predict_controls

    pairs = list(dict.fromkeys((float(dt), float(dc)) for dt, dc in pairs))
    controls = np.unique([dc for _, dc in pairs])
    by_control = {dc: [] for dc in controls}
    for dt, dc in pairs:
        by_control[dc].append(dt)
    results = {}
    for start, batch in predict_controls(frame, controls, fitted, c):
        for j, mu in enumerate(batch):
            dc = float(controls[start + j])
            for dt in by_control[dc]:
                results[(dt, dc)] = pair_scores(frame, dt, dc, h, mu, fitted, c) if kernel is None else kernel_pair_scores(frame, dt, dc, h, mu, kernel, c)
    return results


def direct_weights(sub, dose, selected=None):
    counts = sub.groupby(dose).firm_code.nunique().sort_index()
    used = counts if selected is None else counts[counts.index.isin(selected)]
    n = int(counts.sum())
    mass = int(used.sum())
    weights = (used / mass).to_dict() if mass else {}
    return weights, dict(n_target_firms=n, n_used_firms=mass, coverage=mass/n if n else 0.,
                         target_dose_points=len(counts), used_dose_points=len(used))


def aggregate_results(frame, results, c, year, h, targets, mode="observed", grid=None, supported=None):
    rows, vectors, details = [], {}, {}
    for scope, sub in scopes(frame).items():
        selected = set(targets[scope])
        if mode == "observed":
            weights, detail = direct_weights(sub, c.dose, selected)
        else:
            target = sub[sub[c.dose].isin(selected)]
            w, _ = averaging_weights(target, supported[scope], grid, np.sqrt(5)*h, averaging_config(c))
            weights = dict(zip(supported[scope][w > 0], w[w > 0]))
            _, detail = direct_weights(sub, c.dose, selected)
        point = 0.
        vector = np.zeros(frame.ind_code.nunique())
        valid = bool(weights)
        for d, weight in weights.items():
            r, vs = results[(float(d), float(d - c.diff_step))]
            valid = valid and r[scope]["status"] == "completed"
            point += weight * r[scope]["estimate"] / c.diff_step
            vector += weight * vs[scope] / c.diff_step
        if not valid:
            point, vector = np.nan, np.full_like(vector, np.nan)
        vectors[scope] = vector
        details[scope] = detail
        rows.append(dict(year=year, estimand="finite_difference_ACRT", pair_id=-1, effect=scope,
                         **summarize(point, vector, c), **detail, status="completed" if valid else "failed"))
    vector = vectors["large"] - vectors["sme"]
    point = rows[2]["estimate"] - rows[1]["estimate"]
    rows.append(dict(year=year, estimand="finite_difference_ACRT", pair_id=-1, effect="moderation",
                     **summarize(point, vector, c), coverage=min(details[s]["coverage"] for s in ["sme", "large"]),
                     n_target_firms=sum(details[s]["n_target_firms"] for s in ["sme", "large"]),
                     n_used_firms=sum(details[s]["n_used_firms"] for s in ["sme", "large"]),
                     status="completed" if np.isfinite(point) else "failed"))
    vectors["moderation"] = vector
    return pd.DataFrame(rows), np.column_stack([vectors[e] for e in PRIMARY])


def availability(frame, results, c, grid, supported):
    masks, detail = {}, []
    for scope, sub in scopes(frame).items():
        doses = np.sort(sub[c.dose].unique())
        okay = {d for d in doses if results[(float(d), float(d-c.diff_step))][0][scope]["status"] == "completed"
                and np.isfinite([results[(float(d), float(d-c.diff_step))][0][scope][k] for k in ["estimate", "se"]]).all()}
        bounds = supported[scope]
        grid_okay = {d for d in doses if len(bounds) and bounds[0] <= d <= bounds[-1]}
        masks[scope] = dict(observed=okay, grid=grid_okay)
        for d in doses:
            r = results[(float(d), float(d-c.diff_step))][0][scope]
            detail.append(dict(scope=scope, dose=d, firms=int(sub.loc[sub[c.dose].eq(d), "firm_code"].nunique()),
                               observed_computable=d in okay, grid_covered=d in grid_okay, error=r.get("error", "")))
    return masks, pd.DataFrame(detail)
