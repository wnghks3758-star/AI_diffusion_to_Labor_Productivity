"""PS2: separate, industry-held-out regressions of normalized kernel weights.

Reference: causalweight didcontDMLpanel.R, psmethod=2. GPS is fitted on the
industry unit used by this application. No Gaussian residual variance, density
floor, clipping, or interpolation is introduced.
"""
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .io import input_key
from .models import columns_for, fit_regression, industry_folds
from .numerics import np_kernel


def fit_kernel_gps(frame, centers, h, c):
    c = c.resolve_models()
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        raise ValueError("PS2 evaluation points must be finite, nonempty and strictly increasing")
    columns = columns_for(frame, c.gps_features, c)
    if (frame.groupby("ind_code")[columns + [c.dose]].nunique(dropna=False) > 1).any().any():
        raise ValueError("GPS covariates and dose must be constant within industry")
    ind = frame.groupby("ind_code", as_index=False).first()
    X, D = ind[columns], ind[c.dose].to_numpy()
    raw = np.column_stack([np_kernel(D, center, h) for center in centers])
    eligible = np.ones(len(ind), dtype=bool) if c.training_min_dose is None else D >= c.training_min_dose
    if eligible.sum() < 2:
        raise ValueError("Training cutoff leaves fewer than two PS2 training industries")
    # Excluded doses must not determine the scale of training targets.
    normalizers = raw[eligible].sum(axis=0)
    targets = np.divide(raw, normalizers[None, :], out=np.zeros_like(raw), where=normalizers[None, :] > 0)
    splits, seed = c.fold_settings("gps")
    regression_config = c.for_model("gps")
    repeated, audits = [], []
    for repeat in range(c.n_repeats):
        fold_seed = (seed + repeat) % (2**32)
        settings = dict(
            psmethod=2, centers=centers.tolist(), h=h, normalization="eligible_industry_kernel_mass",
            training_min_dose=c.training_min_dose,
            model=c.gps_model, params=c.gps_params, options=c.model_options.get("gps", {}),
            spline_n_knots=regression_config.spline_n_knots, spline_degree=regression_config.spline_degree,
            ridge_alphas=regression_config.ridge_alphas, lasso_alphas=regression_config.lasso_alphas,
            splits=splits, fold_seed=fold_seed, inner_splits=regression_config.inner_splits,
            learner_seed=c.seed, model_threads=c.model_threads, missing_policy=c.missing_policy,
        )
        key = input_key(ind[["ind_code", c.dose] + columns], settings)
        file = Path(c.cache_dir) / "kernel_gps" / f"{key}.joblib"
        if file.exists():
            stored = joblib.load(file)
            predictions, folds = stored["predictions"], stored["folds"]
            hit = True
        else:
            predictions = np.full_like(targets, np.nan)
            folds = []
            with threadpool_limits(limits=1):
                for fold, (train, test) in enumerate(industry_folds(ind.ind_code, splits, fold_seed)):
                    train = train[eligible[train]]
                    if len(train) < 2:
                        raise ValueError("Training cutoff leaves too few PS2 training industries in a fold")
                    zero = []
                    for j in range(len(centers)):
                        if not np.any(targets[train, j] > 0):
                            predictions[test, j] = 0.
                            zero.append(j)
                            continue
                        model, _ = fit_regression(
                            X.iloc[train], targets[train, j], ind.ind_code.iloc[train],
                            c.gps_model, c.gps_params, regression_config,
                        )
                        predictions[test, j] = model.predict(X.iloc[test])
                    folds.append(dict(
                        fold=fold, train_industry_positions=train.tolist(), test_industry_positions=test.tolist(),
                        zero_training_target_columns=zero,
                        mse=np.mean((predictions[test[eligible[test]]] - targets[test[eligible[test]]])**2, axis=0).tolist() if eligible[test].any() else None,
                        diagnostic_industry_positions=test[eligible[test]].tolist(),
                    ))
            if not np.isfinite(predictions).all():
                raise ValueError("Nonfinite held-out PS2 kernel predictions")
            file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(predictions=predictions, folds=folds), file, compress=3)
            hit = False
        repeated.append(predictions)
        audits.append(dict(repeat=repeat, fold_seed=fold_seed, cache_key=key, cache_hit=hit, folds=folds))
        print(f"PS2 GPS repeat {repeat + 1}/{c.n_repeats}: {len(centers)} kernel targets (cached={hit})", flush=True)
    normalized = repeated[0] if len(repeated) == 1 else np.median(np.stack(repeated), axis=0)
    # Undo target normalization. This point-specific positive scaling cancels
    # from normalized ATET control weights, and leaves the reference fit intact.
    predictions = normalized * normalizers[None, :]
    return dict(
        centers=centers, industries=ind.ind_code.to_numpy(), predictions=predictions,
        audit=dict(
            psmethod=2, target="E[normalized K_h(D-d)|X]", columns=columns,
            model=c.gps_model, params=c.gps_params, model_options=c.model_options.get("gps", {}),
            n_repeats=c.n_repeats, prediction_aggregation=c.prediction_aggregation,
            n_splits=len(audits[0]["folds"]), repeats=audits,
            cache_hit=all(a["cache_hit"] for a in audits), centers=centers.tolist(),
            kernel_normalizers=normalizers.tolist(), bandwidth=h, kernel_halfwidth=float(np.sqrt(5)*h),
            training_min_dose=c.training_min_dose,training_eligible_industries=int(eligible.sum()),
            gaussian_assumption=False, residual_variance_model=False, density_floor=None,
            negative_predictions=int((predictions < 0).sum()),
            zero_predictions=int((predictions == 0).sum()),
            positivity_policy="Predictions must be positive on active kernel support; invalid pairs are marked failed without clipping",
        ),
    )


def kernel_log_ratio(frame, dt, dc, kernel, high, low):
    points = kernel["centers"]
    positions = np.searchsorted(points, [dt, dc])
    if (positions >= len(points)).any() or not np.array_equal(points[positions], [dt, dc]):
        raise ValueError("Missing PS2 predictions for the requested treatment pair")
    mapping = np.searchsorted(kernel["industries"], frame.ind_code.to_numpy())
    if (mapping >= len(kernel["industries"])).any() or not np.array_equal(kernel["industries"][mapping], frame.ind_code.to_numpy()):
        raise ValueError("PS2 industry order does not match the estimation sample")
    qt = kernel["predictions"][mapping, positions[0]]
    qc = kernel["predictions"][mapping, positions[1]]
    if not np.isfinite(qt).all() or not np.isfinite(qc).all():
        raise ValueError("Nonfinite PS2 kernel prediction")
    if ((high > 0) & (qt <= 0)).any() or ((low > 0) & ((qt <= 0) | (qc <= 0))).any():
        raise ValueError("Nonpositive PS2 prediction on active kernel support; no clipping or density floor")
    ratio = np.zeros(len(frame))
    positive = (qt > 0) & (qc > 0)
    ratio[positive] = np.log(qt[positive]) - np.log(qc[positive])
    return ratio
