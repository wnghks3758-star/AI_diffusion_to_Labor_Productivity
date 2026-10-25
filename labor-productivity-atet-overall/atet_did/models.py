"""Training-fold preprocessing, industry-held-out models and reusable Mu fits."""

import hashlib
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from .io import input_key

from .numerics import reference_fold_variance


def industry_folds(groups, k, seed):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    k = min(k, len(unique))
    if k < 2:
        raise ValueError("Need at least two training industries")
    return [
        (
            np.flatnonzero(np.isin(groups, unique[a])),
            np.flatnonzero(np.isin(groups, unique[b])),
        )
        for a, b in KFold(k, shuffle=True, random_state=seed).split(unique)
    ]


def columns_for(frame, configured, c):
    columns = list(configured)
    if c.include_industry_dummies:
        columns += [v for v in frame if v.startswith("ind_dummy")]
    return list(dict.fromkeys(columns))


def fit_regression(X, y, groups, kind, params, c):
    """All tuning/preprocessing is inside the outer training sample.

    Lasso is sklearn Lasso with grouped CV, not a bit-identical glmnet engine.
    """
    steps = []
    if c.missing_policy == "median_indicator":
        steps.append(
            (
                "impute",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            )
        )
    elif X.isna().any().any():
        raise ValueError("Missing controls; choose median_indicator or clean the input")
    continuous = [i for i, v in enumerate(X.columns) if not v.startswith("ind_dummy")]
    if kind == "spline":
        steps.append(
            (
                "basis",
                ColumnTransformer(
                    [
                        (
                            "continuous",
                            SplineTransformer(
                                n_knots=c.spline_n_knots,
                                degree=c.spline_degree,
                                knots="quantile",
                                include_bias=False,
                                extrapolation="linear",
                            ),
                            continuous,
                        )
                    ],
                    remainder="passthrough",
                ),
            )
        )
    if kind in ("spline", "lasso", "ridge", "linear"):
        steps.append(("scale", StandardScaler()))
    if kind in ("spline", "ridge"):
        model = Ridge(**{"solver": "lsqr", **params})
    elif kind == "lasso":
        model = Lasso(**{"max_iter": 20000, "tol": 1e-5, **params})
    elif kind == "linear":
        model = LinearRegression(**params)
    elif kind == "random_forest":
        model = RandomForestRegressor(
            **{
                **dict(
                    n_estimators=300,
                    max_depth=3,
                    min_samples_leaf=8,
                    max_features=0.75,
                    random_state=c.seed,
                    n_jobs=c.model_threads,
                ),
                **params,
            }
        )
    else:
        model = XGBRegressor(
            **{
                **dict(
                    objective="reg:squarederror",
                    tree_method="exact",
                    n_estimators=300,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=7,
                    random_state=c.seed,
                    n_jobs=c.model_threads,
                ),
                **params,
            }
        )
    estimator = Pipeline(steps + [("model", model)])
    tune = kind in ("spline", "lasso", "ridge") and "alpha" not in params
    if tune:
        values = c.lasso_alphas if kind == "lasso" else c.ridge_alphas
        search = GridSearchCV(
            estimator,
            {"model__alpha": values},
            cv=industry_folds(groups, c.inner_splits, c.seed),
            scoring="neg_mean_squared_error",
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X, y)
        estimator = search.best_estimator_
        diagnostic = dict(
            selected_alpha=search.best_params_["model__alpha"],
            inner_cv_mse=-search.best_score_,
            inner_folds=min(c.inner_splits, len(np.unique(groups))),
        )
    else:
        estimator.fit(X, y)
        diagnostic = {}
    if not np.isfinite(estimator.predict(X.iloc[: min(2, len(X))])).all():
        raise ValueError("Nonfinite model prediction")
    diagnostic.update(
        model=kind,
        params=estimator.named_steps["model"].get_params(),
        missing_training_cells=int(X.isna().sum().sum()),
    )
    return estimator, diagnostic


def fit_gps_oof(ind, columns, c, implementation):
    regression_config = c.for_model("gps")
    splits, seed = c.fold_settings("gps")
    settings = dict(model=c.gps_model, params=c.gps_params, splits=splits, fold_seed=seed,
                    training_min_dose=c.training_min_dose,
                    learner_seed=c.seed, inner_splits=regression_config.inner_splits, model_threads=c.model_threads, missing_policy=c.missing_policy,
                    spline_n_knots=regression_config.spline_n_knots, spline_degree=regression_config.spline_degree,
                    ridge_alphas=regression_config.ridge_alphas, lasso_alphas=regression_config.lasso_alphas,
                    model_options=c.model_options.get("gps", {}), implementation=implementation)
    key = input_key(ind[list(dict.fromkeys(["ind_code", c.dose] + columns))], settings)
    file = Path(c.cache_dir) / "gps_oof" / f"{key}.joblib"
    if file.exists():
        result = joblib.load(file)
        result["cache_hit"] = True
        return result
    mean, variance = np.full(len(ind), np.nan), np.full(len(ind), np.nan)
    audits = []
    for fold, (train, test) in enumerate(industry_folds(ind.ind_code, splits, seed)):
        eligible = np.ones(len(ind), dtype=bool) if c.training_min_dose is None else ind[c.dose].to_numpy() >= c.training_min_dose
        train = train[eligible[train]]
        validation = test[eligible[test]]
        if len(train) < 2 or not len(validation):
            raise ValueError("Training cutoff leaves too few GPS training/variance-validation industries in a fold")
        model, diagnostic = fit_regression(ind[columns].iloc[train], ind[c.dose].to_numpy()[train],
                                           ind.ind_code.iloc[train], c.gps_model, c.gps_params, regression_config)
        mean[test] = model.predict(ind[columns].iloc[test])
        variance[test] = reference_fold_variance(ind[c.dose].to_numpy()[validation], mean[validation])
        audits.append(dict(fold=fold, train_industry_positions=train.tolist(), test_industry_positions=test.tolist(),
                           variance_validation_industry_positions=validation.tolist(),
                           gps=diagnostic, common_variance=float(variance[test[0]])))
    result = dict(mean=mean, variance=variance, folds=audits, cache_key=key, cache_hit=False)
    file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, file, compress=3)
    return result


def _fit_single_nuisance(frame, c):
    gps_columns = columns_for(frame, c.gps_features, c)
    mu_columns = columns_for(frame, c.mu_features, c) + [c.dose]
    if not gps_columns:
        raise ValueError("GPS needs at least one covariate")
    if (
        (frame.groupby("ind_code")[gps_columns + [c.dose]].nunique(dropna=False) > 1)
        .any()
        .any()
    ):
        raise ValueError("GPS covariates/dose must be constant within industry")
    ind = frame.groupby("ind_code", as_index=False).first()
    columns = list(dict.fromkeys(["ind_code", "Y_diff"] + gps_columns + mu_columns))
    fit_keys = [
        "training_min_dose",
        "psmethod", "model_options",
        "gps_model",
        "mu_model",
        "gps_params",
        "mu_params",
        "spline_n_knots",
        "spline_degree",
        "ridge_alphas",
        "lasso_alphas",
        "n_splits",
        "gps_n_splits", "mu_n_splits", "gps_fold_seed", "mu_fold_seed",
        "inner_splits",
        "seed",
        "model_threads",
        "missing_policy",
    ]
    fit_source = hashlib.sha256(
        Path(__file__).read_bytes()
        + (Path(__file__).parent / "numerics.py").read_bytes()
    ).hexdigest()
    settings = dict(
        config={k: getattr(c, k) for k in fit_keys}, implementation=fit_source
    )
    key = input_key(frame[columns], settings)
    folder = Path(c.cache_dir) / "nuisance" / key
    file = folder / "fitted.joblib"
    if file.exists():
        result = joblib.load(file)
        result["audit"]["cache_hit"] = True
        return result
    entries = []
    mu_audits = []
    mu_splits, mu_seed = c.fold_settings("mu")
    with threadpool_limits(limits=1):
        gps = fit_gps_oof(ind, gps_columns, c, fit_source) if c.psmethod == 1 else None
        for fold, (train, test) in enumerate(
            industry_folds(ind.ind_code, mu_splits, mu_seed)
        ):
            if c.training_min_dose is not None:
                train = train[ind[c.dose].to_numpy()[train] >= c.training_min_dose]
            if len(train) < 2:
                raise ValueError("Training cutoff leaves fewer than two Mu training industries in a fold")
            train_codes = ind.ind_code.iloc[train]
            test_codes = ind.ind_code.iloc[test]
            tr = np.flatnonzero(frame.ind_code.isin(train_codes))
            te = np.flatnonzero(frame.ind_code.isin(test_codes))
            mu, md = fit_regression(
                frame[mu_columns].iloc[tr],
                frame.Y_diff.to_numpy()[tr],
                frame.ind_code.iloc[tr],
                c.mu_model,
                c.mu_params,
                c.for_model("mu"),
            )
            entries.append(dict(model=mu, test=te, fold=fold))
            mu_audits.append(
                dict(
                    fold=fold,
                    train_industry_positions=train.tolist(),
                    test_industry_positions=test.tolist(),
                    train_firms=len(tr),
                    test_firms=len(te),
                    mu=md,
                )
            )
    mapping = frame.ind_code.map(
        pd.Series(np.arange(len(ind)), index=ind.ind_code)
    ).to_numpy(int)
    shared = c.fold_settings("gps") == c.fold_settings("mu")
    paired = [{**g, **m} for g, m in zip(gps["folds"], mu_audits)] if shared and gps is not None else []
    result = dict(
        mean=gps["mean"][mapping] if gps is not None else None,
        variance=gps["variance"][mapping] if gps is not None else None,
        models=entries,
        mu_columns=mu_columns,
        key=key,
        folder=str(folder),
        audit=dict(
            cache_key=key,
            cache_hit=False,
            gps_columns=gps_columns,
            mu_columns=mu_columns,
            folds=paired,
            gps_folds=gps["folds"] if gps is not None else [], mu_folds=mu_audits,
            gps_cache_hit=gps["cache_hit"] if gps is not None else None,
            gps_cache_key=gps["cache_key"] if gps is not None else None,
            gps_n_splits=len(gps["folds"]) if gps is not None else min(c.fold_settings("gps")[0], len(ind)),
            mu_n_splits=len(mu_audits),
            gps_fold_seed=c.fold_settings("gps")[1], mu_fold_seed=mu_seed,
            shared_fold_partition=shared,
            variance_source="outer validation industry residual MSE" if gps is not None else None,
            psmethod=c.psmethod,
            density_floor=None,
        ),
    )
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, file, compress=3)
    return result


def fit_nuisance(frame, c):
    """Combine repeated held-out predictions, never the final effect estimates.

    Each repeat advances the GPS and Mu fold seeds by one. Learner seeds stay
    fixed. PS1 remains Gaussian: its conditional mean and validation-fold
    residual variance are each combined by their componentwise median.
    """
    c = c.resolve_models()
    gps_splits, gps_seed = c.fold_settings("gps")
    mu_splits, mu_seed = c.fold_settings("mu")
    repeats = []
    for repeat in range(c.n_repeats):
        single = replace(
            c, n_repeats=1, prediction_aggregation="single",
            gps_fold_seed=(gps_seed + repeat) % (2**32),
            mu_fold_seed=(mu_seed + repeat) % (2**32),
        )
        fitted = _fit_single_nuisance(frame, single)
        repeats.append(fitted)
    if len(repeats) == 1:
        result = repeats[0]
        result["n_repeats"] = 1
        result["prediction_aggregation"] = c.prediction_aggregation
        result["audit"].update(
            n_repeats=1, prediction_aggregation=c.prediction_aggregation,
            gps_fold_seeds=[gps_seed % (2**32)], mu_fold_seeds=[mu_seed % (2**32)],
        )
        return result
    keys = [f["key"] for f in repeats]
    key = hashlib.sha256(("median:" + ":".join(keys)).encode()).hexdigest()[:24]
    entries = [{**entry, "repeat": repeat} for repeat, fitted in enumerate(repeats) for entry in fitted["models"]]
    audit = dict(
        cache_key=key, cache_hit=all(f["audit"]["cache_hit"] for f in repeats),
        gps_columns=repeats[0]["audit"]["gps_columns"],
        mu_columns=repeats[0]["mu_columns"],
        gps_n_splits=repeats[0]["audit"]["gps_n_splits"],
        mu_n_splits=repeats[0]["audit"]["mu_n_splits"],
        n_repeats=c.n_repeats, prediction_aggregation="median",
        gps_fold_seeds=[(gps_seed + r) % (2**32) for r in range(c.n_repeats)],
        mu_fold_seeds=[(mu_seed + r) % (2**32) for r in range(c.n_repeats)],
        gps_folds=[dict(repeat=r, **fold) for r, f in enumerate(repeats) for fold in f["audit"]["gps_folds"]],
        mu_folds=[dict(repeat=r, **fold) for r, f in enumerate(repeats) for fold in f["audit"]["mu_folds"]],
        repeats=[f["audit"] for f in repeats],
        gps_aggregation="componentwise median of conditional means and fold residual variances" if c.psmethod == 1 else "kernel predictions are fitted and combined separately",
        mu_aggregation="componentwise median of held-out outcome predictions at each control dose",
        variance_source="median of repeat-specific outer validation industry residual MSE" if c.psmethod == 1 else None,
        psmethod=c.psmethod,
        density_floor=None,
    )
    return dict(
        mean=np.median(np.stack([f["mean"] for f in repeats]), axis=0) if c.psmethod == 1 else None,
        variance=np.median(np.stack([f["variance"] for f in repeats]), axis=0) if c.psmethod == 1 else None,
        models=entries, mu_columns=repeats[0]["mu_columns"],
        n_repeats=c.n_repeats, prediction_aggregation="median",
        key=key, folder=str(Path(c.cache_dir) / "nuisance"), audit=audit,
    )


def predict_controls(frame, centers, fitted, c):
    """Yield small batches of OOF Mu predictions; fitted models are reused."""
    centers = np.asarray(centers, float)
    if (
        centers.ndim != 1
        or not len(centers)
        or not np.isfinite(centers).all()
        or np.any(np.diff(centers) <= 0)
    ):
        raise ValueError("Sorted finite unique centers required")
    for start in range(0, len(centers), c.prediction_batch_points):
        values = centers[start : start + c.prediction_batch_points]
        n_repeats = fitted.get("n_repeats", 1)
        pred = np.full((n_repeats, len(values), len(frame)), np.nan)
        for entry in fitted["models"]:
            test = entry["test"]
            repeat = entry.get("repeat", 0)
            X = frame[fitted["mu_columns"]].iloc[test]
            expanded = pd.concat(
                [X.assign(**{c.dose: float(d)}) for d in values], ignore_index=True
            )
            pieces = []
            for p in range(0, len(expanded), c.max_prediction_rows):
                pieces.append(
                    entry["model"].predict(expanded.iloc[p : p + c.max_prediction_rows])
                )
            # Index after selecting the repeat so advanced indexing cannot swap axes.
            pred[repeat][:, test] = np.concatenate(pieces).reshape(len(values), len(test))
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite OOF Mu prediction")
        yield start, pred[0] if n_repeats == 1 else np.median(pred, axis=0)
