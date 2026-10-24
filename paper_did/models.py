"""Training-fold preprocessing, industry-held-out models and reusable Mu fits."""

import hashlib
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

from labor_did.io import input_key

from .numerics import reference_fold_variance


def source_hash():
    h = hashlib.sha256()
    files = sorted(Path(__file__).parent.glob("*.py"))
    legacy = Path(__file__).resolve().parents[1] / "labor_did"
    files += [
        legacy / n
        for n in [
            "data.py",
            "industry.py",
            "config.py",
            "support.py",
            "weights.py",
            "aggregation.py",
            "io.py",
        ]
    ]
    for path in files:
        h.update(path.read_bytes())
    return h.hexdigest()


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
    splits, seed = c.fold_settings("gps")
    settings = dict(model=c.gps_model, params=c.gps_params, splits=splits, fold_seed=seed,
                    learner_seed=c.seed, inner_splits=c.inner_splits, model_threads=c.model_threads, missing_policy=c.missing_policy,
                    spline_n_knots=c.spline_n_knots, spline_degree=c.spline_degree,
                    ridge_alphas=c.ridge_alphas, lasso_alphas=c.lasso_alphas, implementation=implementation)
    key = input_key(ind[list(dict.fromkeys(["ind_code", c.dose] + columns))], settings)
    file = Path(c.cache_dir) / "gps_oof" / f"{key}.joblib"
    if file.exists():
        result = joblib.load(file)
        result["cache_hit"] = True
        return result
    mean, variance = np.full(len(ind), np.nan), np.full(len(ind), np.nan)
    audits = []
    for fold, (train, test) in enumerate(industry_folds(ind.ind_code, splits, seed)):
        model, diagnostic = fit_regression(ind[columns].iloc[train], ind[c.dose].to_numpy()[train],
                                           ind.ind_code.iloc[train], c.gps_model, c.gps_params, c)
        mean[test] = model.predict(ind[columns].iloc[test])
        variance[test] = reference_fold_variance(ind[c.dose].to_numpy()[test], mean[test])
        audits.append(dict(fold=fold, train_industry_positions=train.tolist(), test_industry_positions=test.tolist(),
                           gps=diagnostic, common_variance=float(variance[test[0]])))
    result = dict(mean=mean, variance=variance, folds=audits, cache_key=key, cache_hit=False)
    file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, file, compress=3)
    return result


def fit_nuisance(frame, c):
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
        gps = fit_gps_oof(ind, gps_columns, c, fit_source)
        for fold, (train, test) in enumerate(
            industry_folds(ind.ind_code, mu_splits, mu_seed)
        ):
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
                c,
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
    paired = [{**g, **m} for g, m in zip(gps["folds"], mu_audits)] if shared else []
    result = dict(
        mean=gps["mean"][mapping],
        variance=gps["variance"][mapping],
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
            gps_folds=gps["folds"], mu_folds=mu_audits,
            gps_cache_hit=gps["cache_hit"], gps_cache_key=gps["cache_key"],
            gps_n_splits=len(gps["folds"]), mu_n_splits=len(mu_audits),
            gps_fold_seed=c.fold_settings("gps")[1], mu_fold_seed=mu_seed,
            shared_fold_partition=shared,
            variance_source="outer validation industry residual MSE",
            density_floor=None,
        ),
    )
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, file, compress=3)
    return result


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
        pred = np.full((len(values), len(frame)), np.nan)
        for entry in fitted["models"]:
            test = entry["test"]
            X = frame[fitted["mu_columns"]].iloc[test]
            expanded = pd.concat(
                [X.assign(**{c.dose: float(d)}) for d in values], ignore_index=True
            )
            pieces = []
            for p in range(0, len(expanded), c.max_prediction_rows):
                pieces.append(
                    entry["model"].predict(expanded.iloc[p : p + c.max_prediction_rows])
                )
            pred[:, test] = np.concatenate(pieces).reshape(len(values), len(test))
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite OOF Mu prediction")
        yield start, pred
