"""Training-only GPS fits for small industry samples; no held-out dose calibration."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .io import input_key


def gps_estimator(config, columns):
    if config.gps_model == "random_forest":
        params = dict(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=10,
            max_features=0.75,
            random_state=config.seed,
            n_jobs=config.model_threads,
        )
        return RandomForestRegressor(**{**params, **config.gps_params})
    from .models import gps_transform

    return Pipeline(
        [
            ("features", gps_transform(columns, config)),
            ("ridge", RidgeCV(alphas=config.ridge_alphas, gcv_mode="svd")),
        ]
    )


def nested_fold(X, y, labels, train, test, config):
    """Every target and tuning decision uses train only, including variance targets."""
    model = gps_estimator(config, list(X.columns)).fit(X.iloc[train], y[train])
    mean = model.predict(X.iloc[test])
    X_train = X.iloc[train].reset_index(drop=True)
    y_train = y[train]
    inner_labels = np.asarray(labels)[train]
    k = min(config.inner_splits, int(pd.Series(inner_labels).value_counts().min()))
    if k >= 2:
        inner = StratifiedKFold(k, shuffle=True, random_state=config.seed)
        folds = inner.split(X_train, inner_labels)
    else:
        k = min(config.inner_splits, len(train))
        if k < 2:
            raise ValueError("At least two training industries are required")
        folds = KFold(k, shuffle=True, random_state=config.seed).split(X_train)
    oof = np.empty(len(train))
    for a, b in folds:
        inner_model = gps_estimator(config, list(X.columns)).fit(
            X_train.iloc[a], y_train[a]
        )
        oof[b] = inner_model.predict(X_train.iloc[b])
    residual_sq = (y_train - oof) ** 2
    target = np.log(residual_sq + 1e-6)
    variance = gps_estimator(config, list(X.columns)).fit(X_train, target)
    smear = np.mean(np.exp(target - variance.predict(X_train)))
    floor = max(float(np.percentile(residual_sq, 1)), config.gps_variance_floor)
    sigma = np.sqrt(
        np.maximum(np.exp(variance.predict(X.iloc[test])) * smear - 1e-6, floor)
    )
    return (
        mean,
        sigma,
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "inner_splits": k,
            "variance_floor": floor,
        },
    )


def fit_nested_gps(frame, training, columns, config):
    settings = {
        k: getattr(config, k)
        for k in [
            "gps_model",
            "gps_fit_scope",
            "gps_params",
            "gps_n_knots",
            "gps_degree",
            "ridge_alphas",
            "gps_variance_floor",
            "n_splits",
            "n_repeats",
            "inner_splits",
            "seed",
            "model_threads",
        ]
    }
    key = input_key(
        training[["ind_code", "ind_refined", config.dose] + columns], settings
    )
    path = Path(config.cache_dir) / "gps" / f"{key}.joblib"
    hit = path.exists()
    if hit:
        predicted = joblib.load(path)
    else:
        X = training[columns]
        y = training[config.dose].to_numpy(float)
        labels = training.ind_refined.to_numpy()
        k = min(config.n_splits, int(pd.Series(labels).value_counts().min()))
        if k < 2:
            raise ValueError("Each GPS stratum needs at least two industries")
        means = np.empty((config.n_repeats, len(y)))
        scales = np.empty_like(means)
        folds = []
        with threadpool_limits(limits=1):
            for rep in range(config.n_repeats):
                cv = StratifiedKFold(k, shuffle=True, random_state=config.seed + rep)
                for train, test in cv.split(X, labels):
                    mean, sigma, audit = nested_fold(X, y, labels, train, test, config)
                    means[rep, test], scales[rep, test] = mean, sigma
                    folds.append({"repetition": rep, **audit})
        predicted = {
            "mean": np.median(means, axis=0),
            "sigma": np.median(scales, axis=0),
            "actual_splits": k,
            "folds": folds,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(predicted, path)
    mapping = frame.ind_code.map(
        pd.Series(np.arange(len(training)), index=training.ind_code)
    ).to_numpy(int)
    y = training[config.dose].to_numpy(float)
    return (
        predicted["mean"][mapping],
        predicted["sigma"][mapping],
        {
            "cache_hit": hit,
            "cache_key": key,
            "features": columns,
            "n_training_rows": len(training),
            "actual_splits": predicted["actual_splits"],
            "fit_scope": "nested",
            "model": config.gps_model,
            "folds": predicted["folds"],
            "oof_mse": float(np.mean((y - predicted["mean"]) ** 2)),
            "oof_gaussian_nll": float(
                -np.mean(norm.logpdf(y, predicted["mean"], predicted["sigma"]))
            ),
            "density_assumption": "heteroskedastic Gaussian; RF changes regressions, not this assumption",
        },
    )
