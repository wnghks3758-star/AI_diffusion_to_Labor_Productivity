"""Industry-held-out nuisance fits and disk-backed counterfactual predictions."""

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from .io import input_key, versions, write_json


def feature_names(frame, configured, config):
    dummies = (
        [c for c in frame if c.startswith("ind_dummy")]
        if config.include_industry_dummies
        else []
    )
    return list(dict.fromkeys(configured + dummies))


def make_folds(frame, config, repetition=0, n_splits=None):
    group_info = frame[["ind_code", "ind_refined"]].drop_duplicates()
    if group_info.ind_code.duplicated().any():
        raise ValueError("Industry has multiple strata")
    requested = n_splits or config.n_splits
    actual = min(
        requested, int(group_info.groupby("ind_refined").ind_code.nunique().min())
    )
    if actual < 2:
        raise ValueError(
            "Each industry stratum must contain >=2 industries; change dummy rules"
        )
    cv = StratifiedGroupKFold(
        n_splits=actual, shuffle=True, random_state=config.seed + repetition
    )
    folds = list(cv.split(frame, frame.ind_refined, frame.ind_code))
    for train, test in folds:
        assert not set(frame.iloc[train].ind_code) & set(frame.iloc[test].ind_code)
    return folds


def gps_transform(columns, config):
    continuous = [c for c in columns if not c.startswith("ind_dummy")]
    dummies = [c for c in columns if c.startswith("ind_dummy")]
    if config.gps_model == "linear":
        return StandardScaler()
    parts = []
    if continuous:
        parts.append(
            (
                "continuous",
                SplineTransformer(
                    n_knots=config.gps_n_knots,
                    degree=config.gps_degree,
                    knots="quantile",
                    include_bias=False,
                    extrapolation="linear",
                ),
                continuous,
            )
        )
    if dummies:
        parts.append(("dummies", "passthrough", dummies))
    return Pipeline(
        [("features", ColumnTransformer(parts)), ("scaler", StandardScaler())]
    )


def _gps_fold(X, train, test, config):
    transform = gps_transform(list(X.columns), config)
    z_train = transform.fit_transform(X.iloc[train])
    z_test = transform.transform(X.iloc[test])
    offset = z_train.mean(axis=0)
    U, singular, Vt = np.linalg.svd(z_train - offset, full_matrices=False)
    return train, test, U, singular, (z_test - offset) @ Vt.T


def _ridge_predict(fold, y, alphas, predict_train=False):
    train, _, U, singular, test_projection = fold
    target = y[train]
    mean = target.mean()
    projection = U.T @ (target - mean)
    alphas = np.atleast_1d(alphas).astype(float)
    coefficients = (
        singular[:, None]
        * projection[:, None]
        / (singular[:, None] ** 2 + alphas[None, :])
    )
    predicted = test_projection @ coefficients + mean
    if predict_train:
        return predicted, U @ (singular[:, None] * coefficients) + mean
    return predicted


def _best_alpha(folds, y, alphas):
    losses = [
        np.mean((_ridge_predict(fold, y, alphas) - y[fold[1], None]) ** 2, axis=0)
        for fold in folds
    ]
    return alphas[np.argmin(np.mean(losses, axis=0))]


def fit_gps(frame, config):
    """Original 1772 learner: global CV alpha, OOF residuals and smearing."""
    columns = feature_names(frame, config.gps_features, config)
    if not columns:
        raise ValueError("GPS requires at least one feature")
    if (frame.groupby("ind_code")[columns].nunique(dropna=False) > 1).any().any():
        raise ValueError("GPS features must be constant within each industry")
    training = frame.groupby("ind_code", as_index=False).agg(
        {
            c: "first"
            for c in dict.fromkeys(["ind_code", "ind_refined", config.dose] + columns)
            if c != "ind_code"
        }
    )
    if training[columns + [config.dose]].isna().any().any():
        raise ValueError("Missing GPS inputs")
    X = training[columns]
    y = training[config.dose].to_numpy(float)
    settings = {
        k: getattr(config, k)
        for k in [
            "gps_model",
            "gps_n_knots",
            "gps_degree",
            "ridge_alphas",
            "gps_variance_floor",
            "n_splits",
            "n_repeats",
            "seed",
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
        labels = training.ind_refined.to_numpy()
        k = min(config.n_splits, int(pd.Series(labels).value_counts().min()))
        if k < 2:
            raise ValueError("Each GPS stratum needs at least two industries")
        with threadpool_limits(limits=1):
            folds_by_rep = [
                [
                    _gps_fold(X, tr, te, config)
                    for tr, te in StratifiedKFold(
                        n_splits=k, shuffle=True, random_state=config.seed + rep
                    ).split(X, labels)
                ]
                for rep in range(config.n_repeats)
            ]
            alpha = _best_alpha(folds_by_rep[0], y, config.ridge_alphas)
            means = np.empty((config.n_repeats, len(y)))
            scales = np.empty_like(means)
            for rep, folds in enumerate(folds_by_rep):
                for fold in folds:
                    means[rep, fold[1]] = _ridge_predict(fold, y, alpha)[:, 0]
            mean = np.median(means, axis=0)
            eps = 1e-6
            residual_sq = (y - mean) ** 2
            target = np.log(residual_sq + eps)
            alpha_v = _best_alpha(folds_by_rep[0], target, config.ridge_alphas)
            for rep, folds in enumerate(folds_by_rep):
                for fold in folds:
                    log_test, log_train = _ridge_predict(
                        fold, target, alpha_v, predict_train=True
                    )
                    smear = np.mean(np.exp(target[fold[0]] - log_train[:, 0]))
                    floor = max(
                        np.percentile(residual_sq[fold[0]], 1),
                        config.gps_variance_floor,
                    )
                    scales[rep, fold[1]] = np.sqrt(
                        np.maximum(np.exp(log_test[:, 0]) * smear - eps, floor)
                    )
            predicted = {
                "mean": mean,
                "sigma": np.median(scales, axis=0),
                "mean_alpha": alpha,
                "variance_alpha": alpha_v,
                "actual_splits": k,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(predicted, path)
    mapping = frame.ind_code.map(
        pd.Series(np.arange(len(training)), index=training.ind_code)
    ).to_numpy(int)
    return (
        predicted["mean"][mapping],
        predicted["sigma"][mapping],
        {
            "cache_hit": hit,
            "cache_key": key,
            "features": columns,
            "n_training_rows": len(training),
            **{
                k: predicted[k]
                for k in ["mean_alpha", "variance_alpha", "actual_splits"]
            },
        },
    )


def mu_estimator(config):
    if config.mu_model == "xgboost":
        parameters = {
            "objective": "reg:squarederror",
            "tree_method": "exact",
            "random_state": config.seed,
            "n_jobs": config.model_threads,
            **config.mu_params,
        }
        return XGBRegressor(**parameters)
    alpha = float(config.mu_params.get("alpha", 1.0))
    if config.mu_model == "ridge":
        return Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha, solver="svd"))]
        )
    return Pipeline(
        [
            (
                "spline",
                SplineTransformer(
                    n_knots=int(config.mu_params.get("n_knots", 3)),
                    degree=int(config.mu_params.get("degree", 3)),
                    include_bias=False,
                    extrapolation="linear",
                ),
            ),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver="svd")),
        ]
    )


def _tune_mu(X, y, frame, config):
    estimator = mu_estimator(config)
    if not config.mu_search_n_iter or not config.mu_search_space:
        return {}, None
    if config.mu_model != "xgboost":
        space = {"ridge__alpha": config.ridge_alphas}
    else:
        space = config.mu_search_space
    folds = make_folds(frame, config, n_splits=config.inner_splits)
    candidates = int(np.prod([len(v) for v in space.values()]))
    search = RandomizedSearchCV(
        estimator,
        space,
        n_iter=min(config.mu_search_n_iter, candidates),
        cv=folds,
        scoring="neg_mean_squared_error",
        n_jobs=config.model_jobs,
        random_state=config.seed,
        refit=False,
        pre_dispatch=config.model_jobs,
    )
    with joblib.parallel_config(backend="threading"), threadpool_limits(limits=1):
        search.fit(X, y)
    return search.best_params_, float(search.best_score_)


def fit_mu(frame, config):
    from .config import INDUSTRY_CONTROLS

    fixed = config.mu_params_by_year.get(str(int(frame.Year.iloc[0])))
    if fixed is not None:
        config = replace(
            config, mu_params={**config.mu_params, **fixed}, mu_search_n_iter=0
        )
    firm = [c for c in config.mu_features if c not in INDUSTRY_CONTROLS]
    industry = [c for c in config.mu_features if c in INDUSTRY_CONTROLS]
    columns = feature_names(frame, firm, config) + industry + [config.dose]
    X = frame[columns].to_numpy(float)
    y = frame.Y_diff.to_numpy(float)
    settings = {
        k: getattr(config, k)
        for k in [
            "mu_model",
            "mu_params",
            "mu_search_space",
            "mu_search_n_iter",
            "n_splits",
            "n_repeats",
            "inner_splits",
            "seed",
            "tuning_scope",
        ]
    }
    settings["versions"] = versions()
    key = input_key(frame[["ind_code", "ind_refined", "Y_diff"] + columns], settings)
    folder = Path(config.cache_dir) / "mu" / key
    manifest_path = folder / "manifest.json"
    hit = manifest_path.exists()
    if hit:
        manifest = json.loads(manifest_path.read_text())
        if not all((folder / entry["file"]).exists() for entry in manifest["models"]):
            hit = False
    if not hit:
        folder.mkdir(parents=True, exist_ok=True)
        best, cv_score = (
            _tune_mu(X, y, frame, config)
            if config.tuning_scope == "global_cv"
            else ({}, None)
        )
        tasks = [
            (rep, j, train, test)
            for rep in range(config.n_repeats)
            for j, (train, test) in enumerate(make_folds(frame, config, rep))
        ]

        def fit_task(task):
            rep, j, train, test = task
            params, score = (best, cv_score)
            if config.tuning_scope == "nested":
                params, score = _tune_mu(
                    X[train], y[train], frame.iloc[train].reset_index(drop=True), config
                )
            model = mu_estimator(config)
            model.set_params(**params)
            model.fit(X[train], y[train])
            filename = f"rep{rep:03d}_fold{j:02d}.joblib"
            joblib.dump(
                {"model": model, "test": test, "train": train}, folder / filename
            )
            return {
                "file": filename,
                "repetition": rep,
                "fold": j,
                "train_rows": len(train),
                "test_rows": len(test),
                "best_params": params,
                "cv_score": score,
            }

        with threadpool_limits(limits=1):
            # Nested searches already parallelize; avoid nested thread pools.
            jobs = 1 if config.tuning_scope == "nested" else config.model_jobs
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                entries = list(executor.map(fit_task, tasks))
        manifest = {
            "models": entries,
            "features": columns,
            "cache_key": key,
            "settings": settings,
        }
        write_json(manifest_path, manifest)
    return (
        folder,
        manifest,
        {
            "cache_hit": hit,
            "cache_key": key,
            "features": columns,
            "models": manifest["models"],
            "actual_splits": len(manifest["models"]) // config.n_repeats,
            "n_repeats": config.n_repeats,
        },
    )


def tree_ranges(model, d_values, feature_index):
    if not hasattr(model, "get_booster"):
        return [(j, j + 1) for j in range(len(d_values))]
    thresholds = []
    for tree in model.get_booster().get_dump(dump_format="json"):
        nodes = [json.loads(tree)]
        while nodes:
            node = nodes.pop()
            if "leaf" in node:
                continue
            if node["split"] == f"f{feature_index}":
                thresholds.append(node["split_condition"])
            nodes.extend(node["children"])
    bins = np.searchsorted(
        np.unique(np.asarray(thresholds, dtype=np.float32)),
        d_values.astype(np.float32),
        side="right",
    )
    edges = np.r_[0, np.flatnonzero(np.diff(bins)) + 1, len(d_values)]
    return list(zip(edges[:-1], edges[1:]))


@contextmanager
def predict_mu(frame, centers, config, folder, manifest):
    """Median across repetitions, stored on disk; one fitted fold resident at a time."""
    centers = np.asarray(centers, float)
    temporary = Path(
        tempfile.mkdtemp(prefix="mu_predictions_", dir=Path(config.cache_dir))
    )
    if (
        centers.ndim != 1
        or not len(centers)
        or not np.isfinite(centers).all()
        or np.any(np.diff(centers) <= 0)
    ):
        shutil.rmtree(temporary)
        raise ValueError(
            "Prediction centers must be nonempty, finite and strictly increasing"
        )
    prediction_key = input_key(
        frame[manifest["features"]],
        {
            "centers": centers.tolist(),
            "mu_cache_key": manifest["cache_key"],
            "n_repeats": config.n_repeats,
        },
    )
    cache_file = folder / f"median_{prediction_key}.npy"

    class CachedPredictions:
        def __init__(self, hit, extra=None):
            self.audit = {"cache_hit": hit, "n_centers": len(centers), **(extra or {})}

        def batches(self):
            data = np.load(cache_file, mmap_mode="r")
            try:
                for start in range(0, len(centers), config.prediction_batch_points):
                    stop = min(start + config.prediction_batch_points, len(centers))
                    yield start, stop, np.asarray(data[start:stop])
            finally:
                del data

    if cache_file.exists():
        shutil.rmtree(temporary)
        yield CachedPredictions(True)
        return
    shape = (config.n_repeats, len(centers), len(frame))
    store = np.memmap(
        temporary / "repeats.dat", mode="w+", dtype="float64", shape=shape
    )
    store[:] = np.nan
    X = frame[manifest["features"]].to_numpy(float)
    predicted_rows = 0
    try:
        with threadpool_limits(limits=1):
            for entry in manifest["models"]:
                bundle = joblib.load(folder / entry["file"])
                model = bundle["model"]
                test = bundle["test"]
                ranges = tree_ranges(model, centers, X.shape[1] - 1)
                for first, last in ranges:
                    values = X[test].copy()
                    values[:, -1] = centers[first]
                    prediction = np.empty(len(test))
                    for start in range(0, len(test), config.max_prediction_rows):
                        stop = min(start + config.max_prediction_rows, len(test))
                        prediction[start:stop] = model.predict(values[start:stop])
                    store[entry["repetition"], first:last, test] = prediction[:, None]
                    predicted_rows += len(test)
                del bundle, model
            store.flush()
        median_file = temporary / "median.npy"
        median = np.lib.format.open_memmap(
            median_file, mode="w+", dtype="float64", shape=(len(centers), len(frame))
        )
        for start in range(0, len(centers), config.prediction_batch_points):
            stop = min(start + config.prediction_batch_points, len(centers))
            block = np.median(np.asarray(store[:, start:stop, :]), axis=0)
            if not np.isfinite(block).all():
                raise ValueError("Incomplete Mu cross-fit predictions")
            median[start:stop] = block
        median.flush()
        del median
        import os

        os.replace(median_file, cache_file)
        yield CachedPredictions(
            False,
            {
                "disk_prediction_bytes": int(np.prod(shape)) * 8,
                "predicted_rows": predicted_rows,
                "original_prediction_rows": int(np.prod(shape)),
            },
        )
    finally:
        del store
        shutil.rmtree(temporary, ignore_errors=True)
