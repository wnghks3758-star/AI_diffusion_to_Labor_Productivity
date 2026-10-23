"""psmethod=2: cross-fit conditional expectations of kernel responses, no Gaussian fit."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .io import input_key
from .models import feature_names
from .score import normalized_atet_score
from .weights import effective_sample_size, epanechnikov_kernel


@dataclass
class KernelGpsResult:
    centers: np.ndarray
    industries: np.ndarray
    predictions: np.ndarray  # repetition x industry x evaluation point
    firm_industry_positions: np.ndarray
    audit: dict

    def firm_scores(self, repetition=None):
        values = (
            np.median(self.predictions, axis=0)
            if repetition is None
            else self.predictions[repetition]
        )
        return values[self.firm_industry_positions]


def fit_kernel_gps(
    frame, centers, halfwidth, config, *, mode="shared", unconditional=False
):
    """Halfwidth is physical support radius (sqrt(5)*h for the R np kernel).

    Shared RF regresses all kernel responses jointly. Separate RF follows R's
    one-regression-per-evaluation-point structure. No density floor is applied.
    """
    centers = np.asarray(centers, float)
    if (
        centers.ndim != 1
        or not len(centers)
        or not np.isfinite(centers).all()
        or np.any(np.diff(centers) <= 0)
    ):
        raise ValueError("centers must be finite, nonempty and strictly increasing")
    if (
        not np.isfinite(halfwidth)
        or halfwidth <= 0
        or mode not in ("shared", "separate")
    ):
        raise ValueError("positive halfwidth and shared/separate mode are required")
    columns = feature_names(frame, config.gps_features, config)
    if (
        (frame.groupby("ind_code")[columns + [config.dose]].nunique(dropna=False) > 1)
        .any()
        .any()
    ):
        raise ValueError(
            "Industry GPS inputs and dose must be constant within industry"
        )
    training = frame.groupby("ind_code", as_index=False).first()
    X = training[columns].to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Kernel GPS requires finite features")
    target = epanechnikov_kernel(
        training[config.dose].to_numpy()[:, None], centers[None, :], halfwidth
    )
    params = dict(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=8,
        max_features=0.75,
        n_jobs=config.model_threads,
        random_state=config.seed,
    )
    params.update(config.gps_params)
    settings = dict(
        centers=centers.tolist(),
        halfwidth=halfwidth,
        mode=mode,
        unconditional=unconditional,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        seed=config.seed,
        params=params,
        implementation=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    key = input_key(
        training[["ind_code", "ind_refined", config.dose] + columns], settings
    )
    path = Path(config.cache_dir) / "kernel_gps" / f"{key}.joblib"
    hit = path.exists()
    if hit:
        stored = joblib.load(path)
        predictions = stored["predictions"]
        folds = stored["folds"]
    else:
        labels = training.ind_refined.to_numpy()
        k = min(config.n_splits, int(pd.Series(labels).value_counts().min()))
        if k < 2:
            raise ValueError("Each stratum needs at least two industries")
        predictions = np.full((config.n_repeats, len(training), len(centers)), np.nan)
        folds = []
        with threadpool_limits(limits=1):
            for rep in range(config.n_repeats):
                cv = StratifiedKFold(k, shuffle=True, random_state=config.seed + rep)
                for fold, (train, test) in enumerate(cv.split(X, labels)):
                    if unconditional:
                        predictions[rep, test] = target[train].mean(axis=0)
                    elif mode == "shared":
                        model = RandomForestRegressor(**params).fit(
                            X[train], target[train]
                        )
                        predictions[rep, test] = np.asarray(
                            model.predict(X[test])
                        ).reshape(len(test), len(centers))
                    else:
                        for j in range(len(centers)):
                            model = RandomForestRegressor(**params).fit(
                                X[train], target[train, j]
                            )
                            predictions[rep, test, j] = model.predict(X[test])
                    folds.append(
                        dict(
                            repetition=rep,
                            fold=fold,
                            train_rows=len(train),
                            test_rows=len(test),
                            train_positions=train.tolist(),
                            test_positions=test.tolist(),
                        )
                    )
        if not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError("Invalid kernel regression predictions")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(predictions=predictions, folds=folds), path, compress=3)
    mapping = frame.ind_code.map(
        pd.Series(np.arange(len(training)), index=training.ind_code)
    ).to_numpy(int)
    return KernelGpsResult(
        centers,
        training.ind_code.to_numpy(),
        predictions,
        mapping,
        dict(
            cache_key=key,
            cache_hit=hit,
            mode=mode,
            n_industries=len(training),
            features=columns,
            halfwidth=halfwidth,
            folds=folds,
            params=params,
            density_floor=None,
            target="E[K_halfwidth(D-d)|X]",
            aggregation="median of OOF predictions",
        ),
    )


def atet_from_kernel_scores(
    frame,
    dtreat,
    dcontrol,
    halfwidth,
    pt,
    pc,
    mu_control,
    *,
    trim=0.1,
    min_ess=5,
    dose="office_ratio",
    group="ind_code",
):
    """R's two-arm normalization/trim with supplied kernel-GPS scores.

    Zero/negative density on active support is reported as failure, never floored.
    Cluster SE follows the centered causalweight score, with G/(G-1) correction.
    """
    D = frame[dose].to_numpy(float)
    high = epanechnikov_kernel(D, dtreat, halfwidth)
    low = epanechnikov_kernel(D, dcontrol, halfwidth)
    pt, pc = np.asarray(pt, float), np.asarray(pc, float)
    if pt.shape != D.shape or pc.shape != D.shape:
        raise ValueError("GPS score shape mismatch")
    invalid = (
        (~np.isfinite(pt))
        | (~np.isfinite(pc))
        | (pt < 0)
        | (pc < 0)
        | ((low > 0) & (pc <= 0))
        | ((high > 0) & (pt <= 0))
    )
    if invalid.any():
        raise ValueError(
            f"Nonpositive/invalid GPS on active support: {invalid.sum()} firms"
        )
    comparison = low * np.divide(pt, pc, out=np.zeros_like(pt), where=pc > 0)
    if high.sum() <= 0 or comparison.sum() <= 0:
        raise ValueError("Empty kernel arm")
    keep = (high / high.sum() <= trim) & (comparison / comparison.sum() <= trim)
    high = np.where(keep, high, 0)
    comparison = np.where(keep, comparison, 0)
    if high.sum() <= 0 or comparison.sum() <= 0:
        raise ValueError("Empty arm after trimming")
    r = frame.Y_diff.to_numpy(float) - np.asarray(mu_control, float)
    point, _, diagnostic = normalized_atet_score(r, high, comparison)
    a, b = high / high.sum(), comparison / comparison.sum()
    codes, groups = pd.factorize(frame[group], sort=True)
    ah = np.bincount(codes, weights=a)
    bh = np.bincount(codes, weights=b)
    contribution = a * (r - point) - b * r
    cluster = np.bincount(codes, weights=contribution, minlength=len(groups))
    used_groups = np.unique(codes[keep])
    G = len(used_groups)
    se = float(np.sqrt(np.sum(cluster**2) * G / (G - 1))) if G > 1 else np.nan
    ess_high, ess_low = effective_sample_size(ah), effective_sample_size(bh)
    return dict(
        atet=point,
        se_author_cluster=se,
        ess_treat=ess_high,
        ess_control=ess_low,
        supported=min(ess_high, ess_low) > min_ess,
        max_industry_weight=max(ah.max(), bh.max()),
        max_firm_weight=max(a.max(), b.max()),
        trimmed_firms=int((~keep).sum()),
        **diagnostic,
    )
