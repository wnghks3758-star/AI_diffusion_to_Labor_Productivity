"""causalweight panel score algebra, without an ESS cutoff or density floor.

Reference: causalweight 1.1.6, didcontDMLpanel.R (commit 84b6412).
Industry bandwidth units and cluster-aware folds are application adaptations.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from labor_did.weights import effective_sample_size, epanechnikov_kernel


def paper_bandwidth(industry_dose, factor=0.7):
    """R sd() uses ddof=1. No robust-sigma substitution or minimum-h floor."""
    d = np.asarray(industry_dose, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Need at least two finite industry doses")
    h = float(2.34 * np.std(d, ddof=1) * len(d) ** (-0.25) * factor)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Bandwidth must be positive")
    return h


def np_kernel(dose, center, bandwidth):
    """R np's variance-one Epanechnikov: physical support +/- sqrt(5)*h."""
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")
    return epanechnikov_kernel(np.asarray(dose, float), center, np.sqrt(5) * bandwidth)


def gaussian_log_ratio(dtreat, dcontrol, mean, variance):
    """R's dnorm density ratio; evaluate in log space without clipping densities.

    The common within-fold variance cancels the normalizing density constant.
    This does not assume homoskedasticity across different outer folds.
    """
    mean, variance = np.asarray(mean, float), np.asarray(variance, float)
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or (variance <= 0).any()
    ):
        raise ValueError("Finite means and positive fold variances are required")
    # Difference of squares avoids separately squaring two similar numbers.
    value = -(dtreat - dcontrol) * (dtreat + dcontrol - 2 * mean) / (2 * variance)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite Gaussian log density ratio")
    return value


def reference_fold_variance(observed, predicted):
    """R default: mean squared residual of the OUTER VALIDATION industries.

    Intentionally reproduces the author's variance construction. It is not the
    training-only heteroskedastic/smearing learner of the historical Python path.
    """
    residual = np.asarray(observed, float) - np.asarray(predicted, float)
    variance = float(np.mean(residual**2))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Degenerate outer-fold variance")
    return variance


def atet_score(
    y_diff, mu_control, kernel_treat, kernel_control, log_ratio, groups, trim=0.1
):
    """Two normalized residual means and R-centered cluster score.

    All observations share the same two-arm trimming mask. ESS is diagnostic
    only. Log-space normalization changes no estimand and adds no density floor.
    """
    y, mu, high, low, ratio = [
        np.asarray(v, float)
        for v in (y_diff, mu_control, kernel_treat, kernel_control, log_ratio)
    ]
    if (
        y.ndim != 1
        or not len(y)
        or any(v.shape != y.shape for v in (mu, high, low, ratio))
    ):
        raise ValueError("Score inputs must have equal nonzero vector lengths")
    if not 0 < trim <= 1 or any(
        not np.isfinite(v).all() for v in (y, mu, high, low, ratio)
    ):
        raise ValueError("Finite inputs and trim in (0,1] required")
    if (high < 0).any() or (low < 0).any() or high.sum() <= 0 or low.sum() <= 0:
        raise ValueError("Each nonnegative kernel arm must have positive mass")
    a = high / high.sum()
    active = low > 0
    log_b = np.log(low[active]) + ratio[active]
    b = np.zeros_like(low)
    b[active] = np.exp(log_b - logsumexp(log_b))
    keep = (a <= trim) & (b <= trim)
    a = np.where(keep, a, 0.0)
    b = np.where(keep, b, 0.0)
    if a.sum() <= 0 or b.sum() <= 0:
        raise ValueError("An arm is empty after two-arm trimming")
    a /= a.sum()
    b /= b.sum()
    r = y - mu
    theta = float((a - b) @ r)
    codes, labels = pd.factorize(np.asarray(groups), sort=True)
    if len(codes) != len(y) or (codes < 0).any():
        raise ValueError("Each firm needs a nonmissing industry")
    contribution = a * (r - theta) - b * r
    clusters = np.bincount(codes, weights=contribution, minlength=len(labels))
    ah = np.bincount(codes, weights=a, minlength=len(labels))
    bh = np.bincount(codes, weights=b, minlength=len(labels))
    G = len(np.unique(codes[keep]))
    se = float(np.sqrt(np.sum(clusters**2) * G / (G - 1))) if G > 1 else np.nan
    pvalue = float(2 * norm.sf(abs(theta / se))) if se > 0 else np.nan
    return dict(
        atet=theta,
        se=se,
        pvalue=pvalue,
        n_firms=len(y),
        n_retained=int(keep.sum()),
        n_industries=len(labels),
        n_clusters_retained=G,
        n_trimmed=int((~keep).sum()),
        ess_treat=effective_sample_size(ah),
        ess_control=effective_sample_size(bh),
        max_industry_weight=float(max(ah.max(), bh.max())),
        max_firm_weight=float(max(a.max(), b.max())),
        weight_treat=a,
        weight_control=b,
        keep=keep,
        cluster_contributions=clusters,
    )
