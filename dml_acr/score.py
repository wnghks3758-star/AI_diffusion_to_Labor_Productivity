"""Numeric primitives adapted from the uploaded f4_7 notebook."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def normalized_acr_score(residual, omega_high, omega_low, diff_step):
    """ACR of two self-normalized residual means and their ratio-linearized influence.

    a_i = w_high_i / sum(w_high); b_i = w_low_i / sum(w_low).
    theta = (sum(a*r) - sum(b*r)) / h.
    IF_i = N * (a_i*(r_i-r_high) - b_i*(r_i-r_low)) / h.
    Nuisance predictions, selected support and final averaging weights are held fixed.
    """
    h = float(diff_step)
    r, high, low = [np.asarray(x,dtype=float) for x in (residual,omega_high,omega_low)]
    if not np.isfinite(h) or h <= 0:
        raise ValueError('diff_step must be finite and strictly positive.')
    if r.ndim != 1 or high.shape != r.shape or low.shape != r.shape or not len(r):
        raise ValueError('Residual and weight vectors must have equal nonzero length.')
    if not np.isfinite(r).all() or any((~np.isfinite(w)).any() or (w<0).any() or w.max()<=0 for w in (high,low)):
        raise ValueError('Finite residuals and nonnegative weights with positive arm mass are required.')
    # First divide by max for scale invariance and to avoid overflow; no arbitrary epsilon.
    a=high/high.max();a/=a.sum()
    b=low/low.max();b/=b.sum()
    mean_high,mean_low=float(a@r),float(b@r)
    point=(mean_high-mean_low)/h
    influence=len(r)*(a*(r-mean_high)-b*(r-mean_low))/h
    return point,influence,{'weight_high_sum':float(a.sum()),'weight_low_sum':float(b.sum()),
                            'max_high_firm_weight':float(a.max()),'max_low_firm_weight':float(b.max()),
                            'mean_high_residual':mean_high,'mean_low_residual':mean_low,
                            'influence_mean':float(influence.mean())}
