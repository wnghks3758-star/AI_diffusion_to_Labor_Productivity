"""Numeric primitives adapted from the uploaded f4_7 notebook."""
import numpy as np
import pandas as pd
from scipy.stats import norm

def get_fast_options(options=None):
    return {"support_batch_points": 256, **(options or {})}


def epanechnikov_kernel(D, d, bandwidth):
    u = (D - d) / bandwidth
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2) / bandwidth, 0)

def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    squared_sum = np.sum(weights ** 2)
    return float(weights.sum() ** 2 / squared_sum) if squared_sum > 0 and np.isfinite(squared_sum) else 0.0

def get_trimmed_weights(df, d_t, d_prime_t, D_pred, sigma_pred, bandwidth, var_name='office_ratio',
                        group_col='ind_code', trim_threshold=0.1, p_floor=5e-3):
    """기존 기업 단위 trimming 후 두 항의 산업 단위 ESS를 계산한다."""
    D = df[var_name].to_numpy()
    omega_d = epanechnikov_kernel(D, d_t, bandwidth)
    omega_dp = epanechnikov_kernel(D, d_prime_t, bandwidth)
    p_d = np.clip(norm.pdf(d_t - D_pred, scale=sigma_pred), p_floor, None)
    p_dp = np.clip(norm.pdf(d_prime_t - D_pred, scale=sigma_pred), p_floor, None)
    ratio = p_d / p_dp
    raw_weight = omega_dp * ratio
    total_weight = raw_weight.sum()
    valid_mask = raw_weight / total_weight <= trim_threshold if total_weight > 0 else np.zeros(len(df), dtype=bool)
    ratio = np.where(valid_mask, ratio, 0.0)
    omega_d = np.where(valid_mask, omega_d, 0.0)
    omega_dp = np.where(valid_mask, omega_dp, 0.0)
    if ratio.sum() > 0:
        ratio /= ratio.sum()
    comparison_weight = omega_dp * ratio
    industry_index, industries = pd.factorize(df[group_col], sort=False)
    treated_by_industry = np.bincount(industry_index, weights=omega_d, minlength=len(industries))
    comparison_by_industry = np.bincount(industry_index, weights=comparison_weight, minlength=len(industries))
    support = {'d': d_t, 'ess_d': effective_sample_size(treated_by_industry),
               'ess_dp': effective_sample_size(comparison_by_industry),
               'n_kept': int(np.count_nonzero((omega_d > 0) | (comparison_weight > 0))),
               'n_trimmed': int(np.count_nonzero((raw_weight > 0) & ~valid_mask))}
    return omega_d, comparison_weight, support

def select_supported_centers(df, d_values, bandwidth, diff_step, var_name='office_ratio',
                             group_col='ind_code', min_ess=10, trim_threshold=0.1, p_floor=5e-3,
                             return_used_masks=False, return_weights=False, options=None, required_d=None):
    """Batch kernel/GPS calculations; compress constant industry inputs with firm counts.

    Trimming still uses each firm's weight divided by the total firm weight.
    Heterogeneous industries fall back to uncompressed firm rows.
    """
    config = get_fast_options(options)
    d_values = np.asarray(d_values, dtype=float)
    industry_codes, industries = pd.factorize(df[group_col], sort=False)
    if np.any(industry_codes < 0):
        raise ValueError(f'Missing {group_col}')
    values = df[[var_name, 'D_pred', 'sigma_pred']].to_numpy(dtype=float)
    if len(df):
        _, first = np.unique(industry_codes, return_index=True)
        constant = np.array_equal(values, values[first][industry_codes], equal_nan=True)
    else:
        first, constant = np.array([], dtype=int), True
    if constant:
        values = values[first]
        counts = np.bincount(industry_codes, minlength=len(industries)).astype(float)
        row_to_entry, entry_industry = industry_codes, np.arange(len(industries))
    else:
        counts = np.ones(len(df))
        row_to_entry, entry_industry = np.arange(len(df)), industry_codes
    order = np.argsort(entry_industry, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(entry_industry[order])) + 1] if len(order) else np.array([], dtype=int)
    D, D_pred, sigma_pred = values.T
    selected, rows, used_masks, weights_cache = [], [], {}, {}
    required_d=set() if required_d is None else set(map(float,required_d))
    batch = config['support_batch_points']

    def industry_ess(weights):
        weighted = weights * counts[None, :]
        totals = np.add.reduceat(weighted[:, order], starts, axis=1) if len(starts) else weighted
        denominator = np.sum(totals ** 2, axis=1)
        return np.divide(totals.sum(axis=1) ** 2, denominator, out=np.zeros(len(weights)),
                         where=(denominator > 0) & np.isfinite(denominator))

    for start in range(0, len(d_values), batch):
        centers = d_values[start:start + batch]
        low = centers - diff_step
        omega_d = epanechnikov_kernel(D[None, :], centers[:, None], bandwidth)
        omega_dp = epanechnikov_kernel(D[None, :], low[:, None], bandwidth)
        p_d = np.clip(norm.pdf(centers[:, None] - D_pred, scale=sigma_pred), p_floor, None)
        p_dp = np.clip(norm.pdf(low[:, None] - D_pred, scale=sigma_pred), p_floor, None)
        ratio = p_d / p_dp
        raw = omega_dp * ratio
        total = np.sum(raw * counts, axis=1, keepdims=True)
        normalized = np.divide(raw, total, out=np.full_like(raw, np.inf), where=total > 0)
        valid = (normalized <= trim_threshold) & (total > 0)
        ratio = np.where(valid, ratio, 0.0)
        omega_d = np.where(valid, omega_d, 0.0)
        omega_dp = np.where(valid, omega_dp, 0.0)
        ratio_sum = np.sum(ratio * counts, axis=1, keepdims=True)
        np.divide(ratio, ratio_sum, out=ratio, where=ratio_sum > 0)
        comparison = omega_dp * ratio
        ess_d, ess_dp = industry_ess(omega_d), industry_ess(comparison)
        n_kept = np.sum(((omega_d > 0) | (comparison > 0)) * counts, axis=1).astype(int)
        n_trimmed = np.sum(((raw > 0) & ~valid) * counts, axis=1).astype(int)
        near_ess = np.isclose(ess_d, min_ess, rtol=1e-12, atol=1e-12) | np.isclose(ess_dp, min_ess, rtol=1e-12, atol=1e-12)
        near_trim = np.isclose(normalized, trim_threshold, rtol=1e-12, atol=1e-12).any(axis=1)
        for j, d in enumerate(centers):
            d_t, d_low = float(d), float(low[j])
            support = {'d': d_t, 'ess_d': float(ess_d[j]), 'ess_dp': float(ess_dp[j]),
                       'n_kept': int(n_kept[j]), 'n_trimmed': int(n_trimmed[j]),
                       'd_high': d_t, 'd_low': d_low, 'bandwidth': bandwidth, 'diff_step': diff_step,
                       'supported': bool(min(ess_d[j], ess_dp[j]) > min_ess)}
            # Recheck strict decision boundaries in the original summation order.
            original_weights = None
            if near_ess[j] or near_trim[j]:
                original_weights = get_trimmed_weights(df, d_t, d_low,
                    df['D_pred'].to_numpy(), df['sigma_pred'].to_numpy(), bandwidth,
                    var_name, group_col, trim_threshold, p_floor)
                support.update(original_weights[2])
                support['supported'] = min(support['ess_d'], support['ess_dp']) > min_ess
            rows.append(support)
            if support['supported']:
                selected.append(d_t)
            if support['supported'] or d_t in required_d:
                if return_weights or return_used_masks:
                    a, b = (original_weights[:2] if original_weights is not None else
                            (omega_d[j, row_to_entry], comparison[j, row_to_entry]))
                    if return_weights:
                        weights_cache[d_t] = (a, b, support.copy())
                    if return_used_masks:
                        used_masks[d_low] = (a > 0) | (b > 0)
    result = np.asarray(selected, dtype=float), pd.DataFrame(rows)
    if return_used_masks:
        result = (*result, used_masks)
    return (*result, weights_cache) if return_weights else result
