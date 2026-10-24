"""Balanced panel, original size/missing handling, support and industry encoding."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .industry import kernel_ess, prepare_year_data, refine_industry_codes
from .io import file_hash


def load_data(path, cache_dir="work/cache/dml_acr"):
    path = Path(path)
    digest = file_hash(path)
    cache = Path(cache_dir) / "data" / f"{digest}.pkl"
    if cache.exists():
        return pd.read_pickle(cache), digest
    data = pd.read_excel(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(cache)
    return data, digest


def create_large_firm_dummy(
    df, base_year=2022, size_quantile=0.5, small_industry_max_firms=1
):
    """기준연도 이전 매출로 규모를 구분한다. 산업 내 기업이 2개 이하면 미분류(-1)로 둔다."""
    df = df.copy()
    firm_sale = (
        df[df["Year"] <= base_year]
        .groupby(["ind_code", "firm_code"])["sale"]
        .mean()
        .reset_index()
    )  # 회사별 평균값
    firm_sale["n_firms"] = firm_sale.groupby("ind_code")["firm_code"].transform(
        "nunique"
    )  # 산업별 회사 수
    firm_sale["ind_median_size"] = firm_sale.groupby("ind_code")["sale"].transform(
        lambda x: x.quantile(size_quantile)
    )  # 산업별 median
    firm_sale["large_firm"] = np.where(
        firm_sale["n_firms"] <= small_industry_max_firms,
        -1,
        (firm_sale["sale"] >= firm_sale["ind_median_size"]).astype(int),
    )
    firm_sale["large_dummy"] = (firm_sale["large_firm"] == 1).astype(int)
    firm_sale["small_dummy"] = (firm_sale["large_firm"] == 0).astype(int)
    firm_map = firm_sale.set_index("firm_code")[
        ["large_firm", "large_dummy", "small_dummy"]
    ]
    df = df.join(firm_map, on="firm_code")
    return df.dropna(subset=["large_firm"]).copy()


def cluster_bandwidth(frame, config):
    d = frame.groupby("ind_code")[config.dose].first().dropna().to_numpy(float)
    if len(d) < 2:
        raise ValueError("Need at least two industries")
    sd = np.std(d, ddof=0)
    iqr = np.subtract(*np.percentile(d, [75, 25]))
    sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
    scale = sd if config.bandwidth_statistic == "std" else sigma
    h = max(
        2.34 * scale * len(d) ** (-0.25) * config.bandwidth_scale, config.min_bandwidth
    )
    return h * (np.sqrt(5) if config.kernel_convention == "np" else 1.0)


def prepare_panel(raw, config):
    wanted = set(config.years) | {config.base_year}
    data = raw.copy()
    required = {"firm_code", "Year", "ind_code", "sale", config.dose, config.outcome}
    controls = set(
        config.gps_features + config.mu_features + config.sample_reference_features
    ) - {"large_dummy", "small_dummy"}
    missing = (required | controls) - set(data.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if data[["firm_code", "Year", "ind_code"]].isna().any().any():
        raise ValueError("Missing identifiers")
    if data.duplicated(["firm_code", "Year"]).any():
        raise ValueError("Duplicate firm-year")
    if (data.groupby("firm_code").ind_code.nunique() > 1).any():
        raise ValueError("Firm changes industry")
    counts = data.loc[data.Year.isin(wanted)].groupby("firm_code").Year.nunique()
    data = data.loc[data.firm_code.isin(counts[counts.eq(len(wanted))].index)].copy()
    if not len(data):
        raise ValueError("No balanced firms")
    if (data.groupby("ind_code")[config.dose].nunique(dropna=False) > 1).any():
        raise ValueError(
            "This implementation requires industry-constant exposure across years"
        )
    if not np.isfinite(data[config.dose]).all():
        raise ValueError("Non-finite dose")
    if config.dose_transform == "log_percent":
        if data[config.dose].le(0).any():
            raise ValueError("Log dose must be positive")
        data[config.dose] = np.log(data[config.dose] * 100)
    data["ln_firm_prod"] = data[config.outcome]
    if config.covariate_time == "baseline":
        base = data.loc[data.Year.eq(config.base_year)].set_index("firm_code")
        for c in controls:
            data[c] = data.firm_code.map(base[c])
    data = create_large_firm_dummy(
        data, config.base_year, config.size_quantile, config.small_industry_max_firms
    )
    # XGBoost accepts missing firm controls. Preserve these rows as in the notebook.
    active = data.loc[data.Year.isin(wanted)]
    check = [config.outcome] + sorted(controls)
    if np.isinf(active[check].to_numpy(float)).any():
        raise ValueError("Infinite selected controls or outcomes")
    if active[config.outcome].isna().any():
        raise ValueError("Missing outcomes; clean the panel explicitly")
    if config.mu_model != "xgboost" and active[sorted(controls)].isna().any().any():
        raise ValueError(
            "Ridge learners require complete controls; XGBoost accepts missing controls"
        )
    annual = []
    for year in config.years:
        frame = prepare_year_data(data, year, config.base_year)
        # Original common-lower filtering always used robust sigma and scale 0.5.
        d = frame.groupby("ind_code")[config.dose].first().dropna().to_numpy(float)
        sd = np.std(d)
        iqr = np.subtract(*np.percentile(d, [75, 25]))
        sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
        bw = max(2.34 * sigma * len(d) ** (-0.25) * 0.5, config.min_bandwidth)
        annual.append((d, bw))
    candidates = np.unique(np.concatenate([d for d, _ in annual]))
    lower = next(
        (
            float(d)
            for d in candidates
            if d <= config.max_dose
            and all(
                kernel_ess(values, d, bw) >= config.min_ess_prime
                for values, bw in annual
            )
        ),
        None,
    )
    if lower is None:
        raise ValueError("No common lower support; reduce min_ess_prime")
    data = data.loc[data[config.dose].between(lower, config.max_dose)].copy()
    data = refine_industry_codes(
        data, min_nunique=config.industry_min_groups, min_obs=config.industry_min_obs
    )
    if config.industry_scheme != "refined":
        data = data.drop(columns=[c for c in data if c.startswith("ind_dummy")])
        data["ind_refined"] = (
            data["대분류"].fillna("Other")
            if config.industry_scheme == "major"
            else "All"
        )
        data = pd.concat(
            [
                data,
                pd.get_dummies(
                    data.ind_refined, prefix="ind_dummy", drop_first=True, dtype=int
                ),
            ],
            axis=1,
        )
    if data.ind_refined.isna().any():
        raise ValueError("Unmapped industry; use industry_scheme=major or none")
    if not config.include_industry_dummies or config.industry_scheme == "none":
        data = data.drop(columns=[c for c in data if c.startswith("ind_dummy")])
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(
            data[["firm_code", "Year"]].reset_index(drop=True), index=False
        ).values.tobytes()
    ).hexdigest()
    return data, {
        "analysis_firms": int(data.firm_code.nunique()),
        "analysis_industries": int(data.ind_code.nunique()),
        "sample_firm_sha256": digest,
        "common_lower_dose": lower,
        "common_lower_ess": config.min_ess_prime,
        "dummy_columns": [c for c in data if c.startswith("ind_dummy")],
        "size_missing_handling": "original notebook convention; missing sale is compared as non-large",
    }


def prepare_year(data, year, config):
    frame = prepare_year_data(data, year, config.base_year).reset_index(drop=True)
    if year < config.base_year and config.pre_sign == "chronological":
        frame["Y_diff"] *= -1
    return frame


def observed_grid(frame, config):
    values = np.sort(frame.groupby("ind_code")[config.dose].first().unique())
    low, high = values[0], values[-1]
    steps = int(np.floor((high - low) / config.diff_step))
    if steps > 20_000_000:
        raise ValueError("Grid exceeds 20 million points; increase diff_step")
    grid = low + config.diff_step * np.arange(steps + 1)
    if high - grid[-1] > 1e-12:
        grid = np.r_[grid, high]
    return grid, cluster_bandwidth(frame, config)
