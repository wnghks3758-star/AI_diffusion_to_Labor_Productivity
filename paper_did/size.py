"""Pre-treatment, industry-relative firm size; preserve the original default."""

import numpy as np
import pandas as pd


def classify_size(panel, c):
    if c.size_variable not in panel:
        raise ValueError(f"Missing size variable: {c.size_variable}")
    years = sorted(c.size_years if c.size_years is not None else panel.loc[panel.Year.le(c.base_year), "Year"].unique().tolist())
    if not years or set(years) - set(panel.Year.unique()):
        raise ValueError("Requested size years are absent from the panel")
    history = panel.loc[panel.Year.isin(years), ["firm_code", "ind_code", c.size_variable]].copy()
    history["_value"] = pd.to_numeric(history[c.size_variable], errors="raise")
    if np.isinf(history._value).any():
        raise ValueError("Infinite size observations")
    firms = history.groupby(["ind_code", "firm_code"], as_index=False).agg(
        size_value=("_value", c.size_time_aggregation), valid_years=("_value", "count"))
    firms["industry_firms"] = firms.groupby("ind_code").firm_code.transform("nunique")
    firms["threshold"] = firms.groupby("ind_code").size_value.transform(c.size_threshold)
    firms["large_firm"] = (firms.size_value >= firms.threshold).astype(int)
    small = firms.industry_firms.le(c.size_small_industry_max_firms)
    missing = firms.size_value.isna() | firms.threshold.isna()
    firms.loc[small | (missing & (c.size_missing_policy == "unclassified")), "large_firm"] = -1
    firms["large_dummy"] = firms.large_firm.eq(1).astype(int)
    firms["small_dummy"] = firms.large_firm.eq(0).astype(int)
    mapping = firms.set_index("firm_code")[["large_firm", "large_dummy", "small_dummy"]]
    result = panel.drop(columns=[k for k in mapping.columns if k in panel]).join(mapping, on="firm_code")
    result = result.loc[:, list(panel.columns) + [k for k in mapping.columns if k not in panel]]
    if result.large_firm.isna().any():
        raise ValueError("Size mapping did not cover every analysis firm")
    audit = dict(variable=c.size_variable, years=years, time_aggregation=c.size_time_aggregation,
                 threshold=c.size_threshold, missing_policy=c.size_missing_policy,
                 small_industry_max_firms=c.size_small_industry_max_firms,
                 analysis_firms=len(firms), sme_firms=int(firms.large_firm.eq(0).sum()),
                 large_firms=int(firms.large_firm.eq(1).sum()), unclassified_firms=int(firms.large_firm.eq(-1).sum()),
                 missing_size_firms=int(missing.sum()), small_industry_firms=int(small.sum()),
                 valid_year_counts={str(k): int(v) for k, v in firms.valid_years.value_counts().sort_index().items()})
    return result, audit
