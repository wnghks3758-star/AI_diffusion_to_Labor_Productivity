"""The original balanced panel, industry encoding and relative-size rules."""

import hashlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from .io import file_hash

def merge_adjacent_divisions(
    df, major, prefix, ind_col="ind_code", year_col="Year", min_nunique=6, min_obs=6
):
    mask = df["대분류"] == major
    df.loc[mask, "ind_refined"] = prefix + "_" + df.loc[mask, "중분류"]
    all_years = np.sort(df[year_col].dropna().unique())
    cats = sorted(
        df.loc[mask, "ind_refined"].dropna().unique(),
        key=lambda x: int(x.split("_")[1]),
    )
    current = []
    final = []
    for cat in cats:
        current.append(cat)
        sub = df[df["ind_refined"].isin(current)]
        n_ind = (
            sub.groupby(year_col)[ind_col].nunique().reindex(all_years, fill_value=0)
        )
        n_obs = sub.groupby(year_col).size().reindex(all_years, fill_value=0)
        if n_ind.min() >= min_nunique and n_obs.min() >= min_obs:
            final.append(current.copy())
            current = []
    if current:
        if final:
            final[-1].extend(current)
        else:
            final.append(current)
    mapping = {}
    for group in final:
        divs = [x.split("_")[1] for x in group]
        name = (
            f"{prefix}_{divs[0]}"
            if len(group) == 1
            else f"{prefix}_{divs[0]}_{divs[-1]}"
        )
        for cat in group:
            mapping[cat] = name
    df["ind_refined"] = df["ind_refined"].replace(mapping)
    return df

def refine_industry_codes(
    df, ind_col="ind_code", year_col="Year", min_nunique=6, min_obs=6
):
    """산업구조로 ind_refined와 산업 더미를 생성한다."""
    df = df.copy()
    division_to_major = {}
    major_ranges = {
        "A": range(1, 4),
        "B": range(5, 9),
        "C": range(10, 35),
        "D": range(35, 36),
        "E": range(36, 40),
        "F": range(41, 43),
        "G": range(45, 48),
        "H": range(49, 53),
        "I": range(55, 57),
        "J": range(58, 64),
        "K": range(64, 67),
        "L": range(68, 69),
        "M": range(70, 74),
        "N": range(74, 77),
        "O": range(84, 85),
        "P": range(85, 86),
        "Q": range(86, 88),
        "R": range(90, 92),
        "S": range(94, 97),
        "T": range(97, 99),
        "U": range(99, 100),
    }
    for major, divisions in major_ranges.items():
        for d in divisions:
            division_to_major[str(d).zfill(2)] = major
    df["중분류"] = df[ind_col].astype(str).str.zfill(3).str[:2]
    df["대분류"] = df["중분류"].map(division_to_major)
    non_mfg_macro = {
        "A": "Primary_Infra",
        "B": "Primary_Infra",
        "D": "Primary_Infra",
        "E": "Primary_Infra",
        "F": "Primary_Infra",
        "G": "Trade",
        "H": "Transport_Hospitality",
        "I": "Transport_Hospitality",
        "K": "Finance",
        "L": "Finance",
        "M": "Professional_Business",
        "N": "Professional_Business",
        "P": "Social_Service",
        "Q": "Social_Service",
        "R": "Personal_Service",
        "S": "Personal_Service",
    }
    df["ind_refined"] = df["대분류"].map(non_mfg_macro)
    all_years = np.sort(df[year_col].dropna().unique())
    df = merge_adjacent_divisions(
        df, "C", "MFG", ind_col, year_col, min_nunique, min_obs
    )
    df = merge_adjacent_divisions(
        df, "J", "ICT", ind_col, year_col, min_nunique, min_obs
    )
    categories = df["ind_refined"].dropna().unique()
    rare_groups = []
    for cat in categories:
        subset = df[df["ind_refined"] == cat]
        yearly_nunique = (
            subset.groupby(year_col)[ind_col].nunique().reindex(all_years, fill_value=0)
        )
        if yearly_nunique.min() < min_nunique:
            rare_groups.append(cat)
    if rare_groups:
        df["ind_refined"] = df["ind_refined"].replace(rare_groups, "Other")
    industry_dummies = pd.get_dummies(
        df["ind_refined"], prefix="ind_dummy", drop_first=True, dtype=int
    )
    df = pd.concat([df, industry_dummies], axis=1)
    return df

def prepare_year_data(df, year, base_year=2022):
    """Join baseline by stable firm identifier; preserve original pre-period sign convention."""
    temp = df[df["Year"].isin([year, base_year])].copy()
    if temp.duplicated(["firm_code", "Year"]).any():
        raise ValueError("Duplicate firm_code-Year observations are not allowed.")
    base_values = temp[temp["Year"] == base_year].set_index("firm_code")["ln_firm_prod"]
    temp["Y_base"] = temp["firm_code"].map(base_values)
    temp["Y_diff"] = (
        temp["ln_firm_prod"] - temp["Y_base"]
        if year > base_year
        else temp["Y_base"] - temp["ln_firm_prod"]
    )
    return temp[temp["Year"] == year].dropna(subset=["Y_diff"]).copy()

def kernel_ess(D, d, bandwidth):
    """주어진 d에서 산업단위 kernel ESS 계산."""
    u = (D - d) / bandwidth
    weights = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2) / bandwidth, 0)
    if np.sum(weights**2) == 0:
        return 0
    return weights.sum() ** 2 / np.sum(weights**2)

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

def prepare(raw, c):
    # Reuse only the established balanced sample/size classification/industry encoding.
    # No legacy GPS, ESS scan, bandwidth or density-floor calculation enters estimation.
    legacy = SimpleNamespace(
        data_path=c.data_path,
        base_year=c.base_year,
        years=c.years,
        dose=c.dose,
        outcome=c.outcome,
        dose_transform=c.dose_transform,
        covariate_time=c.covariate_time,
        pre_sign=c.pre_sign,
        gps_features=c.gps_features,
        mu_features=c.mu_features,
        include_industry_dummies=c.include_industry_dummies,
        industry_scheme=c.industry_scheme,
        industry_min_groups=c.industry_min_groups,
        industry_min_obs=c.industry_min_obs,
        min_ess_prime=0,
        mu_model="xgboost",
        # Fixed defaults used by the former sample adapter, not estimation tuning.
        sample_reference_features=[], size_quantile=0.5,
        small_industry_max_firms=1, min_bandwidth=0.01, max_dose=np.inf,
    )
    panel, sample = prepare_panel(raw, legacy)
    panel, size_audit = classify_size(panel, c)
    sample["size_classification"] = size_audit
    if c.size_missing_policy == "unclassified":
        sample["size_missing_handling"] = "Missing size observations remain in total and are unclassified for size contrasts"
    sample["covariate_time"] = c.covariate_time
    sample["mu_includes_firm_controls"] = c.include_firm_controls
    sample["missing_policy"] = c.missing_policy
    return panel, sample, legacy
