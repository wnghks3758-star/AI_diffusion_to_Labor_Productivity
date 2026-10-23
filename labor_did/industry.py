"""Numeric primitives adapted from the uploaded f4_7 notebook."""

import numpy as np
import pandas as pd


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
