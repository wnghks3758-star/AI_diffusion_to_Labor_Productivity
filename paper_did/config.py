"""One explicit configuration for the paper-aligned trial and model switches."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from labor_did.config import FIRM_CONTROLS, INDUSTRY_CONTROLS

MODELS = ("spline", "random_forest", "lasso", "ridge", "linear", "xgboost")


@dataclass
class PaperConfig:
    data_path: str = (
        "data/main_data/firm_year_all_ind_dart_add_control_gpt_continuous.xlsx"
    )
    base_year: int = 2022
    years: list = field(default_factory=lambda: [2020, 2021, 2023, 2024, 2025])
    dose: str = "office_ratio"
    outcome: str = "ln_firm_prod"
    dose_transform: str = "log_percent"
    covariate_time: str = "baseline"
    pre_sign: str = "legacy"
    gps_features: list = field(
        default_factory=lambda: [
            c for c in INDUSTRY_CONTROLS if c != "ind_avg_hours_pre"
        ]
    )
    mu_industry_features: list = field(default_factory=lambda: INDUSTRY_CONTROLS.copy())
    mu_firm_features: list = field(
        default_factory=lambda: ["large_dummy", "small_dummy"] + FIRM_CONTROLS
    )
    include_firm_controls: bool = True
    include_industry_dummies: bool = True
    industry_scheme: str = "refined"
    industry_min_groups: int = 4
    industry_min_obs: int = 4
    gps_model: str = "spline"
    mu_model: str = "xgboost"
    gps_params: dict = field(default_factory=dict)
    mu_params: dict = field(default_factory=dict)
    spline_n_knots: int = 5
    spline_degree: int = 3
    ridge_alphas: list = field(default_factory=lambda: np.logspace(-4, 1, 8).tolist())
    lasso_alphas: list = field(default_factory=lambda: np.logspace(-4, 1, 30).tolist())
    n_splits: int = 3
    gps_n_splits: int | None = None
    mu_n_splits: int | None = None
    gps_fold_seed: int | None = None
    mu_fold_seed: int | None = None
    inner_splits: int = 10
    seed: int = 1
    model_threads: int = 1
    bandwidth_factor: float = 0.7
    trim: float = 0.1
    missing_policy: str = "median_indicator"
    pair_scale: str = "quantile"
    atet_pairs: list = field(default_factory=lambda: [[0.25, 0.75]])
    compute_acrt: bool = True
    acrt_support: str = "computable"
    diff_step: float = 0.0005
    gap_policy: str = "interpolate"
    size_difference: str = "means"
    size_variable: str = "sale"
    size_years: list | None = None
    size_time_aggregation: str = "mean"
    size_threshold: str = "median"
    size_missing_policy: str = "legacy"
    size_small_industry_max_firms: int = 1
    alpha: float = 0.05
    n_bootstrap: int = 5000
    bootstrap_batch_size: int = 256
    prediction_batch_points: int = 8
    max_prediction_rows: int = 50000
    cache_dir: str = ".cache/paper"
    label: str = "paper_default"

    def __post_init__(self):
        for name in ("gps_model", "mu_model"):
            if getattr(self, name) not in MODELS:
                raise ValueError(f"{name} must be one of {MODELS}")
        for name, choices in dict(
            acrt_support=("strict", "computable"),
            industry_scheme=("refined", "major", "none"),
            covariate_time=("baseline", "current"),
            pre_sign=("legacy", "chronological"),
            dose_transform=("log_percent", "none"),
            pair_scale=("quantile", "value"),
            size_difference=("means", "common"),
            gap_policy=("interpolate", "exclude", "bridge", "strict"),
            missing_policy=("median_indicator", "error"),
            size_time_aggregation=("mean", "median"),
            size_threshold=("median", "mean"),
            size_missing_policy=("legacy", "unclassified"),
        ).items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        for name in ("gps_params", "mu_params"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be a dictionary")
        for name in (
            "include_firm_controls",
            "include_industry_dummies",
            "compute_acrt",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("n_splits", "inner_splits", "spline_n_knots", "n_bootstrap"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 2:
                raise ValueError(f"{name} must be >=2")
        for name in ("gps_n_splits", "mu_n_splits"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 2):
                raise ValueError(f"{name} must be None or an integer >=2")
        for name in ("gps_fold_seed", "mu_fold_seed"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be None or a nonnegative integer")
        for name in (
            "industry_min_groups",
            "industry_min_obs",
            "model_threads",
            "prediction_batch_points",
            "max_prediction_rows",
            "bootstrap_batch_size",
            "spline_degree",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{name} must be >=1")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not isinstance(self.size_variable, str) or not self.size_variable or self.size_variable in {self.dose, self.outcome, "Y_diff"}:
            raise ValueError("Specify a size variable distinct from dose/outcome")
        if not isinstance(self.size_small_industry_max_firms, int) or isinstance(self.size_small_industry_max_firms, bool) or self.size_small_industry_max_firms < 0:
            raise ValueError("size_small_industry_max_firms must be a nonnegative integer")
        if self.size_years is not None:
            if not self.size_years or len(set(self.size_years)) != len(self.size_years):
                raise ValueError("Specify distinct size years")
            if any(not isinstance(y, int) or isinstance(y, bool) or y > self.base_year for y in self.size_years):
                raise ValueError("Size years must be integers at or before base_year")
        if (
            not self.years
            or len(set(self.years)) != len(self.years)
            or self.base_year in self.years
        ):
            raise ValueError("Distinct analysis years must exclude base_year")
        for name in ("bandwidth_factor", "diff_step"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.trim <= 1 or not 0 < self.alpha < 1:
            raise ValueError("Invalid trim/alpha")
        for values in (self.ridge_alphas, self.lasso_alphas):
            if not values or any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Positive regularization values required")
        if not self.atet_pairs:
            raise ValueError("Specify at least one ATET pair")
        for pair in self.atet_pairs:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(np.isfinite(v) for v in pair)
                or not pair[0] < pair[1]
            ):
                raise ValueError("Each ATET pair is [control,treat], control < treat")
            if self.pair_scale == "quantile" and not 0 <= pair[0] < pair[1] <= 1:
                raise ValueError("Quantiles must lie in [0,1]")
        for features in (
            self.gps_features,
            self.mu_industry_features,
            self.mu_firm_features,
        ):
            if len(features) != len(set(features)) or set(features) & {
                self.dose,
                self.outcome,
                "Y_diff",
            }:
                raise ValueError("Duplicate or prohibited covariate")

    @property
    def mu_features(self):
        return list(
            dict.fromkeys(
                self.mu_industry_features
                + (self.mu_firm_features if self.include_firm_controls else [])
            )
        )

    def fold_settings(self, role):
        if role not in ("gps", "mu"):
            raise ValueError("role must be gps or mu")
        splits, seed = getattr(self, role + "_n_splits"), getattr(self, role + "_fold_seed")
        return self.n_splits if splits is None else splits, self.seed if seed is None else seed

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @property
    def fingerprint(self):
        values = self.to_dict()
        values.pop("label")
        values.pop("cache_dir")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[
            :16
        ]
