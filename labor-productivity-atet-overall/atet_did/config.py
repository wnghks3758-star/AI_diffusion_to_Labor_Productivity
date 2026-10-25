"""Study settings and separately resolved model-specific JSON parameters."""

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

INDUSTRY_CONTROLS = [
    "ln_ind_avg_pay",
    "ln_ind_ast",
    "ind_rnd_ratio",
    "ln_cum_patent",
    "ind_college_share_pre",
    "ind_large_est_share_pre",
    "ind_avg_age_pre",
    "ind_age50_share_pre",
    "ind_female_share_pre",
    "ind_permanent_share_pre",
    "ind_avg_tenure_pre",
    "ind_avg_hours_pre",
    "ind_ln_avg_pay_pre",
]
FIRM_CONTROLS = ["firm_rnd_ratio", "ln_firm_avg_pay", "ln_firm_ast"]

MODELS = ("spline", "random_forest", "lasso", "ridge", "linear", "xgboost")


@dataclass
class ATETConfig:
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
    psmethod: int = 1
    model_params_file: str = "model_parameters.json"
    gps_params: dict | None = None
    mu_params: dict | None = None
    model_options: dict = field(default_factory=dict)
    model_params_sha256: str | None = None
    spline_n_knots: int = 5
    spline_degree: int = 3
    ridge_alphas: list = field(default_factory=lambda: np.logspace(-4, 1, 8).tolist())
    lasso_alphas: list = field(default_factory=lambda: np.logspace(-4, 1, 30).tolist())
    n_splits: int = 3
    n_repeats: int = 1
    prediction_aggregation: str = "single"
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
    size_variable: str = "sale"
    size_years: list | None = None
    size_time_aggregation: str = "mean"
    size_threshold: str = "median"
    size_missing_policy: str = "legacy"
    size_small_industry_max_firms: int = 1
    alpha: float = 0.05
    prediction_batch_points: int = 8
    max_prediction_rows: int = 50000
    cache_dir: str = ".cache/paper"
    label: str = "ps1_ATET_overall"
    reference_doses: list = field(default_factory=lambda: [0.153, 0.175])
    sample_exclusion: str = "none"
    support_policy: str = "strict"
    training_min_dose: float | None = None

    def __post_init__(self):
        if isinstance(self.psmethod, bool) or not isinstance(self.psmethod, int) or self.psmethod not in (1, 2):
            raise ValueError("psmethod must be 1 (Gaussian GPS) or 2 (kernel GPS)")
        if not self.reference_doses or len(set(self.reference_doses)) != len(self.reference_doses):
            raise ValueError("Specify distinct reference_doses in original dose units")
        if any(isinstance(d, bool) or not isinstance(d, (int, float)) or not np.isfinite(d) for d in self.reference_doses):
            raise ValueError("reference_doses must contain finite numbers")
        if self.dose_transform == "log_percent" and any(d <= 0 for d in self.reference_doses):
            raise ValueError("log_percent reference_doses must be positive")
        if self.training_min_dose is not None and not np.isfinite(self.training_min_dose):
            raise ValueError("training_min_dose must be finite or None")
        if not all(isinstance(y, int) for y in [self.base_year] + self.years):
            raise ValueError("years must be integers")
        for name in ("gps_model", "mu_model"):
            if getattr(self, name) not in MODELS:
                raise ValueError(f"{name} must be one of {MODELS}")
        for name, choices in dict(
            sample_exclusion=("none", "training", "analysis"),
            support_policy=("strict", "supported"),
            industry_scheme=("refined", "major", "none"),
            covariate_time=("baseline", "current"),
            pre_sign=("legacy", "chronological"),
            dose_transform=("log_percent", "none"),
            missing_policy=("median_indicator", "error"),
            prediction_aggregation=("single", "median"),
            size_time_aggregation=("mean", "median"),
            size_threshold=("median", "mean"),
            size_missing_policy=("legacy", "unclassified"),
        ).items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        for name in ("gps_params", "mu_params"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be a dictionary or None")
        if not isinstance(self.model_options, dict) or set(self.model_options) - {"gps", "mu"}:
            raise ValueError("model_options must contain only gps/mu dictionaries")
        allowed = {"spline_n_knots", "spline_degree", "ridge_alphas", "lasso_alphas", "inner_splits"}
        for role, options in self.model_options.items():
            if not isinstance(options, dict) or set(options) - allowed:
                raise ValueError(f"Unsupported preprocessing/tuning options for {role}")
            # Validate role-specific options with the same rules as legacy settings.
            replace(self, model_options={}, **options)
        for name in (
            "include_firm_controls",
            "include_industry_dummies",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("n_splits", "inner_splits", "spline_n_knots"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 2:
                raise ValueError(f"{name} must be >=2")
        if not isinstance(self.n_repeats, int) or isinstance(self.n_repeats, bool) or self.n_repeats < 1:
            raise ValueError("n_repeats must be an integer >=1")
        if self.prediction_aggregation == "single" and self.n_repeats != 1:
            raise ValueError("single prediction_aggregation requires n_repeats=1; choose median for repeated predictions")
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
        for name in ("bandwidth_factor",):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.trim <= 1 or not 0 < self.alpha < 1:
            raise ValueError("Invalid trim/alpha")
        for values in (self.ridge_alphas, self.lasso_alphas):
            if not values or any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError("Positive regularization values required")
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

    def resolve_models(self, root=None):
        """Freeze selected profiles once; inline legacy parameters remain supported.

        Resolved result config.json files are self-contained and can be reloaded
        even if the original parameter bank has subsequently changed or moved.
        """
        if self.gps_params is not None and self.mu_params is not None:
            return self
        root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
        path = Path(self.model_params_file)
        if not path.is_absolute():
            path = root / path
        payload = path.read_bytes()
        try:
            bank = json.loads(payload.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid model parameter JSON: {path}") from exc
        parameters, options = {}, dict(self.model_options)
        for role, model in [("gps", self.gps_model), ("mu", self.mu_model)]:
            inline = getattr(self, role + "_params")
            if inline is not None:
                parameters[role] = inline
                continue
            try:
                profiles = bank["gps"][f"ps{self.psmethod}"] if role == "gps" else bank["mu"]
                profile = profiles[model]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Missing {role}/{model} profile for PS{self.psmethod} in {path}") from exc
            if not isinstance(profile, dict) or set(profile) - {"params", "options"}:
                raise ValueError(f"{role}/{model} profile must contain params and optional options")
            params, extra = profile.get("params", {}), profile.get("options", {})
            if not isinstance(params, dict) or not isinstance(extra, dict):
                raise ValueError(f"{role}/{model}: params and options must be dictionaries")
            parameters[role], options[role] = dict(params), dict(extra)
        return replace(
            self, gps_params=parameters["gps"], mu_params=parameters["mu"], model_options=options,
            model_params_file=str(path.resolve()), model_params_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def for_model(self, role):
        if role not in ("gps", "mu"):
            raise ValueError("role must be gps or mu")
        options = self.model_options.get(role, {})
        return replace(self, model_options={}, **options) if options else self
