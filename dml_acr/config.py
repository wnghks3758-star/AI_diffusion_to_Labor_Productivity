"""Validated JSON configuration; all estimand and computation settings are explicit."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json
import math

INDUSTRY_CONTROLS = [
    'ln_ind_avg_pay', 'ln_ind_ast', 'ind_rnd_ratio', 'ln_cum_patent',
    'ind_college_share_pre', 'ind_large_est_share_pre', 'ind_avg_age_pre',
    'ind_age50_share_pre', 'ind_female_share_pre', 'ind_permanent_share_pre',
    'ind_avg_tenure_pre', 'ind_avg_hours_pre', 'ind_ln_avg_pay_pre']
FIRM_CONTROLS = ['firm_rnd_ratio', 'ln_firm_avg_pay', 'ln_firm_ast']
PRE_CONTROLS = INDUSTRY_CONTROLS[3:]

@dataclass
class AnalysisConfig:
    data_path: str = 'data/main_data/firm_year_all_ind_dart_add_control_gpt_continuous.xlsx'
    base_year: int = 2022
    years: list = field(default_factory=lambda: [2020, 2021, 2023, 2024, 2025])
    dose: str = 'office_ratio'
    outcome: str = 'ln_firm_prod'
    dose_transform: str = 'log_percent'
    pre_sign: str = 'legacy'  # legacy: Y_base-Y_pre; chronological: Y_pre-Y_base
    covariate_time: str = 'current'  # baseline freezes controls at base_year
    gps_features: list = field(default_factory=lambda: list(INDUSTRY_CONTROLS))
    mu_features: list = field(default_factory=lambda: ['large_dummy','small_dummy'] + FIRM_CONTROLS + INDUSTRY_CONTROLS)
    sample_reference_features: list = field(default_factory=list)
    gps_level: str = 'industry'
    gps_model: str = 'spline'
    gps_n_knots: int = 3
    gps_degree: int = 3
    ridge_alphas: list = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
    gps_variance_floor: float = 0.0001
    mu_model: str = 'xgboost'
    mu_params: dict = field(default_factory=lambda: {'tree_method':'exact', 'n_estimators':300, 'max_depth':3, 'learning_rate':0.03})
    mu_search_space: dict = field(default_factory=lambda: {
        'max_depth':[2,3,4,5], 'learning_rate':[0.01,0.03,0.05],
        'n_estimators':[200,300,500,700], 'subsample':[0.5,0.6,0.7,0.8,0.9],
        'colsample_bytree':[0.5,0.6,0.7,0.8,0.9], 'min_child_weight':[1,3,5,7,10],
        'gamma':[0,0.1,0.5,1,2], 'reg_alpha':[0,0.1,0.5,1,5,10],
        'reg_lambda':[1,3,5,10,20]})
    mu_search_n_iter: int = 150
    tuning_scope: str = 'global_cv'  # nested: tune Mu within each outer training fold
    n_splits: int = 4
    n_repeats: int = 50
    inner_splits: int = 3
    include_industry_dummies: bool = True
    industry_min_groups: int = 4
    industry_min_obs: int = 6
    industry_scheme: str = 'refined'  # refined / major / none
    small_industry_max_firms: int = 1
    size_quantile: float = 0.5
    ess: float = 5.0
    min_ess_prime: float = 5.0
    trim_threshold: float = 0.1
    density_floor: float = 0.005
    bandwidth_scale: float = 0.5
    min_bandwidth: float = 0.01
    diff_step: float = 0.0005
    max_dose: float = 10.0
    gap_policy: str = 'strict'
    size_difference: str = 'means'
    bootstrap_unit: str = 'industry'
    n_bootstrap: int = 5000
    alpha: float = 0.05
    seed: int = 42
    model_jobs: int = 2
    model_threads: int = 1
    support_batch_points: int = 256
    grid_chunk_points: int = 4096
    prediction_batch_points: int = 8
    max_prediction_rows: int = 50000
    bootstrap_batch_size: int = 256
    cache_dir: str = 'work/cache/dml_acr'
    save_bootstrap_draws: bool = False
    label: str = 'baseline'

    def __post_init__(self):
        choices = {'dose_transform':('log_percent','none'), 'pre_sign':('legacy','chronological'),
            'covariate_time':('current','baseline'), 'gps_level':('industry','firm'),
            'gps_model':('linear','spline'), 'mu_model':('xgboost','ridge','spline_ridge'),
            'industry_scheme':('refined','major','none'), 'gap_policy':('bridge','strict'),
            'size_difference':('means','common'), 'bootstrap_unit':('industry','firm'),
            'tuning_scope':('global_cv','nested')}
        for key, values in choices.items():
            if getattr(self,key) not in values: raise ValueError(f'{key}: expected {values}')
        for key in ['diff_step','bandwidth_scale','min_bandwidth','density_floor','gps_variance_floor']:
            if not math.isfinite(getattr(self,key)) or getattr(self,key) <= 0: raise ValueError(f'{key} must be positive')
        for key in ['n_splits','inner_splits','n_bootstrap','gps_n_knots']:
            if not isinstance(getattr(self,key),int) or getattr(self,key)<2: raise ValueError(f'{key} must be >=2')
        for key in ['n_repeats','model_jobs','model_threads','industry_min_groups','industry_min_obs',
                    'support_batch_points','grid_chunk_points','prediction_batch_points','max_prediction_rows','bootstrap_batch_size','gps_degree']:
            if not isinstance(getattr(self,key),int) or getattr(self,key)<1: raise ValueError(f'{key} must be >=1')
        if not 0 < self.trim_threshold <= 1: raise ValueError('trim_threshold must be in (0,1]')
        if not 0 < self.alpha < 1 or not 0 < self.size_quantile < 1: raise ValueError('Invalid alpha/size_quantile')
        if self.ess < 0 or self.min_ess_prime < 0: raise ValueError('ESS thresholds must be >=0')
        if self.mu_search_n_iter < 0: raise ValueError('mu_search_n_iter must be >=0')
        if not self.years or len(set(self.years)) != len(self.years) or self.base_year in self.years:
            raise ValueError('years must be unique and exclude base_year')
        for features in [self.gps_features,self.mu_features,self.sample_reference_features]:
            if len(features)!=len(set(features)): raise ValueError('Duplicate features')
            if self.dose in features or self.outcome in features or 'Y_diff' in features:
                raise ValueError('Do not include the dose, outcome or Y_diff as a covariate')
        if not self.ridge_alphas or min(self.ridge_alphas)<=0: raise ValueError('ridge_alphas must be positive')

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, values): return cls(**values)

    @classmethod
    def load(cls, path): return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2)+'\n')

    @property
    def fingerprint(self):
        values=self.to_dict()
        for key in ['label','cache_dir']: values.pop(key)
        # Preserve fingerprints of configurations written before fixed-sample support.
        if not values['sample_reference_features']: values.pop('sample_reference_features')
        return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()[:16]
