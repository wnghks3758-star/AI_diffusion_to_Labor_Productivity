"""Check the original notebook in a real kernel with a reduced analysis budget.

This never edits or executes the original full-budget analysis cells.
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key, relative in {
    'JUPYTER_CONFIG_DIR': 'work/jupyter/config',
    'JUPYTER_DATA_DIR': '.venv/share/jupyter',
    'JUPYTER_RUNTIME_DIR': 'work/jupyter/runtime',
    'IPYTHONDIR': 'work/ipython',
    'MPLCONFIGDIR': 'work/matplotlib',
    'XDG_CACHE_HOME': 'work/cache',
}.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(directory)

import nbformat
from nbclient import NotebookClient

original = json.loads((ROOT / 'f4_7_DML_ACR_Dart_size_mean_difference_chunked.ipynb').read_text())
first_code = next(''.join(c['source']) for c in original['cells'] if c['cell_type'] == 'code')
data_check = '''
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import sys

required = ['Year', 'firm_code', 'ind_code', 'sale', 'ln_firm_prod', 'office_ratio',
    'firm_rnd_ratio', 'ln_firm_avg_pay', 'ln_firm_ast',
    'ln_ind_avg_pay', 'ln_ind_ast', 'ind_rnd_ratio', 'ln_cum_patent',
    'ind_college_share_pre', 'ind_large_est_share_pre', 'ind_avg_age_pre',
    'ind_age50_share_pre', 'ind_female_share_pre', 'ind_permanent_share_pre',
    'ind_avg_tenure_pre', 'ind_avg_hours_pre', 'ind_ln_avg_pay_pre']
report = {'python': sys.version, 'executable': sys.executable,
    'checked_at_utc': datetime.now(timezone.utc).isoformat(),
    'full_analysis_executed': False,
    'packages': {p: md.version(p) for p in ['numpy','pandas','scipy','scikit-learn',
        'xgboost-cpu','openpyxl','matplotlib','jupyterlab','ipykernel']},
    'files': [], 'smoke_runs': []}
for path in sorted(Path('data/main_data').glob('*.xlsx')):
    data = pd.read_excel(path)
    missing = sorted(set(required) - set(data.columns))
    assert not missing, (path, missing)
    duplicates = int(data.duplicated(['firm_code','Year']).sum())
    assert duplicates == 0, (path, duplicates)
    dose_valid = bool((np.isfinite(data.office_ratio) & data.office_ratio.gt(0)).all())
    assert dose_valid, path
    report['files'].append({'path': str(path), 'rows': len(data), 'columns': len(data.columns),
        'years': sorted(int(v) for v in data.Year.dropna().unique()),
        'duplicate_firm_year': duplicates, 'dose_positive_finite': dose_valid,
        'missing_required_columns': missing, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    print(report['files'][-1])
del data
'''
smoke = '''
# Temporary overrides exist only in this validation kernel.
# Exercise RandomizedSearchCV and exact XGBoost without running 150 candidates.
original_tuner = tune_xgb_model
original_parameter_grid = get_xgb_param_dist
def get_xgb_param_dist():
    return {'n_estimators': [8, 12], 'max_depth': [2], 'learning_rate': [0.1],
            'subsample': [0.8], 'colsample_bytree': [0.8]}
def tune_xgb_model(X, y, cv_indices, scoring, n_iter=150, options=None):
    return original_tuner(X, y, cv_indices, scoring, n_iter=2, options=options)

for policy in ['bridge', 'strict']:
    result = action_everything(var_name='office_ratio', group_col='ind_code', base_year=2022,
        n_splits=2, n_repeats=1, n_bootstrap=49, diff_step=0.05,
        ess=5, min_ess_prime=5, bootstrap_unit='industry', ind_counts=4,
        gap_policy=policy, size_difference='means',
        options={'memory_mode':'event_only', 'grid_chunk_points':4096,
                 'search_n_jobs':2, 'crossfit_n_jobs':2})
    assert set(result.year) == {2020, 2021, 2022, 2023, 2024, 2025}, result.year
    finite_cols = ['atet_o', 'sme_atet_o', 'large_atet_o', 'moderation_o',
                   'atet_se', 'sme_se', 'large_se']
    assert set(finite_cols).issubset(result.columns), list(result.columns)
    assert np.isfinite(result[finite_cols].to_numpy(dtype=float)).all(), result[finite_cols]
    assert result.attrs['gap_policy'] == policy
    report['smoke_runs'].append({'gap_policy': policy, 'status': 'PASS',
        'years': result.year.tolist(), 'n_splits': 2, 'n_repeats': 1,
        'n_bootstrap': 49, 'diff_step': 0.05, 'xgb_search_candidates': 2,
        'xgb_n_estimators': [8, 12], 'finite_columns': finite_cols,
        'grid_storage_audit': result.attrs['grid_storage_audit']})
    plt.close('all')
tune_xgb_model = original_tuner
get_xgb_param_dist = original_parameter_grid
report['status'] = 'PASS'
Path('validation/environment_report.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print('Environment validation: PASS (reduced-budget tests only).')
'''

notebook = nbformat.v4.new_notebook(cells=[
    nbformat.v4.new_code_cell('%matplotlib inline'),
    nbformat.v4.new_code_cell(first_code),
    nbformat.v4.new_code_cell(data_check),
    nbformat.v4.new_code_cell(smoke),
])
client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT)}})
try:
    client.execute()
finally:
    destination = ROOT / 'work' / 'environment_check.executed.ipynb'
    nbformat.write(notebook, destination)
    for cell in notebook.cells:
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='')
            elif output.output_type == 'error':
                print('\n'.join(output.traceback), file=sys.stderr)
