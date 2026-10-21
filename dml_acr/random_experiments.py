"""Reproducible, bounded random screening with staged rejection and an audit ledger.

Detailed estimates survive only if both pretests and the 2023 total decline pass.
All proposed configurations and all attempted dispositions remain auditable.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
from .config import AnalysisConfig, INDUSTRY_CONTROLS, FIRM_CONTROLS
from .criteria import PatternCriteria, evaluate_pattern
from .io import file_hash, write_json

PRIMARY = ['total', 'sme', 'large', 'moderation']
YEARS = [2020, 2021, 2023, 2024, 2025]
CRITERIA = PatternCriteria(decline_effects=('total',))
DOMAINS = {
    'gps_features': 'uniform subset size 2..13 of the 13 industry controls',
    'mu_features': 'uniform subset size 2..18 of industry/firm controls and size dummies',
    'gps_model': ['linear', 'spline'], 'gps_n_knots': [3, 4, 5], 'gps_degree': [1, 2, 3],
    'mu_model_probabilities': {'xgboost': .65, 'ridge': .20, 'spline_ridge': .15},
    'xgboost': {'n_estimators': [80, 160, 240], 'max_depth': [2, 3, 4],
        'learning_rate': [.03, .05, .1], 'subsample': [.65, .8, 1.],
        'colsample_bytree': [.65, .8, 1.], 'min_child_weight': [1, 3, 5],
        'reg_alpha': [0., .1, 1.], 'reg_lambda': [1., 3., 10.]},
    'mu_ridge_alpha': [.1, 1., 10., 100.],
    'mu_spline_n_knots': [3, 4], 'mu_spline_degree': [1, 2, 3],
    'ridge_alpha_grids': [[.0001, .001, .01, .1, 1., 10.], [.01, .1, 1., 10., 100.]],
    'n_splits': [2, 3, 4, 5], 'n_repeats': [2, 3, 5], 'inner_splits': [2, 3],
    'include_industry_dummies': [False, True], 'industry_scheme': ['refined', 'major', 'none'],
    'industry_min_groups': [3, 4, 6, 8], 'industry_min_obs': [4, 6, 10],
    'ess': [3., 5., 8., 10.], 'min_ess_prime': [3., 5., 8.],
    'trim_threshold': [.05, .1, .2], 'density_floor': [.0025, .005, .01],
    'bandwidth_scale': [.35, .5, .7], 'diff_step': [.002, .004],
    'gap_policy': ['strict', 'bridge'], 'covariate_time': ['current', 'baseline'],
}


def numerical_source_hashes():
    folder=Path(__file__).parent
    names=['config.py','preprocessing.py','preprocessing_legacy.py','models.py','support.py',
           'support_legacy.py','score.py','aggregation.py','estimation.py','pipeline.py','criteria.py']
    return {name:file_hash(folder/name) for name in names}


def make_random_plan(root='results/random200',max_attempts=200,seed=20261021):
    root=Path(root)
    if (root/'plan.json').exists():raise FileExistsError('Frozen plan exists; resume it')
    if not 1<=max_attempts<=10000:raise ValueError('max_attempts must be 1..10000')
    rng=np.random.default_rng(seed)
    def pick(values):return values[int(rng.integers(len(values)))]
    def subset(values):
        count=int(rng.integers(2,len(values)+1))
        chosen=set(rng.choice(len(values),size=count,replace=False).tolist())
        return [v for i,v in enumerate(values) if i in chosen]
    reference=INDUSTRY_CONTROLS+FIRM_CONTROLS
    entries=[];seen=set()
    while len(entries)<max_attempts:
        mu_model=str(rng.choice(['xgboost','ridge','spline_ridge'],p=[.65,.20,.15]))
        if mu_model=='xgboost':
            params={'tree_method':'hist',**{key:pick(values) for key,values in DOMAINS['xgboost'].items()}}
        else:
            params={'alpha':pick(DOMAINS['mu_ridge_alpha'])}
            if mu_model=='spline_ridge':
                params.update(n_knots=pick(DOMAINS['mu_spline_n_knots']),degree=pick(DOMAINS['mu_spline_degree']))
        keys=['gps_model','gps_n_knots','gps_degree','n_splits','n_repeats','inner_splits',
            'include_industry_dummies','industry_scheme','industry_min_groups','industry_min_obs',
            'ess','min_ess_prime','trim_threshold','density_floor','bandwidth_scale','diff_step',
            'gap_policy','covariate_time']
        values={key:pick(DOMAINS[key]) for key in keys}
        run_id=f'random_{len(entries)+1:03d}'
        config=AnalysisConfig(**values,gps_features=subset(INDUSTRY_CONTROLS),
            mu_features=subset(['large_dummy','small_dummy']+FIRM_CONTROLS+INDUSTRY_CONTROLS),
            sample_reference_features=reference,mu_model=mu_model,mu_params=params,
            ridge_alphas=pick(DOMAINS['ridge_alpha_grids']),mu_search_n_iter=0,mu_search_space={},
            n_bootstrap=1999,label=run_id)
        if config.fingerprint in seen:continue
        seen.add(config.fingerprint)
        entries.append({'run_id':run_id,'config_id':config.fingerprint,'config':config.to_dict()})
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'experiment_type':'bounded_random_screen',
        'max_attempts':max_attempts,'generator':'numpy.default_rng.PCG64','generator_seed':seed,
        'data_sha256':file_hash(AnalysisConfig().data_path),'numerical_source_hashes':numerical_source_hashes(),
        'domains':DOMAINS,'pattern_criteria':CRITERIA.to_dict(),'primary_effects':PRIMARY,'years':YEARS,
        'detailed_save_gate':'All 8 pre-period primary p-values finite and >= .05; 2023 total finite, negative, p < .05.',
        'stop_rule':'Stop immediately on nominal pattern_match among gate-passing cases; preferred_match is optional.',
        'fixed':'Outcome, exposure/transform, baseline, size classification, alpha .05, industry clusters, model seed42; bootstrap1999.',
        'sample_note':'Full reference union filters cases in each timing design. Baseline/current covariates and support can change the realized sample; hashes are retained for accepted cases.',
        'retention':'All candidates and attempted dispositions retained; rejected detailed outcomes removed. Only accepted cases get detailed estimates/figures.',
        'inference_note':'Exploratory selection. Conservative Bonferroni family includes 20 tests for every planned candidate, even if stopped early; cumulative family also includes prior 64 specifications.',
        'prior_specifications':64,'entries':entries}
    write_json(root/'plan.json',plan)
    write_json(root/'pattern_criteria.json',CRITERIA.to_dict())
    return plan


def screen_gate(table,year,config):
    """Return a rejection reason, or None to continue. Missing cells never pass."""
    if year not in [2020,2021,2023]:return None
    effects=PRIMARY if year<2022 else ['total']
    for effect in effects:
        cell=table.loc[table.year.eq(year)&table.effect.eq(effect)]
        if len(cell)!=1:return {'reason':'incomplete_gate','year':year,'effect':effect}
        value=float(cell.estimate.iloc[0]);p=float(cell.p_value.iloc[0])
        if not (np.isfinite(value) and np.isfinite(p)):
            return {'reason':'nonfinite_gate','year':year,'effect':effect}
        if year<2022 and p<config.alpha:
            return {'reason':'pre_significant','year':year,'effect':effect}
        if year==2023 and not (value<0 and p<config.alpha):
            return {'reason':'initial_total_decline_not_met','year':year,'effect':effect}
    return None


def read_ledger(root):
    path=Path(root)/'all_attempts.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize_random(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());records=read_ledger(root)
    ledger=pd.DataFrame(records)
    if len(ledger):ledger.to_csv(root/'all_attempts.csv',index=False)
    family=plan['max_attempts']*len(PRIMARY)*len(YEARS)
    cumulative=family+plan['prior_specifications']*len(PRIMARY)*len(YEARS)
    tables=[]
    for record in records:
        if record['status']!='accepted':continue
        folder=root/'accepted'/record['run_id'];table=pd.read_csv(folder/'estimates.csv')
        table['run_id']=record['run_id'];table['attempt_number']=record['attempt_number']
        primary=table.effect.isin(PRIMARY)
        table.loc[primary,'p_bonferroni_random_budget']=np.minimum(1,table.loc[primary,'p_value']*family)
        table.loc[primary,'p_bonferroni_cumulative_budget']=np.minimum(1,table.loc[primary,'p_value']*cumulative)
        tables.append(table)
    estimates=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    if len(estimates):estimates.to_csv(root/'accepted_estimates.csv',index=False)
    matches=[r['run_id'] for r in records if r.get('pattern',{}).get('pattern_match')]
    preferred=[r['run_id'] for r in records if r.get('pattern',{}).get('preferred_match')]
    counts=Counter(r['status'] for r in records)
    stop='target_found' if matches else ('budget_exhausted' if len(records)>=plan['max_attempts'] else 'running')
    summary={'planned_max_attempts':plan['max_attempts'],'attempted':len(records),
        'accepted_detailed_cases':counts['accepted'],'rejected':counts['rejected'],'failed':counts['failed'],
        'pending':plan['max_attempts']-len(records),'stop_reason':stop,
        'nominal_pattern_matches':matches,'nominal_preferred_matches':preferred,
        'rejection_counts':dict(Counter(r.get('reason','') for r in records if r['status']=='rejected')),
        'random_budget_family_tests':family,'cumulative_budget_family_tests':cumulative,
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Exploratory adaptive selection; non-rejection is not evidence that parallel trends holds. No rejected detailed outcomes retained.'}
    if records:
        summary['elapsed_worker_seconds']=sum(r.get('elapsed_seconds',0) for r in records)
        summary['peak_rss_mib']=max(r.get('peak_rss_mib',0) for r in records)
    write_json(root/'summary.json',summary)
    text=['# 무작위 변수·설정 탐색', '',
        f"시도 {len(records)}/{plan['max_attempts']}; 상세 저장 {counts['accepted']}; 탈락 {counts['rejected']}; 오류 {counts['failed']}.",
        f"종료 상태: {stop}. 전체 패턴 {len(matches)}개, 우수 패턴 {len(preferred)}개.", '',
        '저장 조건: 2020·2021 전체/SME/Large/조절효과 8개 p≥0.05, 2023년 전체 효과 음수 및 p<0.05.',
        '전체 패턴: 저장 조건 + 세 집단의 2023→2024→2025 단조 회복 + SME의 더 빠른 양수 전환 + 음수 조절효과의 지속/강화.',
        '2025년 SME 유의한 양수는 우수 조건이며 자동 정지의 필수 조건은 아닙니다.',
        f'보정: 모든 계획 후보를 포함한 {family}개 검정, 기존 64개 설정 포함 {cumulative}개 검정 Bonferroni.',
        '탈락 상세 결과는 삭제하고, plan.json과 all_attempts.jsonl/CSV에 설정 및 탈락 사유만 남겼습니다.',
        '결과를 보고 선택한 탐색 결과이며 사전 비유의는 평행추세의 입증이 아닙니다.']
    (root/'SEARCH_SUMMARY_KO.md').write_text('\n'.join(text)+'\n')
    return summary


def package_random(root,destination=None):
    """Results/configuration only: never raw input data, nuisance models or scratch."""
    root=Path(root)
    path=Path(destination or root/'checkpoints'/'latest_random_results.zip');path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['plan.json','pattern_criteria.json','all_attempts.jsonl','all_attempts.csv',
                     'summary.json','SEARCH_SUMMARY_KO.md','accepted_estimates.csv']:
            candidate=root/name
            if candidate.exists():archive.write(candidate,candidate.relative_to(root))
        for candidate in sorted((root/'accepted').rglob('*')):
            if candidate.is_file() and candidate.name not in ['worker.log','progress.jsonl']:
                archive.write(candidate,candidate.relative_to(root))
    os.replace(temp,path)
    return path


def run_random(root='results/random200',limit=None):
    root=Path(root)
    if not (root/'plan.json').exists():make_random_plan(root)
    plan=json.loads((root/'plan.json').read_text())
    if plan['numerical_source_hashes']!=numerical_source_hashes():raise ValueError('Numerical source changed since frozen plan')
    if file_hash(AnalysisConfig().data_path)!=plan['data_sha256']:raise ValueError('Input data changed')
    records=read_ledger(root)
    if any(r.get('pattern',{}).get('pattern_match') for r in records):return summarize_random(root)
    seen={r['run_id'] for r in records};attempted=0
    for entry in plan['entries']:
        if entry['run_id'] in seen:continue
        if limit is not None and attempted>=limit:break
        config=AnalysisConfig.from_dict(entry['config'])
        if config.fingerprint!=entry['config_id']:raise ValueError('Candidate changed')
        folder=Path('work/random_search_scratch')/root.name/entry['run_id']
        # A previously interrupted candidate is rerun with the identical configuration.
        if folder.exists():shutil.rmtree(folder)
        folder.mkdir(parents=True);config.save(folder/'worker_config.json')
        print(f"START {entry['run_id']} ({config.mu_model}, GPS{len(config.gps_features)}, Mu{len(config.mu_features)})",flush=True)
        with (folder/'worker.log').open('w') as handle:
            worker=subprocess.run([sys.executable,'-m','dml_acr.random_experiments','worker',
                '--config',str(folder/'worker_config.json'),'--output',str(folder)],stdout=handle,stderr=subprocess.STDOUT)
        status_path=folder/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {'status':'failed','error':f'Worker exited {worker.returncode}'}
        record={'attempt_number':len(records)+1,'run_id':entry['run_id'],'config_id':entry['config_id'],
            'status':'accepted' if status['status']=='completed' else status['status'],
            'elapsed_seconds':status.get('elapsed_seconds',0),'peak_rss_mib':status.get('peak_rss_mib',0),
            'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        if status['status']=='completed':
            record['pattern']=status['pattern']
            destination=root/'accepted'/entry['run_id'];destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():shutil.rmtree(destination)
            (folder/'worker.log').unlink(missing_ok=True);(folder/'worker_config.json').unlink(missing_ok=True)
            (folder/'progress.jsonl').unlink(missing_ok=True)
            shutil.move(str(folder),destination)
        else:
            record.update(reason=status.get('gate',{}).get('reason',status.get('error','unknown')),
                rejection_year=status.get('gate',{}).get('year'),
                rejection_effect=status.get('gate',{}).get('effect'),years_evaluated=status.get('years_evaluated',[]))
            shutil.rmtree(folder)
        records.append(record)
        # Atomic rewrite prevents a partially appended JSON line on interruption.
        text=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records)
        temporary=root/'all_attempts.jsonl.tmp';temporary.write_text(text);os.replace(temporary,root/'all_attempts.jsonl')
        summary=summarize_random(root);package_random(root)
        print(json.dumps(summary,ensure_ascii=False),flush=True);attempted+=1
        if record.get('pattern',{}).get('pattern_match'):break
    summary=summarize_random(root);package_random(root)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['plan','search','summarize','worker'])
    parser.add_argument('--output',default='results/random200');parser.add_argument('--config')
    parser.add_argument('--max-attempts',type=int,default=200);parser.add_argument('--seed',type=int,default=20261021)
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.command=='plan':
        plan=make_random_plan(args.output,args.max_attempts,args.seed)
        result={key:value for key,value in plan.items() if key not in ['entries','domains','numerical_source_hashes']}
    elif args.command=='worker':
        from .pipeline import run_analysis
        _,result=run_analysis(AnalysisConfig.load(args.config),args.output,year_gate=screen_gate,pattern_criteria=CRITERIA)
    elif args.command=='search':result=run_random(args.output,args.limit)
    else:result=summarize_random(args.output)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
