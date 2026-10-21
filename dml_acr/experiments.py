"""Frozen 32-specification plan, isolated workers, resume and complete-result ledger."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone
import json
import subprocess
import sys
import zipfile
import os
import numpy as np
import pandas as pd
from .config import AnalysisConfig,INDUSTRY_CONTROLS,FIRM_CONTROLS,PRE_CONTROLS
from .criteria import PatternCriteria,evaluate_pattern,holm_adjust
from .io import write_json,file_hash
from .visualization import plot_specifications

def make_screen_plan(output_dir='results/screen32'):
    root=Path(output_dir);(root/'configs').mkdir(parents=True,exist_ok=True)
    base=AnalysisConfig(n_splits=3,n_repeats=3,n_bootstrap=999,diff_step=.002,
        mu_search_n_iter=4,mu_params={'tree_method':'hist','n_estimators':160,'max_depth':3,'learning_rate':.05},
        mu_search_space={'n_estimators':[100,160],'max_depth':[2,3],
                         'learning_rate':[.03,.08],'subsample':[.8,1.], 'colsample_bytree':[.8,1.]})
    # Eight model/control designs, four support designs; fixed before outcome inspection.
    designs=[('base',{}),('gps_linear',{'gps_model':'linear','n_splits':2}),
        ('gps_flexible',{'gps_n_knots':5,'gps_degree':2,'industry_min_groups':6}),
        ('mu_ridge',{'mu_model':'ridge','mu_params':{'alpha':1.},'mu_search_space':{}}),
        ('mu_spline',{'mu_model':'spline_ridge','mu_params':{'n_knots':3,'degree':2,'alpha':10.},'mu_search_space':{},'n_splits':4}),
        ('shared_controls',{'gps_level':'firm','gps_features':['large_dummy','small_dummy']+FIRM_CONTROLS+INDUSTRY_CONTROLS}),
        ('baseline_pre_controls',{'covariate_time':'baseline','gps_features':PRE_CONTROLS,
            'mu_features':['large_dummy','small_dummy']+FIRM_CONTROLS+PRE_CONTROLS,'industry_min_groups':6}),
        ('no_industry_dummies',{'include_industry_dummies':False,'industry_min_groups':8,'n_splits':4})]
    supports=[('ess5_strict',{'ess':5.,'min_ess_prime':5.,'trim_threshold':.1,'gap_policy':'strict'}),
        ('ess8_trim05',{'ess':8.,'min_ess_prime':5.,'trim_threshold':.05,'gap_policy':'strict'}),
        ('ess10_bw07',{'ess':10.,'min_ess_prime':8.,'trim_threshold':.1,'bandwidth_scale':.7,'gap_policy':'strict'}),
        ('ess5_bridge',{'ess':5.,'min_ess_prime':0.,'trim_threshold':.2,'gap_policy':'bridge'})]
    entries=[]
    for design_name,model_changes in designs:
        for support_name,support_changes in supports:
            index=len(entries)+1;run_id=f'spec_{index:02d}'
            values=deepcopy(base.to_dict());values.update(deepcopy(model_changes));values.update(support_changes)
            values['label']=f'{run_id}_{design_name}_{support_name}'
            config=AnalysisConfig.from_dict(values)
            path=root/'configs'/f'{run_id}.json';config.save(path)
            entries.append({'run_id':run_id,'config':str(path.relative_to(root)),'config_id':config.fingerprint,
                'label':config.label,'model_design':design_name,'support_design':support_name})
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'planned_runs':len(entries),
        'data_sha256':file_hash(base.data_path),'selection':'Fixed 8 model/control designs × 4 support designs; no outcome-adaptive search.',
        'budget':'Exploratory screen: 3 repetitions, 999 multipliers, step .002; XGB histogram, 4 search candidates.',
        'primary_family':['total','sme','large','moderation'],'years':base.years,
        'pattern_criteria':PatternCriteria().to_dict(),'entries':entries}
    write_json(root/'plan.json',manifest)
    write_json(root/'pattern_criteria.json',PatternCriteria().to_dict())
    return manifest

def summarize(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());ledgers=[];tables=[]
    for entry in plan['entries']:
        folder=root/'runs'/entry['run_id'];status_path=folder/'status.json'
        record={**entry,'status':'pending'}
        if status_path.exists():
            status=json.loads(status_path.read_text());record.update({k:v for k,v in status.items() if k not in ['versions','pattern','traceback']})
            if status['status']=='completed':
                data=pd.read_csv(folder/'estimates.csv');data['run_id']=entry['run_id'];data['label']=entry['label']
                tables.append(data);record.update(evaluate_pattern(data,PatternCriteria(**plan['pattern_criteria'])))
        ledgers.append(record)
    ledger=pd.DataFrame(ledgers)
    estimates=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    if len(estimates):
        primary=estimates.effect.isin(plan['primary_family'])
        # Missing/failed specifications still count toward the preplanned family size.
        family=plan['planned_runs']*len(plan['primary_family'])*len(plan['years'])
        estimates.loc[primary,'p_holm_across_specs']=holm_adjust(estimates.loc[primary,'p_value'],family_size=family)
        estimates.loc[primary,'p_bonferroni_across_specs']=np.minimum(1,estimates.loc[primary,'p_value']*family)
        estimates.to_csv(root/'all_estimates.csv',index=False)
        robust=[]
        for run_id,table in estimates.groupby('run_id'):
            flag=evaluate_pattern(table,PatternCriteria(**plan['pattern_criteria']),p_column='p_holm_across_specs')
            robust.append({'run_id':run_id,'holm_pattern_match':flag['pattern_match'],
                'holm_preferred_match':flag['preferred_match']})
        ledger=ledger.merge(pd.DataFrame(robust),on='run_id',how='left')
    ledger.to_csv(root/'all_specifications.csv',index=False)
    if 'pattern_match' in ledger:
        matches=ledger.loc[ledger.pattern_match.eq(True)]
        preferred=ledger.loc[ledger.preferred_match.eq(True)]
    else:matches=preferred=ledger.iloc[:0]
    matches.to_csv(root/'matching_specifications.csv',index=False)
    preferred.to_csv(root/'preferred_specifications.csv',index=False)
    summary={'planned':len(ledger),'completed':int(ledger.status.eq('completed').sum()),
        'failed':int(ledger.status.eq('failed').sum()),'pending':int(ledger.status.eq('pending').sum()),
        'nominal_pattern_matches':len(matches),'nominal_preferred_matches':len(preferred),
        'holm_pattern_matches':int(ledger.get('holm_pattern_match',pd.Series(dtype=bool)).eq(True).sum()),
        'holm_preferred_matches':int(ledger.get('holm_preferred_match',pd.Series(dtype=bool)).eq(True).sum()),
        'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'note':'Pattern flags are exploratory. Pre-period non-rejection does not establish parallel trends.'}
    flag_names=['pre_all_nonsignificant','initial_significant_decline','recovery',
        'sme_crosses_positive_earlier','sme_2025_significantly_positive','moderation_negative_significant']
    summary['criteria_counts']={name:int(ledger.get(name,pd.Series(dtype=bool)).eq(True).sum()) for name in flag_names}
    if 'peak_rss_mib' in ledger:
        summary['peak_rss_mib']={'minimum':float(ledger.peak_rss_mib.min()),
            'median':float(ledger.peak_rss_mib.median()),'maximum':float(ledger.peak_rss_mib.max())}
    write_json(root/'summary.json',summary)
    text=['# Exploratory specification search', '',f"Completed: {summary['completed']}/{summary['planned']}; failed: {summary['failed']}; pending: {summary['pending']}.",
        f"Nominal pattern matches: {len(matches)}; preferred matches: {len(preferred)}.",
        f"Holm-adjusted pattern matches: {summary['holm_pattern_matches']}.",'',
        'Every planned specification, error, estimate and selection flag is retained.',
        'These are exploratory estimates using a screening computation budget.',
        'A nominal match is not independent confirmation; failed pretests are not proof of parallel trends.',
        'See all_specifications.csv, all_estimates.csv, plan.json and each runs/<run_id> folder.']
    (root/'SEARCH_SUMMARY.md').write_text('\n'.join(text)+'\n')
    return summary,ledger,estimates

def package_checkpoint(root,run_id=None):
    root=Path(root);destination=root/'checkpoints';destination.mkdir(exist_ok=True)
    name=f'{run_id}.zip' if run_id else 'latest_summary.zip';path=destination/name
    names=['plan.json','pattern_criteria.json','all_specifications.csv','all_estimates.csv',
        'matching_specifications.csv','preferred_specifications.csv','summary.json','SEARCH_SUMMARY.md']
    temporary=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            candidate=root/name
            if candidate.exists():archive.write(candidate,candidate.relative_to(root))
        if run_id:
            for candidate in sorted((root/'runs'/run_id).rglob('*')):
                if candidate.is_file():archive.write(candidate,candidate.relative_to(root))
            config=root/'configs'/f'{run_id}.json';archive.write(config,config.relative_to(root))
    os.replace(temporary,path)
    return path

def run_plan(root='results/screen32',limit=None,retry_failed=False):
    root=Path(root)
    if not (root/'plan.json').exists():make_screen_plan(root)
    plan=json.loads((root/'plan.json').read_text());attempted=0
    for entry in plan['entries']:
        folder=root/'runs'/entry['run_id'];status_path=folder/'status.json'
        if status_path.exists():
            status=json.loads(status_path.read_text())
            if status['status']=='completed' or not retry_failed:continue
        if limit is not None and attempted>=limit:break
        config_path=Path(entry['config'])
        if not config_path.is_absolute():config_path=root/config_path
        if not config_path.exists():config_path=root/'configs'/config_path.name
        config=AnalysisConfig.load(config_path)
        if config.fingerprint!=entry['config_id']:raise ValueError('Frozen plan configuration changed')
        if file_hash(config.data_path)!=plan['data_sha256']:raise ValueError('Data changed since plan creation')
        folder.mkdir(parents=True,exist_ok=True)
        print(f"Starting {entry['label']}",flush=True)
        with (folder/'worker.log').open('w') as log:
            worker=subprocess.run([sys.executable,'-m','dml_acr','run','--config',str(config_path),'--output',str(folder)],
                stdout=log,stderr=subprocess.STDOUT)
        if not status_path.exists():write_json(status_path,{'status':'failed','error':f'Worker exited {worker.returncode}'})
        summary,_,_=summarize(root);package_checkpoint(root,entry['run_id']);package_checkpoint(root)
        print(json.dumps(summary),flush=True);attempted+=1
    summary,ledger,estimates=summarize(root)
    if len(estimates):plot_specifications(estimates,root/'specification_plot_2025.png')
    package_checkpoint(root)
    return summary
