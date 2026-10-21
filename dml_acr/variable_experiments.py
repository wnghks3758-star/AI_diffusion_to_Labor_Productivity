"""Variable-only ablation with a frozen complete-case sample and fixed Mu parameters."""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from .config import AnalysisConfig,INDUSTRY_CONTROLS,FIRM_CONTROLS
from .criteria import PatternCriteria,holm_adjust
from .experiments import run_plan,summarize
from .io import file_hash,write_json

VARIABLE_NAMES={
    'ln_ind_avg_pay':'산업 평균임금(로그)','ln_ind_ast':'산업 자산(로그)',
    'ind_rnd_ratio':'산업 R&D 비율','ln_cum_patent':'누적 특허(로그)',
    'ind_college_share_pre':'사전 대졸자 비중','ind_large_est_share_pre':'사전 대규모 사업체 비중',
    'ind_avg_age_pre':'사전 평균연령','ind_age50_share_pre':'사전 50세 이상 비중',
    'ind_female_share_pre':'사전 여성 비중','ind_permanent_share_pre':'사전 상용직 비중',
    'ind_avg_tenure_pre':'사전 평균근속','ind_avg_hours_pre':'사전 평균근로시간',
    'ind_ln_avg_pay_pre':'사전 평균임금(로그)','firm_rnd_ratio':'기업 R&D 비율',
    'ln_firm_avg_pay':'기업 평균임금(로그)','ln_firm_ast':'기업 자산(로그)'}

def make_variable_plan(output_dir='results/variables32'):
    root=Path(output_dir)
    if (root/'plan.json').exists():raise FileExistsError('Plan already exists; resume it rather than overwriting')
    (root/'configs').mkdir(parents=True,exist_ok=True)
    reference=list(INDUSTRY_CONTROLS)+list(FIRM_CONTROLS)
    base=AnalysisConfig(n_splits=3,n_repeats=3,n_bootstrap=999,diff_step=.002,
        mu_search_n_iter=0,mu_search_space={},
        mu_params={'tree_method':'hist','n_estimators':160,'max_depth':3,'learning_rate':.05,
                   'subsample':.8,'colsample_bytree':.8},
        sample_reference_features=reference)
    designs=[('baseline','baseline','전체 변수 기준',{})]
    for feature in INDUSTRY_CONTROLS:
        designs.append(('gps_remove',feature,VARIABLE_NAMES[feature],
                        {'gps_features':[f for f in base.gps_features if f!=feature]}))
    for feature in INDUSTRY_CONTROLS+FIRM_CONTROLS:
        designs.append(('mu_remove',feature,VARIABLE_NAMES[feature],
                        {'mu_features':[f for f in base.mu_features if f!=feature]}))
    core=['ln_ind_ast','ln_cum_patent']
    workforce=INDUSTRY_CONTROLS[4:]
    common_mu=['large_dummy','small_dummy']+FIRM_CONTROLS
    designs.extend([
        ('core_block','core','산업 자산·특허 기본군',{'gps_features':core,'mu_features':common_mu+core}),
        ('add_workforce_block','workforce','기본군에 사전 노동구성 9개 추가',
            {'gps_features':core+workforce,'mu_features':common_mu+core+workforce})])
    assert len(designs)==32
    entries=[]
    for i,(action,feature,description,changes) in enumerate(designs,start=1):
        run_id=f'var_{i:02d}';values=deepcopy(base.to_dict());values.update(changes)
        values['label']=f'{run_id}_{action}_{feature}'
        config=AnalysisConfig.from_dict(values);path=root/'configs'/f'{run_id}.json';config.save(path)
        entries.append({'run_id':run_id,'config':str(path.relative_to(root)),'config_id':config.fingerprint,
            'label':config.label,'model_design':action,'support_design':'fixed_ess5_strict',
            'variable_action':action,'variable':feature,'variable_description':description})
    assert len({entry['config_id'] for entry in entries})==32
    plan={'created_at_utc':datetime.now(timezone.utc).isoformat(),'experiment_type':'variable_only_ablation',
        'planned_runs':32,'data_sha256':file_hash(base.data_path),
        'selection':'Fixed baseline + 13 GPS removals + 16 Mu removals + 2 nested covariate blocks.',
        'budget':'3 repeats, 999 multipliers, step .002; fixed XGB parameters, tuning disabled.',
        'fixed_sample_reference_features':reference,
        'sample_note':'Complete cases of the full reference union are used in every specification; realized support may vary with GPS.',
        'primary_family':['total','sme','large','moderation'],'years':base.years,
        'pattern_criteria':PatternCriteria().to_dict(),'entries':entries}
    write_json(root/'plan.json',plan);write_json(root/'pattern_criteria.json',plan['pattern_criteria'])
    return plan

def compare_variables(root='results/variables32',previous_root='results/screen32'):
    root=Path(root);summary,ledger,estimates=summarize(root)
    if estimates.empty:return summary
    baseline=estimates.loc[estimates.run_id.eq('var_01')].set_index(['year','effect'])
    if baseline.empty:return summary
    comparisons=estimates.merge(baseline[['estimate','p_value','coverage']].rename(columns={
        'estimate':'baseline_estimate','p_value':'baseline_p_value','coverage':'baseline_coverage'}),
        left_on=['year','effect'],right_index=True,how='left')
    comparisons['estimate_change']=comparisons.estimate-comparisons.baseline_estimate
    comparisons['coverage_change']=comparisons.coverage-comparisons.baseline_coverage
    comparisons['descriptive_change_only']=True
    comparisons=comparisons.merge(ledger[['run_id','variable_action','variable','variable_description']],on='run_id')
    comparisons.to_csv(root/'variable_comparisons.csv',index=False)
    primary=comparisons.effect.isin(['total','sme','large','moderation'])
    main=comparisons.loc[primary]
    main.loc[main.year.eq(2025)].to_csv(root/'variable_comparisons_2025.csv',index=False)
    # Cumulative family retains the first 32-setting exploration as well.
    old_path=Path(previous_root)/'all_estimates.csv'
    if old_path.exists():
        old=pd.read_csv(old_path);old=old.loc[old.effect.isin(['total','sme','large','moderation'])].copy()
        old['suite']='model_support_screen';new=main.copy();new['suite']='variable_ablation'
        cumulative=pd.concat([old,new],ignore_index=True)
        cumulative['p_holm_cumulative_64_specs']=holm_adjust(cumulative.p_value.to_numpy(),family_size=1280)
        cumulative.to_csv(root/'cumulative_64spec_primary_estimates.csv',index=False)
    sample_audits=[]
    for run_id in ledger.loc[ledger.status.eq('completed'),'run_id']:
        audit=json.loads((root/'runs'/run_id/'preprocessing.json').read_text())
        sample_audits.append({'run_id':run_id,'n_firms':audit['analysis_firms'],
            'n_industries':audit['analysis_industries'],'sample_hash':audit['sample_firm_sha256']})
    fixed=bool(sample_audits and len({r['sample_hash'] for r in sample_audits})==1)
    if sample_audits and not fixed:raise ValueError('Variable comparison sample changed across specifications')
    write_json(root/'fixed_sample_audit.json',{'fixed_sample_verified':fixed,'specifications':sample_audits})
    summary.update(experiment_type='variable_only_ablation',fixed_sample_verified=fixed,
        comparison_note='Estimate changes are descriptive; no p-values for between-specification changes are computed.')
    write_json(root/'summary.json',summary)
    return summary

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['plan','search','summarize'])
    parser.add_argument('--output',default='results/variables32');parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    root=Path(args.output)
    if args.command=='plan':result=make_variable_plan(root)
    else:
        if not (root/'plan.json').exists():make_variable_plan(root)
        if args.command=='search':run_plan(root,limit=args.limit)
        result=compare_variables(root)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
