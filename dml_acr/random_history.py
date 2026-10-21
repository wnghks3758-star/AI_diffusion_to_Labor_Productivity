"""Combine retained cases and all trial dispositions across random-search batches."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .io import file_hash,write_json
from .random_experiments import PRIMARY,YEARS,read_ledger


def combine_history(current_root='results/random201_400',output=None):
    current_root=Path(current_root);plan=json.loads((current_root/'plan.json').read_text())
    destination=Path(output or current_root/'cumulative');destination.mkdir(parents=True,exist_ok=True)
    roots=[Path(p) for p in plan.get('prior_plan_roots',[])]+[current_root]
    all_records=[];tables=[];seen_configs=set();seen_runs=set();plans=[];random_budget=0
    for root in roots:
        frozen=json.loads((root/'plan.json').read_text());records=read_ledger(root)
        random_budget+=frozen['max_attempts'];offset=frozen.get('generator_offset',0)
        for entry in frozen['entries']:
            if entry['config_id'] in seen_configs:raise ValueError('Duplicate configuration across batches')
            if entry['run_id'] in seen_runs:raise ValueError('Duplicate run ID across batches')
            seen_configs.add(entry['config_id']);seen_runs.add(entry['run_id'])
        for record in records:
            all_records.append({**record,'batch':frozen.get('batch_number',1),
                'global_attempt_number':offset+record['attempt_number']})
        if (root/'accepted_estimates.csv').exists():
            table=pd.read_csv(root/'accepted_estimates.csv')
            table['batch']=frozen.get('batch_number',1)
            table['global_attempt_number']=offset+table.attempt_number
            tables.append(table)
        plans.append({'root':str(root),'plan_sha256':file_hash(root/'plan.json'),
            'planned':frozen['max_attempts'],'attempted':len(records)})
    all_budget=plan['prior_specifications']+plan['max_attempts']
    ledger=pd.DataFrame(all_records)
    if len(ledger):ledger.to_csv(destination/'all_random_attempts.csv',index=False)
    estimates=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    if len(estimates):
        mask=estimates.effect.isin(PRIMARY)
        estimates['p_bonferroni_cumulative_at_batch']=estimates.get('p_bonferroni_cumulative_budget')
        estimates.loc[mask,'p_bonferroni_all_random_budget']=np.minimum(1,estimates.loc[mask,'p_value']*random_budget*len(PRIMARY)*len(YEARS))
        estimates.loc[mask,'p_bonferroni_cumulative_budget']=np.minimum(1,estimates.loc[mask,'p_value']*all_budget*len(PRIMARY)*len(YEARS))
        estimates.to_csv(destination/'all_retained_estimates.csv',index=False)
    summary={'random_planned':random_budget,'random_attempted':len(all_records),
        'retained':sum(r['status']=='accepted' for r in all_records),
        'rejected':sum(r['status']=='rejected' for r in all_records),
        'failed':sum(r['status']=='failed' for r in all_records),
        'nominal_pattern_matches':[r['run_id'] for r in all_records if r.get('pattern',{}).get('pattern_match')],
        'nominal_preferred_matches':[r['run_id'] for r in all_records if r.get('pattern',{}).get('preferred_match')],
        'all_random_family_tests':random_budget*20,'all_exploration_family_tests':all_budget*20,
        'unique_configuration_count':len(seen_configs),'plans':plans,
        'note':'Only gate-passing cases have retained detailed outcomes. Both batch failures and rejected/unevaluated outcomes remain in correction budgets.'}
    write_json(destination/'cumulative_summary.json',summary)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/random201_400')
    parser.add_argument('--output');args=parser.parse_args()
    print(json.dumps(combine_history(args.root,args.output),ensure_ascii=False))
