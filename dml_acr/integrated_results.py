"""Consolidate the complete audit and only current-gate-passing detailed cases."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from .adaptive_experiments import OLD_ROOTS
from .config import AnalysisConfig
from .criteria import evaluate_pattern
from .io import write_json, file_hash
from .random_experiments import CRITERIA, PRIMARY, YEARS, read_ledger, screen_gate


def integrate(current='results/adaptive200',output='results/integrated'):
    destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
    current=Path(current);roots=[Path(p) for p in OLD_ROOTS]+[current]
    plan_entries=[];registry=[];retained=[];tables=[];batches=[];source_hashes=[]
    for root in roots:
        plan=json.loads((root/'plan.json').read_text())
        old=root.name in ['screen32','variables32']
        records=(pd.read_csv(root/'all_specifications.csv').to_dict('records') if old else read_ledger(root))
        dispositions={r['run_id']:r for r in records}
        retained_count=0
        source_hashes.append({'path':str(root/'plan.json'),'sha256':file_hash(root/'plan.json')})
        for entry in plan['entries']:
            run=entry['run_id'];record=dispositions.get(run,{})
            folder=root/('runs' if old else 'accepted')/run
            if isinstance(entry['config'],dict):config=entry['config']
            else:config=AnalysisConfig.load(root/'configs'/f'{run}.json').to_dict()
            assert AnalysisConfig.from_dict(config).fingerprint==entry['config_id']
            plan_entries.append({'suite':root.name,'run_id':run,'config_id':entry['config_id'],'config':config})
            gate=False;flags=None
            if (folder/'estimates.csv').exists():
                table=pd.read_csv(folder/'estimates.csv');flags=evaluate_pattern(table,CRITERIA)
                gate=all(screen_gate(table,year,AnalysisConfig.from_dict(config)) is None for year in [2020,2021,2023])
                if not old:assert gate,'Previously retained case must pass the current gate'
            registry.append({'suite':root.name,'run_id':run,'config_id':entry['config_id'],
                'status':record.get('status','pending'),'current_gate_passed':gate,
                'reason':record.get('reason',''),'source_folder':str(folder) if gate else '',
                'detailed_in_integrated_document':gate})
            if not gate:continue
            retained_count+=1
            retained.append({'suite':root.name,'run_id':run,'config_id':entry['config_id'],
                'status':'accepted','pattern':flags,'source_folder':str(folder),
                'source_estimates_sha256':file_hash(folder/'estimates.csv')})
            table['suite']=root.name;table['run_id']=run;table['config_id']=entry['config_id'];tables.append(table)
            target=destination/'accepted'/run;target.mkdir(parents=True,exist_ok=True)
            for name in ['config.json','estimates.csv','endpoints.csv','audit.json','preprocessing.json',
                         'pattern_flags.json','status.json','event_study.png']:
                path=folder/name
                if path.exists():shutil.copy2(path,target/name)
        batches.append({'suite':root.name,'planned':len(plan['entries']),'attempted':len(records),
            'current_gate_passed':retained_count,
            'rejected':sum(r['status']=='rejected' for r in records),
            'failed':sum(r['status']=='failed' for r in records),
            'historical_fully_completed':sum(r['status']=='completed' for r in records) if old else 0})
    assert len({e['run_id'] for e in plan_entries})==len(plan_entries)
    assert len({e['config_id'] for e in plan_entries})==len(plan_entries)
    count=len(plan_entries);family=count*len(PRIMARY)*len(YEARS)
    data=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame()
    if len(data):
        mask=data.effect.isin(PRIMARY)
        data.loc[mask,'p_bonferroni_cumulative_budget']=np.minimum(1,data.loc[mask,'p_value']*family)
        data.to_csv(destination/'accepted_estimates.csv',index=False)
    pd.DataFrame(registry).to_csv(destination/'all_specifications_audit.csv',index=False)
    pd.DataFrame(batches).to_csv(destination/'batch_summary.csv',index=False)
    (destination/'retained_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in retained))
    write_json(destination/'plan.json',{'roots':[str(r) for r in roots],'source_hashes':source_hashes,
        'planned':count,'primary_effects':PRIMARY,'years':YEARS,'family_tests':family,
        'detailed_save_gate':CRITERIA.to_dict(),'entries':plan_entries})
    summary={'planned':count,'attempted':sum(b['attempted'] for b in batches),
        'retained_cases':len(retained),'batches':batches,'family_tests':family,
        'nominal_pattern_matches':[r['run_id'] for r in retained if r['pattern']['pattern_match']],
        'nominal_preferred_matches':[r['run_id'] for r in retained if r['pattern']['preferred_match']],
        'verified_primary_rows':len(retained)*20,
        'note':'Only the current gate-passing cases are described. Original 64 results are re-evaluated with the current total-only 2023 decline criterion; historical files unchanged.'}
    write_json(destination/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',default='results/adaptive200');p.add_argument('--output',default='results/integrated');args=p.parse_args()
    print(json.dumps(integrate(args.current,args.output),ensure_ascii=False))
