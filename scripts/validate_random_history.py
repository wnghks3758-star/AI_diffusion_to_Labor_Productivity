"""Audit cross-batch uniqueness, immutable history and cumulative corrections."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dml_acr.io import file_hash,write_json


def validate_history(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text())
    previous=plan['previous_plan'];previous_path=Path(previous['path'])
    assert file_hash(previous_path)==previous['sha256']
    assert file_hash(previous_path.parent/'all_attempts.jsonl')==previous['ledger_sha256']
    roots=[Path(p) for p in plan['prior_plan_roots']]+[root]
    records=[];entries=[];tables=[];planned=0
    for folder in roots:
        frozen=json.loads((folder/'plan.json').read_text());planned+=frozen['max_attempts']
        # Check every link in the history, including batches earlier than the last.
        if 'previous_plan' in frozen:
            prior=frozen['previous_plan'];prior_path=Path(prior['path'])
            assert file_hash(prior_path)==prior['sha256']
            assert file_hash(prior_path.parent/'all_attempts.jsonl')==prior['ledger_sha256']
        entries.extend(frozen['entries'])
        assert frozen['numerical_source_hashes']==plan['numerical_source_hashes']
        assert frozen['data_sha256']==plan['data_sha256']
        for line in (folder/'all_attempts.jsonl').read_text().splitlines():
            if line:
                record=json.loads(line);record['global_attempt_number']=frozen.get('generator_offset',0)+record['attempt_number']
                records.append(record)
        if (folder/'accepted_estimates.csv').exists():tables.append(pd.read_csv(folder/'accepted_estimates.csv'))
    assert len({e['config_id'] for e in entries})==len(entries)==planned
    assert len({e['run_id'] for e in entries})==planned
    destination=root/'cumulative';summary=json.loads((destination/'cumulative_summary.json').read_text())
    ledger=pd.read_csv(destination/'all_random_attempts.csv')
    assert len(ledger)==len(records)==summary['random_attempted']
    assert ledger.run_id.tolist()==[r['run_id'] for r in records]
    assert ledger.status.tolist()==[r['status'] for r in records]
    assert ledger.global_attempt_number.tolist()==[r['global_attempt_number'] for r in records]
    assert ledger.global_attempt_number.tolist()==list(range(1,len(records)+1))
    assert summary['unique_configuration_count']==planned
    for status,key in [('accepted','retained'),('rejected','rejected'),('failed','failed')]:
        assert summary[key]==sum(r['status']==status for r in records)
    primary=['total','sme','large','moderation'];rows=0
    if tables:
        expected=pd.concat(tables,ignore_index=True).sort_values(['run_id','year','effect']).reset_index(drop=True)
        combined=pd.read_csv(destination/'all_retained_estimates.csv').sort_values(['run_id','year','effect']).reset_index(drop=True)
        keys=['run_id','year','effect']
        assert not combined.duplicated(keys).any()
        pd.testing.assert_frame_equal(expected[keys],combined[keys])
        for col in ['estimate','p_value','se','ci_lower','ci_upper']:
            np.testing.assert_allclose(expected[col],combined[col],rtol=1e-10,atol=1e-14,equal_nan=True)
        mask=combined.effect.isin(primary);selected=combined.loc[mask]
        random_family=planned*20;all_family=(plan['prior_specifications']+plan['max_attempts'])*20
        np.testing.assert_allclose(selected.p_bonferroni_all_random_budget,
            np.minimum(1,selected.p_value*random_family),atol=1e-14,equal_nan=True)
        np.testing.assert_allclose(selected.p_bonferroni_cumulative_budget,
            np.minimum(1,selected.p_value*all_family),atol=1e-14,equal_nan=True)
        assert summary['all_random_family_tests']==random_family
        assert summary['all_exploration_family_tests']==all_family
        rows=len(selected)
    audit={'passed':True,'random_attempted':len(records),'planned_unique_candidates':planned,
        'retained_cases':summary['retained'],'verified_primary_rows':rows,
        'previous_plan_and_ledger_unchanged':True,'no_duplicate_configurations':True,
        'cumulative_estimates_match_sources':True,'cumulative_corrections_verified':True}
    write_json(ROOT/'validation'/f'{root.name}_history_audit.json',audit)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/random201_400');args=parser.parse_args()
    print(json.dumps(validate_history(args.root),ensure_ascii=False))
