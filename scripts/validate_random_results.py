"""Independently check retained cases, trial accounting and optional-stop behavior."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dml_acr.config import AnalysisConfig
from dml_acr.criteria import PatternCriteria, evaluate_pattern
from dml_acr.io import file_hash,write_json
from dml_acr.random_experiments import numerical_source_hashes


def validate(root,require_finished=True,audit_output=None):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());summary=json.loads((root/'summary.json').read_text())
    records=[json.loads(line) for line in (root/'all_attempts.jsonl').read_text().splitlines() if line]
    assert len(records)==summary['attempted']<=plan['max_attempts']
    assert [r['run_id'] for r in records]==[e['run_id'] for e in plan['entries'][:len(records)]]
    assert [r['attempt_number'] for r in records]==list(range(1,len(records)+1))
    assert plan['numerical_source_hashes']==numerical_source_hashes()
    assert file_hash(AnalysisConfig().data_path)==plan['data_sha256']
    for entry in plan['entries']:
        assert AnalysisConfig.from_dict(entry['config']).fingerprint==entry['config_id']
    accepted=[r['run_id'] for r in records if r['status']=='accepted']
    folders=sorted(p.name for p in (root/'accepted').iterdir() if p.is_dir()) if (root/'accepted').exists() else []
    assert folders==sorted(accepted)
    checks=[];matches=[]
    primary=['total','sme','large','moderation']
    combined=pd.read_csv(root/'accepted_estimates.csv') if accepted else pd.DataFrame()
    for run in accepted:
        folder=root/'accepted'/run;config=AnalysisConfig.load(folder/'config.json')
        table=pd.read_csv(folder/'estimates.csv');d=table.loc[table.effect.isin(primary)].copy()
        assert len(d)==20 and not d.duplicated(['year','effect']).any()
        pre=d.loc[d.year.isin([2020,2021])]
        assert len(pre)==8 and pre.p_value.ge(.05).all()
        assert np.isfinite(pre[['estimate','p_value']].to_numpy()).all()
        initial=d.loc[d.year.eq(2023)&d.effect.eq('total')].iloc[0]
        assert initial.estimate<0 and initial.p_value<.05
        finite=d.loc[np.isfinite(d[['estimate','p_value','se']].to_numpy()).all(axis=1)]
        assert finite.se.gt(0).all() and finite.p_value.between(0,1).all()
        np.testing.assert_allclose(finite.p_value,2*norm.sf(np.abs(finite.estimate/finite.se)),rtol=1e-10,atol=1e-14)
        np.testing.assert_allclose(d.atet_estimate,d.estimate*config.diff_step,rtol=1e-10,atol=1e-14,equal_nan=True)
        pivot=d.pivot(index='year',columns='effect',values='estimate')
        np.testing.assert_allclose(pivot.moderation,pivot.large-pivot.sme,rtol=1e-10,atol=1e-14,equal_nan=True)
        flags=evaluate_pattern(table,PatternCriteria(**plan['pattern_criteria']))
        saved=json.loads((folder/'pattern_flags.json').read_text())
        assert flags==saved
        if flags['pattern_match']:matches.append(run)
        sub=combined.loc[combined.run_id.eq(run)&combined.effect.isin(primary)].sort_values(['year','effect'])
        ordered=d.sort_values(['year','effect'])
        np.testing.assert_allclose(sub.estimate,ordered.estimate,rtol=1e-10,atol=1e-14,equal_nan=True)
        family=plan['max_attempts']*20;cumulative=(plan['max_attempts']+plan['prior_specifications'])*20
        np.testing.assert_allclose(sub.p_bonferroni_random_budget,np.minimum(1,sub.p_value*family),atol=1e-14,equal_nan=True)
        np.testing.assert_allclose(sub.p_bonferroni_cumulative_budget,np.minimum(1,sub.p_value*cumulative),atol=1e-14,equal_nan=True)
        assert not (folder/'worker.log').exists() and not (folder/'worker_config.json').exists()
        checks.append({'run_id':run,'gate_verified':True,'primary_rows':20,'pattern_match':flags['pattern_match']})
    assert matches==summary['nominal_pattern_matches']
    if matches:
        assert len(matches)==1 and records[-1]['run_id']==matches[0]
        assert summary['stop_reason']=='target_found'
    elif require_finished:
        assert len(records)==plan['max_attempts'] and summary['stop_reason']=='budget_exhausted'
    counts={status:sum(r['status']==status for r in records) for status in ['accepted','rejected','failed']}
    assert sum(counts.values())==len(records)
    assert counts['accepted']==summary['accepted_detailed_cases']
    assert counts['rejected']==summary['rejected'] and counts['failed']==summary['failed']
    if require_finished:
        scratch=Path('work/random_search_scratch')/root.name
        assert not scratch.exists() or not any(scratch.iterdir())
    audit={'passed':True,'attempted':len(records),'statuses':counts,'stop_reason':summary['stop_reason'],
           'accepted_checks':checks,'rejected_detailed_cases_retained':0,
           'data_sha256_verified':True,'numerical_source_hashes_verified':True,
           'verified_primary_rows':len(accepted)*20,'inference_and_multiplicity_checked':True}
    write_json(audit_output or ROOT/'validation/random_search_audit.json',audit)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/random200')
    parser.add_argument('--allow-running',action='store_true');args=parser.parse_args()
    print(json.dumps(validate(args.root,not args.allow_running,
        ROOT/'validation'/f'{Path(args.root).name}_audit.json'),ensure_ascii=False))
