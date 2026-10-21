"""Freeze local neighbours of outcome-selected anchors before new estimation.

Selection is explicitly exploratory. No numerical estimation code is changed.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .config import AnalysisConfig, INDUSTRY_CONTROLS, FIRM_CONTROLS
from .criteria import evaluate_pattern
from .io import file_hash, write_json
from .random_experiments import CRITERIA, PRIMARY, YEARS, numerical_source_hashes

OLD_ROOTS = ['results/screen32', 'results/variables32', 'results/random200',
    'results/random201_400', 'results/random401_600', 'results/random601_800']
CORE = ['pre_all_nonsignificant', 'initial_significant_decline', 'recovery',
        'sme_crosses_positive_earlier', 'moderation_negative_significant']
RANKING = ('Gate-passing cases only; descending core flags (5), descending nondecreasing '
    'post-year transitions (6), descending positive SME2025 indicator, '
    'ascending nominal SME2025 p, ascending suite/run ID. Select ten; twenty neighbours each.')


def collect_history(roots=OLD_ROOTS):
    """Read immutable source plans and re-evaluate the same current save criterion."""
    anchors=[];seen=set();sources=[]
    for source in roots:
        root=Path(source);plan=json.loads((root/'plan.json').read_text())
        sources.append({'root':str(root), 'plan_sha256':file_hash(root/'plan.json')})
        for entry in plan['entries']:seen.add(entry['config_id'])
        folder_name='runs' if (root/'runs').exists() else 'accepted'
        for folder in sorted((root/folder_name).iterdir()):
            if not (folder/'estimates.csv').exists():continue
            table=pd.read_csv(folder/'estimates.csv');flags=evaluate_pattern(table,CRITERIA)
            if not (flags['pre_all_nonsignificant'] and flags['initial_significant_decline']):continue
            indexed=table.set_index(['effect','year'])
            transitions=sum(int(indexed.loc[(e,y+1),'estimate']>=indexed.loc[(e,y),'estimate'])
                for e in ['total','sme','large'] for y in [2023,2024])
            anchors.append({'suite':root.name,'run_id':folder.name,'source_folder':str(folder),
                'source_estimates_sha256':file_hash(folder/'estimates.csv'),
                'source_config_sha256':file_hash(folder/'config.json'),
                **flags,'core_pass':sum(bool(flags[k]) for k in CORE),
                'recovery_transitions':transitions,
                'sme2025_positive':bool(indexed.loc[('sme',2025),'estimate']>0),
                'sme2025_p':float(indexed.loc[('sme',2025),'p_value'])})
    ranked=pd.DataFrame(anchors).sort_values(
        ['core_pass','recovery_transitions','sme2025_positive','sme2025_p','suite','run_id'],
        ascending=[False,False,False,True,True,True]).reset_index(drop=True)
    ranked.insert(0,'anchor_rank',range(1,len(ranked)+1))
    return ranked,seen,sources


def local_operations(config):
    """One coordinate = one variable toggle or one scalar/active model parameter."""
    ops={k:[] for k in ['gps_variable','mu_variable','support','model','fold','industry']}
    for key,pool,family in [('gps_features',INDUSTRY_CONTROLS,'gps_variable'),
            ('mu_features',['large_dummy','small_dummy']+FIRM_CONTROLS+INDUSTRY_CONTROLS,'mu_variable')]:
        for feature in pool:
            if feature in config[key] and len(config[key])<=2:continue
            new=[v for v in pool if (v in config[key]) != (v==feature)]
            ops[family].append((key,new))
    def scalar(family,key,values):
        old=config[key] if not key.startswith('mu_params.') else config['mu_params'][key.split('.')[1]]
        for value in values:
            if value!=old:ops[family].append((key,value))
    for key,step,low,high in [('ess',1.,2.,12.),('min_ess_prime',1.,2.,10.),
            ('trim_threshold',.025,.025,.25),('density_floor',.001,.001,.015),
            ('bandwidth_scale',.05,.25,.8)]:
        scalar('support',key,[round(max(low,config[key]-step),6),round(min(high,config[key]+step),6)])
    scalar('fold','n_splits',[max(2,config['n_splits']-1),min(5,config['n_splits']+1)])
    scalar('fold','n_repeats',[max(2,config['n_repeats']-1),min(5,config['n_repeats']+1)])
    if config['gps_model']=='spline':
        scalar('model','gps_n_knots',[max(3,config['gps_n_knots']-1),min(5,config['gps_n_knots']+1)])
        scalar('model','gps_degree',[max(1,config['gps_degree']-1),min(3,config['gps_degree']+1)])
    if config['mu_model']=='xgboost':
        for key,step,low,high in [('n_estimators',40,40,320),('max_depth',1,1,5),
                ('learning_rate',.01,.01,.15),('subsample',.05,.5,1.),('colsample_bytree',.05,.5,1.)]:
            old=config['mu_params'][key]
            values=[max(low,old-step),min(high,old+step)]
            if isinstance(old,int):values=[int(v) for v in values]
            else:values=[round(v,6) for v in values]
            scalar('model','mu_params.'+key,values)
    else:
        old=config['mu_params']['alpha'];scalar('model','mu_params.alpha',[old/2,old*2])
    if config['industry_scheme']=='refined':
        for key,low in [('industry_min_groups',2),('industry_min_obs',2)]:
            scalar('industry',key,[max(low,config[key]-1),config[key]+1])
    if config['industry_scheme']!='none':
        scalar('industry','include_industry_dummies',[not config['include_industry_dummies']])
    return {key:values for key,values in ops.items() if values}


def neighbours(anchor_config,count,rng,seen):
    anchor=deepcopy(anchor_config)
    # Two old variable-only anchors used 999 draws; all new cases use 1999.
    anchor['n_bootstrap']=1999
    ops=local_operations(anchor);families=list(ops)
    result=[]
    for slot in range(count):
        family=families[slot%len(families)];number=1 if slot%2==0 else 2
        for retry in range(10000):
            # A bounded fold family can have only two unique neighbours. Move
            # to another active family after exhausting proposals; never reuse a case.
            selected_family=family if retry<100 else families[int(rng.integers(len(families)))]
            chosen=[ops[selected_family][int(rng.integers(len(ops[selected_family])))]]
            if number==2:
                other=families[int(rng.integers(len(families)))];chosen.append(ops[other][int(rng.integers(len(ops[other])))])
            if len({key for key,value in chosen})!=number:continue
            config=deepcopy(anchor);changes=[]
            for key,value in chosen:
                if key.startswith('mu_params.'):
                    param=key.split('.')[1];old=config['mu_params'][param];config['mu_params'][param]=value
                else:old=config[key];config[key]=value
                changes.append({'coordinate':key,'before':old,'after':value})
            candidate=AnalysisConfig.from_dict(config)
            if candidate.fingerprint in seen:continue
            seen.add(candidate.fingerprint)
            result.append({'config':candidate.to_dict(),'config_id':candidate.fingerprint,
                'changes':changes,'local_coordinate_count':number,'primary_mutation_family':selected_family,
                'bootstrap_normalized_from':anchor_config['n_bootstrap']})
            break
        else:raise RuntimeError('Could not produce a unique local neighbour')
    return result


def make_adaptive_plan(root='results/adaptive200',seed=20261022):
    root=Path(root)
    if (root/'plan.json').exists():raise FileExistsError('Frozen plan exists; resume it')
    ranked,seen,sources=collect_history();selected=ranked.head(10)
    if len(selected)!=10:raise ValueError('Ten eligible anchors required')
    root.mkdir(parents=True,exist_ok=True)
    ranked.to_csv(root/'anchor_pool.csv',index=False);selected.to_csv(root/'selected_anchors.csv',index=False)
    rng=np.random.default_rng(seed);groups=[]
    for row in selected.to_dict('records'):
        folder=Path(row['source_folder']);original=json.loads((folder/'config.json').read_text())
        group=neighbours(original,20,rng,seen)
        for entry in group:
            entry.update(anchor_rank=row['anchor_rank'],anchor_run_id=row['run_id'],anchor_suite=row['suite'],
                anchor_config_id=AnalysisConfig.from_dict(original).fingerprint)
        groups.append(group)
    # Interleave anchors, rather than spending the stop budget on one parent first.
    entries=[]
    for round_number in range(20):
        for group in groups:
            entry=group[round_number];run=f'local_{801+len(entries):03d}'
            entry['run_id']=run;entry['config']['label']=run;entries.append(entry)
    previous_root=Path(OLD_ROOTS[-1]);previous=json.loads((previous_root/'plan.json').read_text())
    assert previous['numerical_source_hashes']==numerical_source_hashes()
    assert previous['data_sha256']==file_hash(AnalysisConfig().data_path)
    plan={**{k:v for k,v in previous.items() if k not in ['entries','domains','previous_plan']},
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'experiment_type':'outcome_guided_frozen_local_screen','max_attempts':200,
        'generator_seed':seed,'generator_offset':800,'all_random_planned_attempts':1000,
        'batch_number':5,'prior_specifications':864,'prior_plan_roots':OLD_ROOTS[2:],
        'previous_plan':{'path':str(previous_root/'plan.json'),'sha256':file_hash(previous_root/'plan.json'),
            'ledger_sha256':file_hash(previous_root/'all_attempts.jsonl'),'attempted':200},
        'anchor_selection':RANKING,'anchor_pool_count':len(ranked),'anchor_count':10,
        'anchor_pool_sha256':file_hash(root/'anchor_pool.csv'),
        'selected_anchors_sha256':file_hash(root/'selected_anchors.csv'),
        'historical_sources':sources,'adaptive_source_sha256':file_hash(__file__),
        'domains':{'local': 'One or two active coordinates per normalized anchor; ten single and ten paired neighbours per anchor.',
            'variable':'One feature added or removed per variable coordinate; GPS pool13, Mu pool18; minimum2.',
            'support':'ESS +/-1, trim +/-.025, density +/-.001, bandwidth +/-.05 within explicit bounds.',
            'model':'GPS knot/degree +/-1; same Mu type, trees +/-40, depth +/-1, LR +/-.01, row/column +/-.05; ridge alpha /2 or *2.',
            'fold':'requested folds or repeats +/-1, within2..5',
            'industry':'refined group/observation threshold +/-1, or dummy toggle; scheme unchanged.'},
        'fixed':'Same anchor model types, timing, dose, outcome, size, baseline, alpha, seed, gap, finite difference, sample reference; all new bootstrap1999. Old variable anchors had999.',
        'inference_note':'Outcome-guided exploratory local search, frozen before new outcomes; Bonferroni is reported but cannot undo selection or establish causal identification.',
        'entries':entries}
    write_json(root/'plan.json',plan);write_json(root/'pattern_criteria.json',CRITERIA.to_dict())
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/adaptive200');args=parser.parse_args()
    plan=make_adaptive_plan(args.output)
    print(json.dumps({k:v for k,v in plan.items() if k not in ['entries','numerical_source_hashes']},ensure_ascii=False))
