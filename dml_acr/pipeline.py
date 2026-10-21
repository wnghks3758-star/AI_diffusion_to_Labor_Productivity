"""Public single-specification function with year-by-year durable checkpoints."""
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter
import resource
import traceback
import numpy as np
import pandas as pd
from scipy.stats import chi2
from .config import AnalysisConfig
from .preprocessing import load_data,prepare_panel,prepare_year
from .estimation import estimate_year
from .aggregation import draw_joint
from .criteria import evaluate_pattern,holm_adjust
from .visualization import plot_event_study
from .io import write_json,versions

def joint_inference(estimates,joint_results,config):
    primary=['total','sme','large','moderation_means' if config.size_difference=='means' else 'moderation_common']
    clusters=np.unique(np.concatenate([r['clusters'] for r in joint_results]))
    columns=[];points=[];names=[]
    for year,result in zip(config.years,joint_results):
        positions=pd.Index(clusters).get_indexer(result['clusters'])
        for effect in primary:
            j=result['effects'].index(effect);vector=np.zeros(len(clusters));vector[positions]=result['scores'][:,j]
            columns.append(vector);points.append(result['points'][j]);names.append((year,'moderation' if effect.startswith('moderation_') else effect))
    scores=np.column_stack(columns);points=np.array(points)
    draws=draw_joint(points,scores,config);se=np.std(draws,axis=0,ddof=1)
    finite=np.isfinite(points)&np.isfinite(se)&(se>0)
    critical=np.nan
    if finite.any():
        maxima=np.max(np.abs((draws[:,finite]-points[finite])/se[finite]),axis=1)
        critical=float(np.quantile(maxima,1-config.alpha))
        for j,(year,effect) in enumerate(names):
            match=estimates.year.eq(year)&estimates.effect.eq(effect)
            estimates.loc[match,'simultaneous_lower']=points[j]-critical*se[j]
            estimates.loc[match,'simultaneous_upper']=points[j]+critical*se[j]
            estimates.loc[match,'p_max_t_within_spec']=float((1+(maxima>=abs(points[j]/se[j])).sum())/(len(maxima)+1)) if finite[j] else np.nan
    mask=estimates.effect.isin(['total','sme','large','moderation'])
    estimates.loc[mask,'p_holm_within_spec']=holm_adjust(estimates.loc[mask,'p_value'])
    pre=np.array([y<config.base_year for y,_ in names])&finite
    joint_p=np.nan;rank=0
    if pre.any():
        cov=scores[:,pre].T@scores[:,pre];rank=int(np.linalg.matrix_rank(cov))
        statistic=float(points[pre]@np.linalg.pinv(cov)@points[pre])
        if rank:joint_p=float(chi2.sf(statistic,rank))
    return estimates,{'max_t_critical_value':critical,'family_tests':len(names),'pre_joint_wald_p_value':joint_p,
        'pre_joint_covariance_rank':rank,'n_joint_clusters':len(clusters)}

def run_analysis(config,output_dir=None,year_gate=None,pattern_criteria=None):
    if isinstance(config,(str,Path)):config=AnalysisConfig.load(config)
    elif isinstance(config,dict):config=AnalysisConfig.from_dict(config)
    output=Path(output_dir or Path('results')/config.fingerprint);output.mkdir(parents=True,exist_ok=True)
    config.save(output/'config.json')
    started=perf_counter();all_rows=[];all_endpoints=[];audits=[];joint=[]
    def progress(year,stage,detail):
        event={'at_utc':datetime.now(timezone.utc).isoformat(),'year':year,'stage':stage,'detail':detail}
        from .io import jsonable
        import json
        with (output/'progress.jsonl').open('a') as handle:handle.write(json.dumps(jsonable(event))+'\n')
        print(f'[{config.label}] {year}: {stage}',flush=True)
    try:
        raw,data_hash=load_data(config.data_path,config.cache_dir)
        data,preprocessing=prepare_panel(raw,config);del raw
        write_json(output/'preprocessing.json',preprocessing)
        for year in config.years:
            progress(year,'started',{})
            frame=prepare_year(data,year,config)
            table,endpoints,audit,joint_result=estimate_year(frame,year,config,
                on_stage=lambda stage,detail:progress(year,stage,detail))
            all_rows.append(table);all_endpoints.append(endpoints);audits.append(audit);joint.append(joint_result)
            pd.concat(all_rows,ignore_index=True).to_csv(output/'estimates.partial.csv',index=False)
            write_json(output/'audit.partial.json',audits)
            progress(year,'completed',{'elapsed_seconds':perf_counter()-started})
            # Optional staged screening avoids fitting later outcomes after rejection.
            # The caller controls whether to retain or remove this scratch folder.
            if year_gate is not None:
                decision=year_gate(pd.concat(all_rows,ignore_index=True),year,config)
                if decision is not None:
                    status={'status':'rejected','label':config.label,'config_id':config.fingerprint,
                        'data_sha256':data_hash,'elapsed_seconds':perf_counter()-started,
                        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                        'years_evaluated':[int(t.year.iloc[0]) for t in all_rows],
                        'gate':decision,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
                    write_json(output/'status.json',status)
                    return pd.concat(all_rows,ignore_index=True),status
        estimates=pd.concat(all_rows,ignore_index=True)
        estimates,joint_audit=joint_inference(estimates,joint,config)
        estimates.to_csv(output/'estimates.csv',index=False)
        pd.concat(all_endpoints,ignore_index=True).to_csv(output/'endpoints.csv',index=False)
        write_json(output/'audit.json',{'preprocessing':preprocessing,'years':audits,'joint':joint_audit})
        pattern=evaluate_pattern(estimates,pattern_criteria)
        write_json(output/'pattern_flags.json',pattern)
        plot_event_study(estimates,output/'event_study.png',config.base_year,title=config.label+' (exploratory; nominal 95% intervals)')
        if config.save_bootstrap_draws:
            np.savez_compressed(output/'joint_bootstrap.npz',**{str(y):r['draws'] for y,r in zip(config.years,joint)})
        status={'status':'completed','label':config.label,'config_id':config.fingerprint,
            'data_sha256':data_hash,'elapsed_seconds':perf_counter()-started,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'versions':versions(),'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'inference_note':'Exploratory specification search; conditional on fitted nuisances, selected support and averaging weights.',
            'pattern':pattern}
        write_json(output/'status.json',status)
        (output/'estimates.partial.csv').unlink(missing_ok=True);(output/'audit.partial.json').unlink(missing_ok=True)
        return estimates,status
    except Exception as exc:
        write_json(output/'status.json',{'status':'failed','label':config.label,'config_id':config.fingerprint,
            'elapsed_seconds':perf_counter()-started,'error':str(exc),'traceback':traceback.format_exc()})
        raise
