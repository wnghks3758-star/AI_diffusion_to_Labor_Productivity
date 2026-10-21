"""Check old and streamed aggregation using identical nuisance predictions.

The notebook is loaded only by this regression check, never by the Python package.
This does not claim that the improved GPS learner reproduces the old GPS learner.
"""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dml_acr.config import AnalysisConfig
from dml_acr.preprocessing import load_data,prepare_panel,prepare_year,observed_grid
from dml_acr.models import fit_gps,fit_mu,predict_mu
from dml_acr.estimation import estimate_year
from dml_acr.io import write_json

root=Path(__file__).resolve().parents[1]
import os
os.chdir(root)
config=AnalysisConfig.load('configs/smoke.json')
raw,_=load_data(config.data_path,config.cache_dir)
data,_=prepare_panel(raw,config)
notebook=json.loads((root/'f4_7_DML_ACR_Dart_size_mean_difference_chunked.ipynb').read_text())
namespace={}
exec(compile(''.join(notebook['cells'][1]['source']),'original_notebook_definitions','exec'),namespace)
namespace['plt'].show=lambda:None
year=2023;frame=prepare_year(data,year,config)
mean,sigma,_=fit_gps(frame,config)
namespace['precompute_gps']=lambda *args,**kwargs:(mean.copy(),sigma.copy())

def predict_reference(frame,low_values,**kwargs):
    folder,manifest,_=fit_mu(frame,config)
    values=np.unique(low_values)
    with predict_mu(frame,values,config,folder,manifest) as prediction:
        combined=np.concatenate([batch for _,_,batch in prediction.batches()])
    return {float(d):combined[i] for i,d in enumerate(values)}

namespace['precompute_mu']=predict_reference
records=[]
for policy in ['strict','bridge']:
    config.gap_policy=policy
    new,_,_,_=estimate_year(frame,year,config)
    grid,bw=observed_grid(frame,config)
    old=namespace['run_year_analysis'](data,year,bw,grid,config.dose,
        base_year=config.base_year,n_bootstrap=config.n_bootstrap,n_splits=config.n_splits,
        n_repeats=config.n_repeats,min_ess=config.ess,trim_threshold=config.trim_threshold,
        p_floor=config.density_floor,diff_step=config.diff_step,bootstrap_unit=config.bootstrap_unit,
        options={'memory_mode':'event_only','verbose':False,'search_n_jobs':2,'crossfit_n_jobs':2},gap_policy=policy)
    old['gap_policy']=policy
    summary=namespace['summarize_event_study']([old],include_baseline=False).iloc[0]
    for effect,point_col,se_col in [('total','atet_o','atet_se'),('sme','sme_atet_o','sme_se'),
        ('large','large_atet_o','large_se'),('moderation','moderation_o','mod_se')]:
        row=new.loc[new.effect.eq(effect)].iloc[0]
        np.testing.assert_allclose([row.estimate,row.se],[summary[point_col],summary[se_col]],rtol=1e-10,atol=1e-10)
        records.append({'policy':policy,'effect':effect,'point_difference':float(row.estimate-summary[point_col]),
                        'se_difference':float(row.se-summary[se_col])})
report={'status':'PASS','year':year,'identical_nuisances':True,'comparisons':records,
        'scope':'Streamed annual aggregation versus notebook event_only with the same nuisance predictions.'}
write_json('validation/modular_notebook_parity.json',report)
print(json.dumps(report,indent=2))
