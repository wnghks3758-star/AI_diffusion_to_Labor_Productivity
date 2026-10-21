"""Industry-held-out nuisance fits and disk-backed counterfactual predictions."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import tempfile
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, RidgeCV
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor
from .io import input_key, write_json, versions

def feature_names(frame,configured,config):
    dummies=[c for c in frame if c.startswith('ind_dummy')] if config.include_industry_dummies else []
    return list(dict.fromkeys(configured+dummies))

def make_folds(frame,config,repetition=0,n_splits=None):
    group_info=frame[['ind_code','ind_refined']].drop_duplicates()
    if group_info.ind_code.duplicated().any():raise ValueError('Industry has multiple strata')
    requested=n_splits or config.n_splits
    actual=min(requested,int(group_info.groupby('ind_refined').ind_code.nunique().min()))
    if actual<2:raise ValueError('Each industry stratum must contain >=2 industries; change dummy rules')
    cv=StratifiedGroupKFold(n_splits=actual,shuffle=True,random_state=config.seed+repetition)
    folds=list(cv.split(frame,frame.ind_refined,frame.ind_code))
    for train,test in folds:
        assert not set(frame.iloc[train].ind_code)&set(frame.iloc[test].ind_code)
    return folds

def gps_transform(columns,config):
    continuous=[c for c in columns if not c.startswith('ind_dummy')]
    dummies=[c for c in columns if c.startswith('ind_dummy')]
    if config.gps_model=='linear':return StandardScaler()
    parts=[]
    if continuous:
        parts.append(('continuous',SplineTransformer(n_knots=config.gps_n_knots,degree=config.gps_degree,
            include_bias=False,extrapolation='linear'),continuous))
    if dummies:parts.append(('dummies','passthrough',dummies))
    return Pipeline([('features',ColumnTransformer(parts)),('scaler',StandardScaler())])

def _gps_fold(X,y,train,test,config):
    transform=gps_transform(list(X.columns),config)
    z_train=transform.fit_transform(X.iloc[train]);z_test=transform.transform(X.iloc[test])
    mean=RidgeCV(alphas=config.ridge_alphas,gcv_mode='svd').fit(z_train,y[train])
    # The variance target is formed inside the outer training sample only.
    inner=KFold(n_splits=min(config.inner_splits,len(train)),shuffle=True,random_state=config.seed)
    oof=cross_val_predict(Ridge(alpha=mean.alpha_,solver='svd'),z_train,y[train],cv=inner,n_jobs=1)
    eps=1e-6
    residual_squared=(y[train]-oof)**2
    target=np.log(residual_squared+eps)
    variance=RidgeCV(alphas=config.ridge_alphas,gcv_mode='svd').fit(z_train,target)
    smearing=np.mean(np.exp(np.clip(target-variance.predict(z_train),-50,50)))
    floor=max(float(np.percentile(residual_squared,1)),config.gps_variance_floor)
    sigma2=np.exp(np.clip(variance.predict(z_test),-50,50))*smearing-eps
    return mean.predict(z_test),np.sqrt(np.maximum(sigma2,floor)),{'mean_alpha':mean.alpha_,'variance_alpha':variance.alpha_}

def fit_gps(frame,config):
    columns=feature_names(frame,config.gps_features,config)
    if not columns:raise ValueError('GPS must have at least one covariate')
    if config.gps_level=='industry':
        if (frame.groupby('ind_code')[columns].nunique(dropna=False)>1).any().any():
            raise ValueError('GPS industry covariates vary within industry; select gps_level="firm"')
        training=frame.groupby('ind_code',as_index=False).first()
    else:training=frame.copy().reset_index(drop=True)
    X=training[columns];y=training[config.dose].to_numpy(float)
    settings={k:getattr(config,k) for k in ['gps_level','gps_model','gps_n_knots','gps_degree',
        'ridge_alphas','gps_variance_floor','n_splits','n_repeats','inner_splits','seed']}
    settings['versions']=versions()
    key=input_key(training[['ind_code','ind_refined',config.dose]+columns],settings)
    path=Path(config.cache_dir)/'gps'/f'{key}.joblib'
    hit=path.exists()
    if hit:predicted=joblib.load(path)
    else:
        means=np.empty((config.n_repeats,len(training)));scales=np.empty_like(means);fold_audit=[]
        with threadpool_limits(limits=1):
            for repetition in range(config.n_repeats):
                folds=make_folds(training,config,repetition)
                for train,test in folds:
                    mean,sigma,audit=_gps_fold(X,y,train,test,config)
                    means[repetition,test]=mean;scales[repetition,test]=sigma
                    fold_audit.append({'repetition':repetition,'train_rows':len(train),'test_rows':len(test),**audit})
        predicted={'mean':np.median(means,axis=0),'sigma':np.median(scales,axis=0),'folds':fold_audit}
        path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(predicted,path)
    if config.gps_level=='industry':
        lookup=pd.Series(np.arange(len(training)),index=training.ind_code)
        mapping=frame.ind_code.map(lookup).to_numpy(int)
    else:mapping=np.arange(len(frame))
    mean=predicted['mean'][mapping];sigma=predicted['sigma'][mapping]
    if not np.isfinite(mean).all() or not np.isfinite(sigma).all():raise ValueError('Invalid GPS prediction')
    return mean,sigma,{'cache_hit':hit,'cache_key':key,'n_training_rows':len(training),
        'features':columns,'folds':predicted['folds'],'oof_mse':float(np.mean((y-predicted['mean'])**2))}

def mu_estimator(config):
    if config.mu_model=='xgboost':
        parameters={'objective':'reg:squarederror','random_state':config.seed,'n_jobs':config.model_threads,**config.mu_params}
        return XGBRegressor(**parameters)
    alpha=float(config.mu_params.get('alpha',1.0))
    if config.mu_model=='ridge':return Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=alpha,solver='svd'))])
    return Pipeline([('spline',SplineTransformer(n_knots=int(config.mu_params.get('n_knots',3)),
        degree=int(config.mu_params.get('degree',3)),include_bias=False,extrapolation='linear')),
        ('scale',StandardScaler()),('ridge',Ridge(alpha=alpha,solver='svd'))])

def _tune_mu(X,y,frame,config):
    estimator=mu_estimator(config)
    if not config.mu_search_n_iter or not config.mu_search_space:return {},None
    if config.mu_model!='xgboost':
        space={'ridge__alpha':config.ridge_alphas}
    else:space=config.mu_search_space
    folds=make_folds(frame,config,n_splits=config.inner_splits)
    candidates=int(np.prod([len(v) for v in space.values()]))
    search=RandomizedSearchCV(estimator,space,n_iter=min(config.mu_search_n_iter,candidates),
        cv=folds,scoring='neg_mean_squared_error',n_jobs=config.model_jobs,
        random_state=config.seed,refit=False,pre_dispatch=config.model_jobs)
    with joblib.parallel_config(backend='threading'),threadpool_limits(limits=1):search.fit(X,y)
    return search.best_params_,float(search.best_score_)

def fit_mu(frame,config):
    columns=feature_names(frame,config.mu_features,config)+[config.dose]
    X=frame[columns].to_numpy(float);y=frame.Y_diff.to_numpy(float)
    settings={k:getattr(config,k) for k in ['mu_model','mu_params','mu_search_space','mu_search_n_iter',
        'n_splits','n_repeats','inner_splits','seed','tuning_scope']}
    settings['versions']=versions()
    key=input_key(frame[['ind_code','ind_refined','Y_diff']+columns],settings)
    folder=Path(config.cache_dir)/'mu'/key;manifest_path=folder/'manifest.json'
    hit=manifest_path.exists()
    if hit:
        manifest=json.loads(manifest_path.read_text())
        if not all((folder/entry['file']).exists() for entry in manifest['models']):hit=False
    if not hit:
        folder.mkdir(parents=True,exist_ok=True)
        best,cv_score=_tune_mu(X,y,frame,config) if config.tuning_scope=='global_cv' else ({},None)
        tasks=[(rep,j,train,test) for rep in range(config.n_repeats)
            for j,(train,test) in enumerate(make_folds(frame,config,rep))]
        def fit_task(task):
            rep,j,train,test=task
            params,score=(best,cv_score)
            if config.tuning_scope=='nested':params,score=_tune_mu(X[train],y[train],frame.iloc[train].reset_index(drop=True),config)
            model=mu_estimator(config);model.set_params(**params);model.fit(X[train],y[train])
            filename=f'rep{rep:03d}_fold{j:02d}.joblib'
            joblib.dump({'model':model,'test':test,'train':train},folder/filename)
            return {'file':filename,'repetition':rep,'fold':j,'train_rows':len(train),
                'test_rows':len(test),'best_params':params,'cv_score':score}
        with threadpool_limits(limits=1):
            # Nested searches already parallelize; avoid nested thread pools.
            jobs=1 if config.tuning_scope=='nested' else config.model_jobs
            with ThreadPoolExecutor(max_workers=jobs) as executor:entries=list(executor.map(fit_task,tasks))
        manifest={'models':entries,'features':columns,'cache_key':key}
        write_json(manifest_path,manifest)
    return folder,manifest,{'cache_hit':hit,'cache_key':key,'features':columns,'models':manifest['models']}

def tree_ranges(model,d_values,feature_index):
    if not hasattr(model,'get_booster'):return [(j,j+1) for j in range(len(d_values))]
    thresholds=[]
    for tree in model.get_booster().get_dump(dump_format='json'):
        nodes=[json.loads(tree)]
        while nodes:
            node=nodes.pop()
            if 'leaf' in node:continue
            if node['split']==f'f{feature_index}':thresholds.append(node['split_condition'])
            nodes.extend(node['children'])
    bins=np.searchsorted(np.unique(np.asarray(thresholds,dtype=np.float32)),d_values.astype(np.float32),side='right')
    edges=np.r_[0,np.flatnonzero(np.diff(bins))+1,len(d_values)]
    return list(zip(edges[:-1],edges[1:]))

@contextmanager
def predict_mu(frame,centers,config,folder,manifest):
    """Median across repetitions, stored on disk; one fitted fold resident at a time."""
    centers=np.asarray(centers,float)
    temporary=Path(tempfile.mkdtemp(prefix='mu_predictions_',dir=Path(config.cache_dir)))
    shape=(config.n_repeats,len(centers),len(frame))
    store=np.memmap(temporary/'repeats.dat',mode='w+',dtype='float64',shape=shape)
    store[:]=np.nan
    X=frame[manifest['features']].to_numpy(float)
    predicted_rows=0
    try:
        with threadpool_limits(limits=1):
            for entry in manifest['models']:
                bundle=joblib.load(folder/entry['file']);model=bundle['model'];test=bundle['test']
                ranges=tree_ranges(model,centers,X.shape[1]-1)
                for first,last in ranges:
                    values=X[test].copy();values[:,-1]=centers[first]
                    prediction=np.empty(len(test))
                    for start in range(0,len(test),config.max_prediction_rows):
                        stop=min(start+config.max_prediction_rows,len(test))
                        prediction[start:stop]=model.predict(values[start:stop])
                    store[entry['repetition'],first:last,test]=prediction[:,None]
                    predicted_rows+=len(test)
                del bundle,model
            store.flush()
        class Predictions:
            audit={'disk_prediction_bytes':int(np.prod(shape))*8,'n_centers':len(centers),
                'predicted_rows':predicted_rows,'original_prediction_rows':int(np.prod(shape))}
            def batches(self):
                for start in range(0,len(centers),config.prediction_batch_points):
                    stop=min(start+config.prediction_batch_points,len(centers))
                    median=np.median(np.asarray(store[:,start:stop,:]),axis=0)
                    if not np.isfinite(median).all():raise ValueError('Incomplete Mu cross-fit predictions')
                    yield start,stop,median
        yield Predictions()
    finally:
        del store
        shutil.rmtree(temporary,ignore_errors=True)
