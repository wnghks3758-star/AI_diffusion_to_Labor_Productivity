"""Read once, validate identifiers, define size, trim support and encode industries."""
from pathlib import Path
import numpy as np
import pandas as pd
from .preprocessing_legacy import refine_industry_codes, prepare_year_data, kernel_ess
from .io import file_hash

def load_data(path, cache_dir='work/cache/dml_acr'):
    path=Path(path)
    digest=file_hash(path)
    cache=Path(cache_dir)/'data'/f'{digest}.pkl'
    if cache.exists(): return pd.read_pickle(cache),digest
    data=pd.read_excel(path)
    cache.parent.mkdir(parents=True,exist_ok=True)
    data.to_pickle(cache)
    return data,digest

def cluster_bandwidth(frame,config):
    d=frame.groupby('ind_code')[config.dose].first().dropna().to_numpy()
    if len(d)<2: raise ValueError('Need at least two industries')
    sd=np.std(d);iqr=np.subtract(*np.percentile(d,[75,25]))
    scale=min(sd,iqr/1.34) if iqr>0 else sd
    return max(2.34*scale*len(d)**(-0.25)*config.bandwidth_scale,config.min_bandwidth)

def add_size(data,config):
    sales=data.loc[data.Year<=config.base_year].groupby(['ind_code','firm_code']).sale.mean().reset_index()
    sales['count']=sales.groupby('ind_code').firm_code.transform('nunique')
    sales['threshold']=sales.groupby('ind_code').sale.transform(lambda x:x.quantile(config.size_quantile))
    sales['large_firm']=np.where(sales['count']<=config.small_industry_max_firms,-1,
                                (sales.sale>=sales.threshold).astype(int))
    sales.loc[~np.isfinite(sales.sale),'large_firm']=np.nan
    sales['large_dummy']=(sales.large_firm==1).astype(int)
    sales['small_dummy']=(sales.large_firm==0).astype(int)
    return data.join(sales.set_index('firm_code')[['large_firm','large_dummy','small_dummy']],on='firm_code').dropna(subset=['large_firm'])

def prepare_panel(raw,config):
    data=raw.copy()
    if config.outcome != 'ln_firm_prod':data['ln_firm_prod']=data[config.outcome]
    required=['firm_code','Year','ind_code','sale','ln_firm_prod',config.dose]
    generated={'large_dummy','small_dummy'}
    required+=sorted((set(config.gps_features)|set(config.mu_features))-generated)
    missing=sorted(set(required)-set(data.columns))
    if missing:raise ValueError(f'Missing input columns: {missing}')
    if data[['firm_code','Year','ind_code']].isna().any().any():raise ValueError('Missing identifiers')
    if data.duplicated(['firm_code','Year']).any():raise ValueError('Duplicate firm-year')
    if (data.groupby('firm_code').ind_code.nunique()>1).any():raise ValueError('Firm changes industry')
    wanted=set(config.years)|{config.base_year}
    counts=data.loc[data.Year.isin(wanted)].groupby('firm_code').Year.nunique()
    eligible=counts[counts.eq(len(wanted))].index
    data=data.loc[data.firm_code.isin(eligible)].copy()
    if (data.groupby('ind_code')[config.dose].nunique(dropna=False)>1).any():
        raise ValueError('This static-exposure implementation requires industry-constant dose across years')
    if not np.isfinite(data[config.dose]).all():raise ValueError('Non-finite dose')
    if config.dose_transform=='log_percent':
        if (data[config.dose]<=0).any():raise ValueError('Log dose must be positive')
        data[config.dose]=np.log(100*data[config.dose])
    if config.covariate_time=='baseline':
        columns=sorted((set(config.gps_features)|set(config.mu_features))-generated)
        base=data.loc[data.Year.eq(config.base_year)].set_index('firm_code')
        for column in columns:data[column]=data.firm_code.map(base[column])
    numeric=sorted(set(required)-{'firm_code','Year','ind_code'})
    finite=np.isfinite(data[numeric].to_numpy(dtype=float)).all(axis=1)
    invalid_firms=data.loc[~finite,'firm_code'].unique()
    data=data.loc[~data.firm_code.isin(invalid_firms)].copy()
    data=add_size(data,config)
    # Keep all available pre-period rows for size classification, then select requested years.
    data=data.loc[data.Year.isin(wanted)].copy()
    lower_candidates=np.sort(data[config.dose].unique())
    annual=[]
    for year in config.years:
        frame=prepare_year_data(data,year,config.base_year)
        annual.append((frame.groupby('ind_code')[config.dose].first().to_numpy(),cluster_bandwidth(frame,config)))
    lower=next((float(d) for d in lower_candidates if d<=config.max_dose and
        all(kernel_ess(values,d,bw)>=config.min_ess_prime for values,bw in annual)),None)
    if lower is None:raise ValueError('No common lower-dose support across all years')
    data=data.loc[data[config.dose].between(lower,config.max_dose)].copy()
    data=refine_industry_codes(data,min_nunique=config.industry_min_groups,min_obs=config.industry_min_obs)
    data=data.drop(columns=[c for c in data if c.startswith('ind_dummy')])
    if config.industry_scheme=='major':data['ind_refined']=data['대분류'].fillna('Other')
    elif config.industry_scheme=='none':data['ind_refined']='All'
    data['ind_refined']=data.ind_refined.fillna('Other')
    if config.include_industry_dummies and config.industry_scheme!='none':
        data=pd.concat([data,pd.get_dummies(data.ind_refined,prefix='ind_dummy',drop_first=True,dtype=int)],axis=1)
    data=data.sort_values(['Year','firm_code']).reset_index(drop=True)
    audit={'input_rows':len(raw),'balanced_firms_before_missing':len(eligible),
        'firms_dropped_for_missing':len(invalid_firms),'analysis_firms':int(data.firm_code.nunique()),
        'analysis_industries':int(data.ind_code.nunique()),'unclassified_firms':int(data.loc[data.large_firm.eq(-1)].firm_code.nunique()),
        'common_lower_dose':lower,'covariate_time':config.covariate_time,
        'industry_categories':sorted(data.ind_refined.unique().tolist()),
        'dummy_columns':[c for c in data if c.startswith('ind_dummy')]}
    if not len(data):raise ValueError('Empty analysis sample')
    return data,audit

def prepare_year(data,year,config):
    result=prepare_year_data(data,year,config.base_year).reset_index(drop=True)
    if year<config.base_year and config.pre_sign=='chronological':result['Y_diff']*=-1
    return result

def observed_grid(frame,config):
    values=np.sort(frame.groupby('ind_code')[config.dose].first().unique())
    low,high=values[0],values[-1]
    steps=int(np.floor((high-low)/config.diff_step))
    if steps>20_000_000:raise ValueError('Grid exceeds 20 million points; increase diff_step')
    grid=low+config.diff_step*np.arange(steps+1)
    if high-grid[-1]>1e-12:grid=np.r_[grid,high]
    return grid,cluster_bandwidth(frame,config)
