"""Predeclared exploratory pattern flags; no specification is discarded."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass
class PatternCriteria:
    alpha: float = .05
    pre_years: tuple = (2020,2021)
    post_years: tuple = (2023,2024,2025)
    pre_effects: tuple = ('total','sme','large','moderation')
    decline_effects: tuple = ('total','sme','large')
    recovery_effects: tuple = ('total','sme','large')
    require_monotonic_recovery: bool = True
    recovery_tolerance: float = 0.

    def to_dict(self):return asdict(self)

def evaluate_pattern(table,criteria=None,p_column='p_value'):
    c=criteria or PatternCriteria()
    indexed=table.set_index(['effect','year'])
    def cell(effect,year,column):
        try:return float(indexed.loc[(effect,year),column])
        except (KeyError,TypeError,ValueError):return np.nan
    needed=[(e,y) for e in c.pre_effects for y in (*c.pre_years,*c.post_years)]
    complete=all(np.isfinite(cell(e,y,v)) for e,y in needed for v in ['estimate',p_column])
    result={'complete':complete,'inference_column':p_column}
    def sig(effect,year,sign):return sign*cell(effect,year,'estimate')>0 and cell(effect,year,p_column)<c.alpha
    result['pre_all_nonsignificant']=complete and all(cell(e,y,p_column)>=c.alpha for e in c.pre_effects for y in c.pre_years)
    first,last=c.post_years[0],c.post_years[-1]
    result['initial_significant_decline']=complete and all(sig(e,first,-1) for e in c.decline_effects)
    recovery=[]
    for effect in c.recovery_effects:
        values=np.array([cell(effect,y,'estimate') for y in c.post_years])
        recovery.append(bool(values[-1]>values[0] and (not c.require_monotonic_recovery or np.all(np.diff(values)>=-c.recovery_tolerance))))
    result['recovery']=complete and all(recovery)
    def crossing(effect):return next((y for y in c.post_years if cell(effect,y,'estimate')>0),None)
    sme,large=crossing('sme'),crossing('large')
    result['sme_first_positive_year']=sme;result['large_first_positive_year']=large
    result['sme_crosses_positive_earlier']=complete and sme is not None and (large is None or sme<large)
    result['sme_2025_significantly_positive']=complete and sig('sme',last,1)
    persistent=all(sig('moderation',y,-1) for y in c.post_years)
    strengthening=sig('moderation',last,-1) and cell('moderation',last,p_column)<=cell('moderation',first,p_column)
    result['moderation_negative_significant']=complete and all(cell('moderation',y,'estimate')<0 for y in c.post_years) and (persistent or strengthening)
    core=['pre_all_nonsignificant','initial_significant_decline','recovery',
          'sme_crosses_positive_earlier','moderation_negative_significant']
    result['pattern_match']=all(result[k] for k in core)
    result['preferred_match']=result['pattern_match'] and result['sme_2025_significantly_positive']
    result['criteria_passed']=sum(bool(result[k]) for k in core+['sme_2025_significantly_positive'])
    return result

def holm_adjust(p_values,family_size=None):
    values=np.asarray(p_values,float)
    finite=np.isfinite(values);clean=values[finite];m=family_size or len(clean)
    if m<len(clean):raise ValueError('family_size smaller than supplied p-values')
    order=np.argsort(clean);ordered=clean[order]
    adjusted=np.minimum(1,np.maximum.accumulate((m-np.arange(len(clean)))*ordered))
    result=np.full(len(values),np.nan);target=np.empty_like(clean);target[order]=adjusted;result[finite]=target
    return result
