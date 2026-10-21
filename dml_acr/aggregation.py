"""Sum cluster influence vectors before drawing bootstrap multipliers."""
import numpy as np
from scipy.stats import norm

def cluster_contribution(influence,cluster_codes,n_clusters):
    influence=np.asarray(influence,float)
    return np.bincount(cluster_codes,weights=influence-influence.mean(),minlength=n_clusters)/len(influence)

def draw_joint(points,cluster_scores,config):
    """All years and groups use the same multiplier stream when clusters align."""
    points=np.asarray(points,float);scores=np.asarray(cluster_scores,float)
    n_clusters=scores.shape[0]
    rng=np.random.default_rng(config.seed)
    draws=np.empty((config.n_bootstrap,len(points)))
    for start in range(0,config.n_bootstrap,config.bootstrap_batch_size):
        stop=min(start+config.bootstrap_batch_size,config.n_bootstrap)
        multipliers=rng.normal(size=(stop-start,n_clusters))
        draws[start:stop]=points+multipliers@scores
    return draws

def inference(point,draws,alpha=0.05):
    if not np.isfinite(point) or not np.isfinite(draws).all():
        return {k:np.nan for k in ['estimate','se','p_value','ci_lower','ci_upper','percentile_lower','percentile_upper']}
    se=float(np.std(draws,ddof=1));critical=norm.ppf(1-alpha/2)
    p_value=float(2*norm.sf(abs(point/se))) if se>0 else (1. if point==0 else 0.)
    lower,upper=np.quantile(draws,[alpha/2,1-alpha/2])
    return {'estimate':float(point),'se':se,'p_value':p_value,
        'ci_lower':float(point-critical*se),'ci_upper':float(point+critical*se),
        'percentile_lower':float(lower),'percentile_upper':float(upper)}
