import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from dml_acr.config import AnalysisConfig
from dml_acr.score import normalized_acr_score
from dml_acr.aggregation import cluster_contribution,draw_joint
from dml_acr.support import averaging_weights,get_trimmed_weights,select_supported_centers
from dml_acr.models import predict_mu,make_folds
from dml_acr.criteria import evaluate_pattern,holm_adjust
from dml_acr.preprocessing import prepare_panel,prepare_year

class AlgebraTests(unittest.TestCase):
    def test_ratio_influence_is_weight_derivative(self):
        r=np.array([.1,.9,1.3,-.4]);a=np.array([1.,2.,3.,0.]);b=np.array([0.,3.,1.,4.])
        point,influence,_=normalized_acr_score(r,a,b,.02)
        eps=1e-6
        for j in range(4):
            direction=np.ones(4);direction[j]+=eps
            changed=normalized_acr_score(r,a*direction,b*direction,.02)[0]
            self.assertAlmostEqual((changed-point)/eps,influence[j]/len(r),places=4)
        self.assertAlmostEqual(influence.mean(),0.,places=12)

    def test_aggregate_before_bootstrap_preserves_paired_covariance(self):
        rng=np.random.default_rng(0);cluster_codes=np.array([0,0,1,2,2,3]);n=4
        influences=rng.normal(size=(6,7));influences-=influences.mean(axis=0)
        points=rng.normal(size=7);weights=rng.uniform(size=7);weights/=weights.sum()
        scores=np.column_stack([cluster_contribution(influences[:,j],cluster_codes,n) for j in range(7)])
        c=AnalysisConfig(n_bootstrap=99,bootstrap_batch_size=13)
        multipliers=np.random.default_rng(c.seed).normal(size=(c.n_bootstrap,n))
        old=(points+multipliers@scores)@weights
        new=draw_joint([points@weights],(scores@weights)[:,None],c)[:,0]
        np.testing.assert_allclose(old,new,rtol=1e-13,atol=1e-13)
        paired=draw_joint([1.,2.,1.],np.column_stack([scores[:,0],scores[:,1],scores[:,1]-scores[:,0]]),c)
        np.testing.assert_allclose(paired[:,2],paired[:,1]-paired[:,0],atol=1e-13)

    def test_compressed_support_matches_firm_weights(self):
        rng=np.random.default_rng(10)
        for heterogeneous in [False,True]:
            group=np.repeat(np.arange(10),[2,3,4,3,2,4,2,3,2,4]);dose=group/4+.1
            mean=dose*.7+rng.normal(size=len(group))*(.01 if heterogeneous else 0)
            frame=pd.DataFrame({'ind_code':group,'office_ratio':dose,'D_pred':mean,'sigma_pred':.7})
            grid=np.linspace(.2,2.,21)
            selected,table=select_supported_centers(frame,grid,.8,.01,min_ess=2,trim_threshold=.07)
            expected=[]
            for d in grid:
                _,_,info=get_trimmed_weights(frame,d,d-.01,frame.D_pred.values,frame.sigma_pred.values,.8,trim_threshold=.07)
                row=table.loc[np.isclose(table.d,d)].iloc[0]
                self.assertAlmostEqual(info['ess_d'],row.ess_d,places=10)
                self.assertAlmostEqual(info['ess_dp'],row.ess_dp,places=10)
                if min(info['ess_d'],info['ess_dp'])>2:expected.append(d)
            np.testing.assert_allclose(selected,expected)

    def test_strict_gap_excludes_firms_and_renormalizes(self):
        frame=pd.DataFrame({'ind_code':[1,2,3],'firm_code':[1,2,3],
            'office_ratio':[.1,.5,.9],'large_firm':[0,0,1]})
        grid=np.array([.1,.3,.5,.7,.9]);supported=grid[[0,1,3,4]]
        c=AnalysisConfig(gap_policy='strict')
        weights,audit=averaging_weights(frame,supported,grid,1.,c)
        np.testing.assert_allclose(weights,[.5,0,0,.5]);self.assertEqual(audit['n_gap_firms'],1)
        c.gap_policy='bridge';weights,audit=averaging_weights(frame,supported,grid,1.,c)
        np.testing.assert_allclose(weights,[1/3,1/6,1/6,1/3]);self.assertEqual(audit['n_used_firms'],3)

class ModelTests(unittest.TestCase):
    def test_group_folds_never_share_industries(self):
        frame=pd.DataFrame({'ind_code':np.repeat(np.arange(12),2),'ind_refined':np.repeat(['a']*6+['b']*6,2)})
        folds=make_folds(frame,AnalysisConfig(n_splits=3))
        for train,test in folds:self.assertFalse(set(frame.iloc[train].ind_code)&set(frame.iloc[test].ind_code))
        np.testing.assert_array_equal(np.sort(np.concatenate([test for _,test in folds])),np.arange(len(frame)))

    def test_disk_predictions_equal_direct_repeated_median(self):
        with TemporaryDirectory() as temp:
            folder=Path(temp);frame=pd.DataFrame({'x':np.arange(12)/12,'office_ratio':np.linspace(.2,1.3,12)})
            X=frame.to_numpy();y=np.sin(X[:,0])+X[:,1]**2;entries=[];models=[]
            for rep in range(2):
                for fold in range(2):
                    test=np.arange(fold,12,2);train=np.setdiff1d(np.arange(12),test)
                    model=XGBRegressor(n_estimators=4,max_depth=2,n_jobs=1,tree_method='exact',random_state=rep).fit(X[train],y[train])
                    name=f'{rep}_{fold}.joblib';joblib.dump({'model':model,'test':test,'train':train},folder/name)
                    entries.append({'file':name,'repetition':rep});models.append((rep,model,test))
            values=np.array([.2,.21,.22,.8,.81,1.2]);expected=np.empty((2,len(values),12))
            for rep,model,test in models:
                for j,d in enumerate(values):
                    features=X[test].copy();features[:,-1]=d;expected[rep,j,test]=model.predict(features)
            c=AnalysisConfig(n_repeats=2,cache_dir=temp,prediction_batch_points=2)
            with predict_mu(frame,values,c,folder,{'models':entries,'features':list(frame.columns)}) as predictions:
                actual=np.concatenate([batch for _,_,batch in predictions.batches()])
            np.testing.assert_array_equal(actual,np.median(expected,axis=0))
            self.assertFalse(list(folder.glob('mu_predictions_*')))

class ConfigurationAndPatternTests(unittest.TestCase):
    def test_invalid_configuration_fails_early(self):
        for changes in [{'diff_step':0},{'trim_threshold':0},{'n_splits':1},{'mu_features':['ln_firm_prod']}, {'gap_policy':'wrong'}]:
            with self.assertRaises(ValueError):AnalysisConfig(**changes)
        with self.assertRaises(TypeError):AnalysisConfig.from_dict({'typo_ess':5})

    def test_flags_and_missing_results(self):
        rows=[]
        post={'total':[-1.5,-.2,.1],'sme':[-1.,.2,.5],'large':[-2.,-.5,-.2],'moderation':[-1.,-.7,-.7]}
        for effect,values in post.items():
            for year in [2020,2021]:rows.append({'effect':effect,'year':year,'estimate':0.,'p_value':.8})
            for year,point in zip([2023,2024,2025],values):rows.append({'effect':effect,'year':year,'estimate':point,'p_value':.001})
        frame=pd.DataFrame(rows);flags=evaluate_pattern(frame)
        self.assertTrue(flags['preferred_match'])
        self.assertFalse(evaluate_pattern(frame.iloc[1:])['pattern_match'])
        frame.loc[(frame.effect=='total')&(frame.year==2020),'p_value']=.01
        self.assertFalse(evaluate_pattern(frame)['pre_all_nonsignificant'])

    def test_holm_counts_pending_family(self):
        np.testing.assert_allclose(holm_adjust([.01,.03,.2]),[.03,.06,.2])
        np.testing.assert_allclose(holm_adjust([.01,.03,.2],family_size=5),[.05,.12,.6])

if __name__=='__main__':unittest.main()
