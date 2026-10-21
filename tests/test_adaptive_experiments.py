import unittest
import numpy as np
from dml_acr.config import AnalysisConfig
from dml_acr.adaptive_experiments import neighbours, local_operations


class AdaptiveTests(unittest.TestCase):
    def test_local_coordinates_preserve_estimand_and_feature_toggle_size(self):
        for model,params in [('xgboost',{'n_estimators':160,'max_depth':3,'learning_rate':.05,'subsample':.8,'colsample_bytree':.8}),
                             ('ridge',{'alpha':1.}),('spline_ridge',{'alpha':10.,'n_knots':3,'degree':2})]:
            base=AnalysisConfig(mu_model=model,mu_params=params,n_repeats=3,n_bootstrap=999).to_dict()
            entries=neighbours(base,20,np.random.default_rng(123),set())
            self.assertEqual(len({e['config_id'] for e in entries}),20)
            self.assertEqual(sum(e['local_coordinate_count']==1 for e in entries),10)
            for entry in entries:
                new=entry['config'];self.assertEqual(new['n_bootstrap'],1999)
                for key in ['mu_model','gps_model','outcome','dose','alpha','seed','base_year',
                            'covariate_time','size_quantile','diff_step','gap_policy','sample_reference_features']:
                    self.assertEqual(base[key],new[key])
                changed=[]
                for key in base:
                    if key=='n_bootstrap':continue
                    if key=='mu_params':
                        changed.extend('mu_params.'+p for p in base[key] if base[key][p]!=new[key][p])
                    elif base[key]!=new[key]:changed.append(key)
                self.assertEqual(set(changed),{c['coordinate'] for c in entry['changes']})
                self.assertEqual(len(changed),entry['local_coordinate_count'])
                for key in ['gps_features','mu_features']:
                    if key in changed:self.assertEqual(len(set(base[key])^set(new[key])),1)

    def test_no_inactive_industry_or_spline_operations(self):
        c=AnalysisConfig(industry_scheme='none',gps_model='linear',mu_model='ridge',mu_params={'alpha':1.}).to_dict()
        ops=local_operations(c)
        self.assertNotIn('industry',ops)
        self.assertEqual([key for key,value in ops['model']],['mu_params.alpha']*2)

    def test_seed_and_excluded_fingerprints_control_generation(self):
        c=AnalysisConfig(n_repeats=3,mu_model='ridge',mu_params={'alpha':1.}).to_dict()
        a=neighbours(c,20,np.random.default_rng(7),set())
        b=neighbours(c,20,np.random.default_rng(7),set())
        self.assertEqual(a,b)
        excluded={e['config_id'] for e in a}
        other=neighbours(c,20,np.random.default_rng(7),excluded.copy())
        self.assertFalse({e['config_id'] for e in other}&excluded)


if __name__=='__main__':unittest.main()
