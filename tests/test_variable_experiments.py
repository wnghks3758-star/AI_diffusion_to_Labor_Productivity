import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import replace
import numpy as np
import pandas as pd
from dml_acr.config import AnalysisConfig
from dml_acr.preprocessing import prepare_panel
from dml_acr.variable_experiments import make_variable_plan

class VariableExperimentTests(unittest.TestCase):
    def test_existing_fingerprints_preserved(self):
        root=Path('results/screen32')
        plan=json.loads((root/'plan.json').read_text())
        for entry in plan['entries']:
            path=root/'configs'/Path(entry['config']).name
            self.assertEqual(AnalysisConfig.load(path).fingerprint,entry['config_id'])

    def test_reference_covariates_hold_complete_case_sample(self):
        rows=[]
        for firm in range(4):
            for year in [2020,2022,2023]:
                rows.append({'firm_code':firm,'ind_code':101+firm//2,'Year':year,'sale':100+firm,
                    'ln_firm_prod':year/1000+firm/10,'office_ratio':.1+.1*(firm//2),
                    'x':float(firm//2),'z':np.nan if firm==0 and year==2020 else .5})
        raw=pd.DataFrame(rows)
        c=AnalysisConfig(years=[2020,2023],gps_features=['x'],mu_features=['x','z'],
            sample_reference_features=['x','z'],min_ess_prime=0,industry_min_groups=1,industry_min_obs=1)
        full,audit1=prepare_panel(raw,c)
        removed,audit2=prepare_panel(raw,replace(c,mu_features=['x']))
        natural,_=prepare_panel(raw,replace(c,mu_features=['x'],sample_reference_features=[]))
        self.assertEqual(set(full.firm_code),set(removed.firm_code))
        self.assertEqual(audit1['sample_firm_sha256'],audit2['sample_firm_sha256'])
        self.assertNotEqual(set(full.firm_code),set(natural.firm_code))
        self.assertEqual(len(raw),12)

    def test_plan_changes_only_learning_features(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);plan=make_variable_plan(root)
            configs=[AnalysisConfig.load(root/e['config']) for e in plan['entries']]
            self.assertEqual(len(configs),32)
            self.assertEqual(sum(e['variable_action']=='gps_remove' for e in plan['entries']),13)
            self.assertEqual(sum(e['variable_action']=='mu_remove' for e in plan['entries']),16)
            baseline=configs[0].to_dict()
            for config in configs:
                current=config.to_dict()
                for key in ['gps_features','mu_features','label']:
                    current.pop(key)
                expected={k:v for k,v in baseline.items() if k not in ['gps_features','mu_features','label']}
                self.assertEqual(current,expected)
            self.assertEqual(len(set(configs[-1].gps_features)-set(configs[-2].gps_features)),9)
            self.assertEqual(configs[0].mu_search_n_iter,0)
            with self.assertRaises(FileExistsError):make_variable_plan(root)

if __name__=='__main__':unittest.main()
