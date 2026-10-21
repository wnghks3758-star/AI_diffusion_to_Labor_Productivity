import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from dml_acr.config import AnalysisConfig
from dml_acr.random_experiments import make_random_plan, make_continuation_plan, screen_gate, run_random, PRIMARY
from dml_acr.pipeline import run_analysis
from dml_acr.io import write_json


def pre_table(year):
    return pd.DataFrame([{'year':year,'effect':e,'estimate':.1,'p_value':.5} for e in PRIMARY])


class RandomExperimentTests(unittest.TestCase):
    def test_continuation_preserves_random_stream_and_expands_family(self):
        with TemporaryDirectory() as temp:
            a=Path(temp)/'first';b=Path(temp)/'second';c=Path(temp)/'complete'
            first=make_random_plan(a,max_attempts=3,seed=31)
            write_json(a/'summary.json',{'attempted':3,'stop_reason':'budget_exhausted'})
            (a/'all_attempts.jsonl').write_text('[]\n')
            second=make_continuation_plan(a,b,max_attempts=4)
            complete=make_random_plan(c,max_attempts=7,seed=31)
            self.assertEqual(second['entries'],complete['entries'][3:])
            self.assertEqual(second['entries'][0]['run_id'],'random_004')
            self.assertEqual(second['prior_specifications'],67)
            self.assertEqual(second['all_random_planned_attempts'],7)
            self.assertFalse({e['config_id'] for e in first['entries']}&{e['config_id'] for e in second['entries']})
            with self.assertRaises(FileExistsError):make_continuation_plan(a,b,max_attempts=4)

    def test_gate_requires_finite_pretests_and_only_total_initial_decline(self):
        c=AnalysisConfig();table=pre_table(2020)
        self.assertIsNone(screen_gate(table,2020,c))
        table.loc[0,'p_value']=.049
        self.assertEqual(screen_gate(table,2020,c)['reason'],'pre_significant')
        table.loc[0,'p_value']=np.nan
        self.assertEqual(screen_gate(table,2020,c)['reason'],'nonfinite_gate')
        table=pre_table(2023);table.loc[0,['estimate','p_value']]=[-.3,.01]
        # SME/Large may be positive and insignificant: only total is a gate.
        self.assertIsNone(screen_gate(table,2023,c))
        table.loc[0,'p_value']=.05
        self.assertEqual(screen_gate(table,2023,c)['reason'],'initial_total_decline_not_met')

    def test_random_candidates_reproducible_and_inference_fixed(self):
        with TemporaryDirectory() as a,TemporaryDirectory() as b:
            x=make_random_plan(a,max_attempts=20,seed=123)
            y=make_random_plan(b,max_attempts=20,seed=123)
            self.assertEqual(x['entries'],y['entries'])
            self.assertEqual(len({e['config_id'] for e in x['entries']}),20)
            self.assertEqual(x['pattern_criteria']['decline_effects'],('total',))
            for e in x['entries']:
                c=e['config']
                self.assertEqual(c['alpha'],.05);self.assertEqual(c['seed'],42)
                self.assertEqual(c['bootstrap_unit'],'industry')
                self.assertEqual(c['size_quantile'],.5)
                self.assertEqual(c['dose'],'office_ratio')
            with self.assertRaises(FileExistsError):make_random_plan(a)

    def test_pipeline_does_not_fit_later_years_after_gate_rejects(self):
        table=pre_table(2020);table.loc[0,'p_value']=.001
        with TemporaryDirectory() as temp, \
             patch('dml_acr.pipeline.load_data',return_value=(pd.DataFrame(),'hash')), \
             patch('dml_acr.pipeline.prepare_panel',return_value=(pd.DataFrame(),{})), \
             patch('dml_acr.pipeline.prepare_year',return_value=pd.DataFrame()), \
             patch('dml_acr.pipeline.estimate_year',return_value=(table,pd.DataFrame(),{},{})) as estimate:
            _,status=run_analysis(AnalysisConfig(),temp,year_gate=screen_gate)
            self.assertEqual(status['status'],'rejected')
            self.assertEqual(estimate.call_count,1)
            self.assertEqual(status['years_evaluated'],[2020])

    def test_runner_retains_only_passes_and_stops_at_first_full_match(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);make_random_plan(root,max_attempts=3,seed=9)
            calls=[]
            def worker(command,**kwargs):
                folder=Path(command[command.index('--output')+1]);calls.append(folder.name)
                if len(calls)==1:
                    write_json(folder/'status.json',{'status':'rejected','gate':{'reason':'pre_significant','year':2020,'effect':'total'}})
                    (folder/'estimates.partial.csv').write_text('sensitive rejected detail')
                else:
                    table=pre_table(2020);table.to_csv(folder/'estimates.csv',index=False)
                    write_json(folder/'status.json',{'status':'completed','pattern':{'pattern_match':True,'preferred_match':False}})
                return SimpleNamespace(returncode=0)
            with patch('dml_acr.random_experiments.subprocess.run',side_effect=worker):
                summary=run_random(root)
            self.assertEqual(len(calls),2)
            self.assertEqual(summary['stop_reason'],'target_found')
            self.assertEqual(summary['accepted_detailed_cases'],1)
            self.assertFalse((root/'accepted'/'random_001').exists())
            self.assertTrue((root/'accepted'/'random_002'/'estimates.csv').exists())
            self.assertEqual(summary['random_budget_family_tests'],60)
            self.assertEqual(summary['cumulative_budget_family_tests'],1340)
            with patch('dml_acr.random_experiments.subprocess.run') as repeated:
                run_random(root)
                repeated.assert_not_called()


if __name__=='__main__':unittest.main()
