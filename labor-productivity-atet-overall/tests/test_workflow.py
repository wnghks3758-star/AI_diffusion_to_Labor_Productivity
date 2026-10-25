"""Check exclusion leakage, aggregation covariance, and checkpoint reuse."""
import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from atet_did.config import ATETConfig
from atet_did.estimation import aggregate
from atet_did.models import fit_nuisance
from atet_did.kernel_gps import fit_kernel_gps,kernel_log_ratio
from atet_did.numerics import np_kernel
from atet_did.pipeline import ROOT, reference_frame, run


@contextmanager
def test_directory():
    parent=(ROOT/".cache"/"test_tmp").resolve()
    parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as folder:
        assert Path(folder).resolve().parent==parent
        yield folder


def fixture():
    rng=np.random.default_rng(12)
    rows=[]
    for industry in range(24):
        dose=.05+.03*industry
        x=float(rng.normal())
        for firm in range(12):
            rows.append(dict(ind_code=industry,firm_code=industry*100+firm,office_ratio=dose,x=x,
                             Y_diff=.2*dose+x+rng.normal(scale=.2),large_firm=int(firm>=6)))
    return pd.DataFrame(rows)


class WorkflowTests(unittest.TestCase):
    def config(self,cache,**kwargs):
        return ATETConfig(dose_transform="none",reference_doses=[.3,.5],gps_features=["x"],
            mu_industry_features=["x"],mu_firm_features=[],include_industry_dummies=False,
            gps_model="linear",mu_model="linear",gps_params={},mu_params={},cache_dir=str(cache),
            n_splits=3,gps_n_splits=3,mu_n_splits=3,n_repeats=1,prediction_aggregation="single",**kwargs)

    def test_training_exclusion_never_fits_or_uses_variance_below_reference(self):
        frame=fixture()
        with test_directory() as temp:
            c=self.config(temp,sample_exclusion="training")
            evaluated,fit_config,info=reference_frame(frame,.5,c)
            self.assertEqual(len(evaluated),len(frame))
            self.assertLess(info["training_eligible_industries"],info["analysis_industries"])
            fitted=fit_nuisance(evaluated,fit_config)
            industries=evaluated.groupby("ind_code",as_index=False).first()
            eligible=industries.office_ratio.ge(.5).to_numpy()
            for fold in fitted["audit"]["gps_folds"]:
                self.assertTrue(eligible[fold["train_industry_positions"]].all())
                self.assertTrue(eligible[fold["variance_validation_industry_positions"]].all())
                self.assertFalse(set(fold["train_industry_positions"]) & set(fold["test_industry_positions"]))
            for fold in fitted["audit"]["mu_folds"]:
                self.assertTrue(eligible[fold["train_industry_positions"]].all())
                self.assertFalse(set(fold["train_industry_positions"]) & set(fold["test_industry_positions"]))
            self.assertEqual(len(fitted["mean"]),len(frame))
            self.assertTrue(np.isfinite(fitted["mean"]).all())
            cached=fit_nuisance(evaluated,fit_config)
            self.assertTrue(cached["audit"]["cache_hit"])
            self.assertEqual(fitted["key"],cached["key"])
            different=fit_nuisance(evaluated,replace(fit_config,training_min_dose=.4))
            self.assertNotEqual(fitted["key"],different["key"])

    def test_analysis_exclusion_removes_comparison_rows_and_resets_indices(self):
        with test_directory() as temp:
            c=self.config(temp,sample_exclusion="analysis")
            frame,fit_config,info=reference_frame(fixture(),.5,c)
            self.assertTrue(frame.office_ratio.ge(.5).all())
            self.assertEqual(list(frame.index),list(range(len(frame))))
            self.assertIsNone(fit_config.training_min_dose)
            self.assertEqual(info["analysis_industries"],info["training_eligible_industries"])
            self.assertEqual(info["excluded_analysis_industries"],info["excluded_training_industries"])

    def test_overall_scores_include_cross_dose_and_size_covariance(self):
        frame=pd.DataFrame(dict(firm_code=range(4),ind_code=[1,1,2,2],office_ratio=[.4,.4,.6,.6],large_firm=[0,1,0,1]))
        with test_directory() as temp:
            c=self.config(temp)
            results={}
            for d,point,scores in [(.4,2.,np.array([1.,-1.])),(.6,4.,np.array([-1.,1.]))]:
                r={scope:dict(status="completed",estimate=point+(1 if scope=="large" else 0)) for scope in ["total","sme","large"]}
                v={scope:scores.copy() for scope in r}
                results[d]=(r,v)
            table,v=aggregate(frame,results,.3,c,2023,"strict")
            total=table[table.effect.eq("total")].iloc[0]
            self.assertEqual(total.estimate,3.)
            self.assertEqual(total.se,0.)
            moderation=table[table.effect.eq("moderation")].iloc[0]
            self.assertEqual(moderation.estimate,1.)
            self.assertEqual(moderation.se,0.)
            results[.6][0]["total"]["status"]="failed"
            strict,_=aggregate(frame,results,.3,c,2023,"strict")
            supported,_=aggregate(frame,results,.3,c,2023,"supported")
            self.assertEqual(strict[strict.effect.eq("total")].iloc[0].status,"failed")
            row=supported[supported.effect.eq("total")].iloc[0]
            self.assertEqual(row.estimate,2.)
            self.assertEqual(row.coverage,.5)

    def test_configuration_rejects_unimplemented_or_invalid_inputs(self):
        for values in [dict(reference_doses=[0]),dict(reference_doses=[.153,.153]),dict(psmethod=3),dict(psmethod=True),dict(sample_exclusion="drop")]:
            with self.assertRaises(ValueError):
                ATETConfig(**values)

    def test_ps2_training_exclusion_median_cache_and_no_gaussian_fit(self):
        import joblib
        frame=fixture()
        with test_directory() as temp:
            c=replace(self.config(temp,sample_exclusion="training"),psmethod=2,
                gps_model="random_forest",gps_params=dict(n_estimators=8,max_depth=2,min_samples_leaf=2),
                n_repeats=2,prediction_aggregation="median",training_min_dose=.5)
            with patch("atet_did.models.fit_gps_oof",side_effect=AssertionError("PS2 must not use Gaussian GPS")):
                nuisance=fit_nuisance(frame,c)
            self.assertIsNone(nuisance["mean"])
            self.assertIsNone(nuisance["variance"])
            centers=np.array([.35,.5,.65])
            kernel=fit_kernel_gps(frame,centers,.25,c)
            ind=frame.groupby("ind_code",as_index=False).first()
            eligible=ind.office_ratio.ge(.5).to_numpy()
            norm=np.stack([np_kernel(ind.office_ratio,point,.25) for point in centers],axis=1)[eligible].sum(axis=0)
            np.testing.assert_allclose(kernel["audit"]["kernel_normalizers"],norm)
            stored=[]
            for repeat in kernel["audit"]["repeats"]:
                for fold in repeat["folds"]:
                    self.assertTrue(eligible[fold["train_industry_positions"]].all())
                    self.assertTrue(eligible[fold["diagnostic_industry_positions"]].all())
                    self.assertFalse(set(fold["train_industry_positions"]) & set(fold["test_industry_positions"]))
                stored.append(joblib.load(Path(temp)/"kernel_gps"/(repeat["cache_key"]+".joblib"))["predictions"])
            np.testing.assert_allclose(kernel["predictions"],np.median(np.stack(stored),axis=0)*norm)
            with patch("atet_did.kernel_gps.fit_regression",side_effect=AssertionError("must reuse cached predictions")):
                cached=fit_kernel_gps(frame,centers,.25,c)
            self.assertTrue(cached["audit"]["cache_hit"])
            np.testing.assert_array_equal(cached["predictions"],kernel["predictions"])
            changed=fit_kernel_gps(frame,centers,.25,replace(c,training_min_dose=.4))
            self.assertNotEqual(kernel["audit"]["repeats"][0]["cache_key"],changed["audit"]["repeats"][0]["cache_key"])

    def test_ps2_nonpositive_predictions_fail_without_clipping(self):
        frame=pd.DataFrame(dict(ind_code=[1,2]))
        kernel=dict(centers=np.array([.3,.6]),industries=np.array([1,2]),predictions=np.array([[.2,.4],[.1,.3]]))
        ratio=kernel_log_ratio(frame,.6,.3,kernel,np.ones(2),np.ones(2))
        np.testing.assert_allclose(ratio,np.log([2.,3.]))
        kernel["predictions"][0,0]=0
        with self.assertRaisesRegex(ValueError,"Nonpositive PS2"):
            kernel_log_ratio(frame,.6,.3,kernel,np.ones(2),np.ones(2))

    def test_ps2_parameter_profiles_and_lasso_kernel_fit(self):
        with test_directory() as temp:
            for model in ["spline","random_forest","lasso","ridge","linear","xgboost"]:
                c=replace(self.config(temp),psmethod=2,gps_model=model,gps_params=None,mu_params=None).resolve_models(ROOT)
                self.assertIsInstance(c.gps_params,dict)
            c=replace(self.config(temp),psmethod=2,gps_model="lasso",gps_params=None,mu_params=None).resolve_models(ROOT)
            self.assertEqual(c.for_model("gps").inner_splits,3)
            kernel=fit_kernel_gps(fixture(),np.array([.35,.5,.65]),.25,c)
            self.assertEqual(kernel["predictions"].shape,(24,3))
            self.assertTrue(np.isfinite(kernel["predictions"]).all())

    def test_ps2_real_panel_all_exclusion_modes(self):
        settings=json.loads((ROOT/"atet_settings.json").read_text(encoding="utf-8"))
        with test_directory() as temp:
            for mode in ["none","training","analysis"]:
                c=ATETConfig(**{**settings,"years":[2023],"reference_doses":[.153,.175],"psmethod":2,"sample_exclusion":mode,
                    "cache_dir":str(Path(temp)/"cache"),"support_policy":"supported","n_repeats":1,
                    "prediction_aggregation":"single","gps_n_splits":2,"mu_n_splits":2,
                    "gps_model":"random_forest","gps_params":dict(n_estimators=5,max_depth=2,min_samples_leaf=5),
                    "mu_model":"linear","mu_params":{}})
                table,out=run(c,Path(temp)/mode)
                self.assertEqual(len(table),8)
                self.assertTrue(table.psmethod.eq(2).all())
                total=table[table.effect.eq("total")]
                self.assertTrue(total.status.eq("completed").all())
                self.assertTrue(total.coverage.gt(0).all())
                audit=json.loads((out/"audit.json").read_text(encoding="utf-8"))
                for year in audit["references_years"]:
                    self.assertFalse(year["kernel_gps"]["gaussian_assumption"])
                    self.assertEqual(year["kernel_gps"]["training_eligible_industries"],year["sample"]["training_eligible_industries"])

    def test_real_panel_all_modes_and_completed_output_reuse(self):
        settings=json.loads((ROOT/"atet_settings.json").read_text(encoding="utf-8"))
        with test_directory() as temp:
            for mode in ["none","training","analysis"]:
                values={**settings,"years":[2023],"reference_doses":[.153,.175],"psmethod":1,"sample_exclusion":mode,"cache_dir":str(Path(temp)/"cache"),
                    "gps_model":"linear","mu_model":"linear","gps_params":{},"mu_params":{},
                    "n_repeats":2,"prediction_aggregation":"median","support_policy":"supported"}
                c=ATETConfig(**values)
                table,out=run(c,Path(temp)/mode)
                self.assertEqual(len(table),8)
                self.assertEqual(set(table.reference_raw),{.153,.175})
                self.assertTrue((out/"report.html").exists())
                self.assertTrue((out/"config.json").read_text(encoding="utf-8"))
                total=table[table.effect.eq("total")]
                self.assertTrue(total.status.eq("completed").all())
                self.assertTrue(np.isfinite(total[["estimate","se"]]).all().all())
                with patch("atet_did.pipeline.prepare",side_effect=AssertionError("should reuse completed result")):
                    cached,cached_out=run(c,out)
                self.assertEqual(cached_out,out)
                np.testing.assert_allclose(cached.estimate,table.estimate,equal_nan=True)
                changed=replace(c,trim=.02)
                with self.assertRaises(FileExistsError):
                    run(changed,out)


if __name__=="__main__":
    unittest.main()
