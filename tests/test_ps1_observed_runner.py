import contextlib
import importlib.metadata
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_workflow import synthetic_panel

from labor_did.io import versions
from paper_did.config import PaperConfig
from ps1_observed import ROOT, run
from run_observed import main


class StandaloneObservedTests(unittest.TestCase):
    def test_versions_supports_standard_xgboost_distribution_on_macos(self):
        installed_version = importlib.metadata.version
        def fake_version(name):
            if name == "xgboost-cpu":
                raise importlib.metadata.PackageNotFoundError(name)
            return "3.1.3" if name == "xgboost" else installed_version(name)
        with patch("labor_did.io.importlib.metadata.version", side_effect=fake_version):
            v = versions()
        self.assertEqual(v["xgboost"], "3.1.3")
        self.assertNotIn("xgboost-cpu", v)

    def config(self, root):
        data = root / "input.xlsx"
        synthetic_panel().to_excel(data, index=False)
        return PaperConfig(data_path=str(data), cache_dir=str(root / "cache"), years=[2023],
                           gps_features=["ln_ind_avg_pay"], mu_industry_features=["ln_ind_avg_pay"],
                           mu_firm_features=["firm_rnd_ratio"], gps_model="linear", mu_model="xgboost",
                           mu_params={"n_estimators": 8, "max_depth": 2}, n_splits=2, inner_splits=2,
                           include_industry_dummies=False, industry_scheme="none", trim=1., diff_step=.1)

    def test_real_run_resume_and_cluster_covariance_without_history_files(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            c = self.config(root)
            t = run(c, root / "out")
            self.assertEqual(len(t), 4)
            self.assertTrue(t.status.eq("completed").all())
            self.assertTrue((root / "out/report.html").exists())
            self.assertTrue((root / "out/finite_difference_ACRT_-1.png").exists())
            with patch("ps1_observed.fit_nuisance", side_effect=AssertionError("completed run must not refit")):
                resumed = run(c, root / "out")
            np.testing.assert_allclose(t.estimate, resumed.estimate)
            with self.assertRaises(FileExistsError):
                run(replace(c, diff_step=.05), root / "out")
            arrays = np.load(root / "out/years/2023/cluster_scores.npz")
            np.testing.assert_allclose(np.linalg.norm(arrays["supported"], axis=0), t.se)
            np.testing.assert_allclose(arrays["supported"][:, 3], arrays["supported"][:, 2] - arrays["supported"][:, 1])
            endpoints = pd.read_csv(root / "out/years/2023/endpoints.csv")
            panel = synthetic_panel().query("Year==2023")
            expected = np.unique(np.log(panel.office_ratio.to_numpy()*100))
            np.testing.assert_allclose(np.sort(endpoints.dose.unique()), np.sort(expected))

    def test_cli_settings_override_uses_new_output(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            c = self.config(root)
            f = root / "config.json"
            f.write_text(json.dumps(c.to_dict()))
            rc = main(["--config", str(f), "--set", "diff_step=0.05", "--out", str(root / "out")])
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads((root / "out/config.json").read_text())["diff_step"], .05)

    def test_shipped_config_and_data_need_no_saved_analysis(self):
        values = json.loads((ROOT / "observed_settings.json").read_text())
        c = PaperConfig(**values)
        self.assertTrue((ROOT / c.data_path).is_file())
        self.assertEqual((c.gps_model, c.mu_model, c.diff_step, c.trim, c.bandwidth_factor), ("xgboost", "xgboost", .0005, .01, .5))
        self.assertEqual(c.fold_settings("gps"), (4, 107))
        self.assertEqual(c.fold_settings("mu"), (3, 73))
        self.assertEqual(c.gps_features, c.mu_industry_features)
        self.assertEqual(c.size_variable, "sale")

    def test_rejects_industry_mismatch_and_unsupported_estimation_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.config(root)
            for changed in [replace(c, mu_industry_features=[]), replace(c, compute_acrt=False), replace(c, size_difference="common")]:
                with self.assertRaises(ValueError):
                    run(changed, root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
