import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import repeat_paper_1538 as repeated
import report_repeat_1538 as reporting
from labor_did.io import write_json
from paper_did.config import PaperConfig


def table(point, se):
    return pd.DataFrame([dict(year=2023, estimand="finite_difference_ACRT", pair_id=-1, effect="total",
                             estimate=point, se=se, p_value=.5, coverage=.99, status="completed")])


class RepeatedPaperTests(unittest.TestCase):
    def test_checkpoint_does_not_publish_cases_before_all_pre_gates_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "plan.json", {})
            write_json(root / "progress.json", dict(completed_repetitions={"2023": 50}, completed_years=[2023]))
            (root / "retained/pending").mkdir(parents=True)
            (root / "retained/pending/estimates.csv").write_text("estimate\n.1\n")
            with zipfile.ZipFile(reporting.checkpoint(root)) as z:
                self.assertEqual(set(z.namelist()), {"plan.json", "progress.json"})
            write_json(root / "progress.json", dict(completed_repetitions={"2023": 50, "2020": 50, "2021": 50}, completed_years=[2023, 2020, 2021]))
            with zipfile.ZipFile(reporting.checkpoint(root)) as z:
                self.assertIn("retained/pending/estimates.csv", z.namelist())

    def test_same_sample_repeats_do_not_divide_se_by_sqrt_repetitions(self):
        out = repeated.median_repeats([table(-.1, .2) for _ in range(50)]).iloc[0]
        self.assertAlmostEqual(out.estimate, -.1)
        self.assertAlmostEqual(out.se, .2)
        self.assertEqual(out.repetitions, 50)

    def test_median_includes_between_split_uncertainty(self):
        out = repeated.median_repeats([table(v, .2) for v in [0., 1., 3.]]).iloc[0]
        self.assertEqual(out.estimate, 1.)
        self.assertAlmostEqual(out.se, np.sqrt(1.04))
        self.assertAlmostEqual(out.split_sd, np.std([0., 1., 3.], ddof=1))

    def test_failed_repetition_is_not_silently_selected_out(self):
        out = repeated.median_repeats([table(1., .2), table(np.nan, np.nan)]).iloc[0]
        self.assertTrue(np.isnan(out.estimate))
        self.assertEqual(out.status, "failed")
        with self.assertRaises(ValueError):
            repeated.median_repeats([table(1., .2), table(1., .2).assign(effect="sme")])

    def test_single_repeat_reproduces_point_and_se(self):
        out = repeated.median_repeats([table(.15, .08)]).iloc[0]
        self.assertEqual(out.estimate, .15)
        self.assertEqual(out.se, .08)

    def test_plan_keeps_fold_counts_and_learner_random_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            c = PaperConfig(gps_n_splits=4, mu_n_splits=3, gps_fold_seed=107, mu_fold_seed=73,
                            gps_model="xgboost", mu_model="xgboost", gps_params={"random_state": 9}, mu_params={"random_state": 19})
            write_json(parent / "config.json", c.to_dict())
            (parent / "estimates.csv").write_text("estimate,se\n.1,.2\n")
            data = root / c.data_path
            data.parent.mkdir(parents=True)
            data.write_bytes(b"test-input-fingerprint-only")
            with patch.object(repeated, "ROOT", root):
                p = repeated.make_plan(root / "out", parent)
            self.assertEqual([(e["config"]["diff_step"], e["repetitions"]) for e in p["entries"]], [(1e-5, 1), (.0005, 50), (1e-5, 50)])
            for role in ["gps", "mu"]:
                self.assertEqual(len(set(p["seeds"][role])), 50)
                self.assertEqual(p["seeds"][role][0], c.fold_settings(role)[1])
            for e in p["entries"]:
                for key in c.to_dict():
                    if key not in ["diff_step", "label"]:
                        self.assertEqual(e["config"][key], c.to_dict()[key])


if __name__ == "__main__":
    unittest.main()
