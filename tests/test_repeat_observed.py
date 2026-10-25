import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from test_workflow import synthetic_panel

import repeat_observed as repeats
from paper_did.config import PaperConfig


class RepeatedObservedTests(unittest.TestCase):
    def config(self, root):
        path = root / "input.xlsx"
        synthetic_panel().to_excel(path, index=False)
        c = PaperConfig(data_path=str(path), gps_model="xgboost", mu_model="xgboost", years=[2023],
                        gps_features=["ln_ind_avg_pay"], mu_industry_features=["ln_ind_avg_pay"],
                        mu_firm_features=["firm_rnd_ratio"], gps_params=dict(n_estimators=5, max_depth=2, random_state=7),
                        mu_params=dict(n_estimators=5, max_depth=2, random_state=9), n_splits=2, inner_splits=2,
                        industry_scheme="none", include_industry_dummies=False, diff_step=.1, trim=1.)
        f = root / "config.json"
        f.write_text(json.dumps(c.to_dict()))
        return c, f

    def test_plan_changes_four_seed_channels_only_and_keeps_original_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c, config = self.config(root)
            plan = repeats.make_plan(root / "out", config)
            self.assertEqual(plan["count"], 50)
            self.assertEqual(plan["entries"][0]["seeds"], dict(gps_fold=1, mu_fold=1, gps_model=7, mu_model=9))
            repeats.init_worker({}, c, root / "out")
            for key in plan["entries"][0]["seeds"]:
                self.assertEqual(len({e["seeds"][key] for e in plan["entries"]}), 50)
            for entry in plan["entries"]:
                changed = repeats.repeat_config(entry)
                self.assertEqual({k:v for k,v in changed.gps_params.items() if k != "random_state"},
                                 {k:v for k,v in c.gps_params.items() if k != "random_state"})
                self.assertEqual({k:v for k,v in changed.mu_params.items() if k != "random_state"},
                                 {k:v for k,v in c.mu_params.items() if k != "random_state"})
                self.assertEqual((changed.diff_step, changed.bandwidth_factor, changed.trim), (c.diff_step, c.bandwidth_factor, c.trim))

    def test_real_two_repeat_common_target_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            c, config = self.config(root)
            dest = root / "out"
            plan = repeats.make_plan(dest, config, count=2)
            progress = repeats.search(dest, workers=1)
            self.assertEqual(progress["status"], "completed")
            self.assertEqual(set(progress["completed"]), {0, 1})
            self.assertFalse((dest / ".scratch").exists())
            table = pd.read_csv(dest / "repeat_estimates.csv")
            common = table[table.target.eq("common")]
            for effect, rows in common.groupby("effect"):
                self.assertEqual(rows.n_used_firms.nunique(), 1)
                self.assertEqual(rows.coverage.nunique(), 1)
            aggregated = pd.read_csv(dest / "combined_estimates.csv").query("target=='common'")
            ordered = common.sort_values(["repetition", "effect"])
            for effect in aggregated.effect:
                vals = ordered[ordered.effect.eq(effect)]
                r = aggregated[aggregated.effect.eq(effect)].iloc[0]
                center = np.median(vals.estimate)
                self.assertAlmostEqual(r.estimate, center)
                self.assertAlmostEqual(r.se, np.sqrt(np.median(vals.se**2 + (vals.estimate-center)**2)))
            self.assertEqual(repeats.search(dest, workers=1)["status"], "completed")
            self.assertFalse((dest / ".scratch").exists())
            plan["entries"][0]["seeds"]["gps_fold"] += 1
            (dest / "plan.json").write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                repeats.search(dest, workers=1)


if __name__ == "__main__":
    unittest.main()
