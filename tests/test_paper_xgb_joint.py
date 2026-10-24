import json
import tempfile
import unittest
from pathlib import Path

import search_paper_batch
import search_paper_xgb_joint as planner
from labor_did.io import write_json
from paper_did.config import PaperConfig


class JointXGBTests(unittest.TestCase):
    def test_fixed_settings_and_independent_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "prior"
            entries = [dict(number=n, config=PaperConfig(seed=n, label=f"paper_search_{n}").to_dict()) for n in range(1, 1501)]
            write_json(old / "plan.json", dict(entries=entries, source_sha256="previous", hypothesis_estimand="finite_difference_ACRT",
                       alpha=.05, retention={}, criteria={}, focused_filter={}))
            write_json(old / "summary.json", dict(cumulative_attempted=1500, retained=187, cumulative_retained=465))
            plan = planner.make_plan(root / "new", [old])
            self.assertEqual(len(plan["entries"]), 200)
            self.assertEqual([e["number"] for e in plan["entries"]], list(range(1501, 1701)))
            self.assertEqual(plan["cumulative_correction_family_size"], 34000)
            configs = [e["config"] for e in plan["entries"]]
            for c in configs:
                self.assertEqual((c["gps_model"], c["mu_model"]), ("xgboost", "xgboost"))
                self.assertEqual((c["bandwidth_factor"], c["trim"]), (.5, .01))
                self.assertEqual(c["gps_features"], c["mu_industry_features"])
                self.assertEqual((c["size_variable"], c["size_years"], c["size_threshold"], c["size_time_aggregation"]), ("sale", None, "median", "mean"))
            self.assertEqual(len({tuple(c["gps_features"]) for c in configs}), 40)
            self.assertTrue(any(not c["mu_firm_features"] for c in configs))
            self.assertTrue(any(len(c["mu_firm_features"]) == 5 for c in configs))
            self.assertTrue(any(c["gps_n_splits"] != c["mu_n_splits"] for c in configs))
            self.assertTrue(any(c["gps_fold_seed"] != c["mu_fold_seed"] for c in configs))
            self.assertTrue(all(c["gps_params"] != c["mu_params"] for c in configs))
            self.assertEqual(json.loads((root / "new/plan.json").read_text())["count"], 200)
            search_paper_batch.validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
