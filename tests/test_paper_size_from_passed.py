import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import search_paper_batch
import search_paper_size_from_passed as planner
from labor_did.io import write_json
from paper_did.config import PaperConfig


def passed_table():
    return pd.DataFrame([dict(year=y, effect=e, estimand="finite_difference_ACRT",
                             estimate=-.1 if y == 2023 and e == "total" else 0.,
                             p_value=.01 if y == 2023 and e == "total" else .5)
                         for y in [2020, 2021, 2023, 2024, 2025] for e in ["total", "sme", "large", "moderation"]])


class SizeFromPassedTests(unittest.TestCase):
    def fixture(self, root):
        prior, start = [], 1
        table_text = passed_table().to_csv(index=False)
        for i, (count, retained) in enumerate(zip([200, 200, 200, 200, 300, 200], [5, 39, 64, 99, 71, 0])):
            p = root / str(i)
            entries = []
            for n in range(start, start + count):
                c = PaperConfig(seed=n, label=f"paper_search_{n}", trim=.05 if n % 2 else .1).to_dict()
                entries.append(dict(number=n, config=c))
                if n < start + retained:
                    folder = p / "retained" / c["label"]
                    write_json(folder / "config.json", c)
                    (folder / "estimates.csv").write_text(table_text)
            write_json(p / "plan.json", dict(entries=entries, source_sha256="earlier",
                       hypothesis_estimand="finite_difference_ACRT", alpha=.05, retention={}, criteria={}, focused_filter={}))
            prior.append(p)
            start += count
        write_json(prior[-1] / "summary.json", dict(cumulative_attempted=1300, retained=0, cumulative_retained=278, pattern_matches=0))
        return prior

    def test_200_passing_parents_only_size_variable_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = self.fixture(root)
            plan = planner.make_plan(root / "new", prior)
            self.assertEqual(plan["eligible_parent_count"], 278)
            self.assertEqual(len(plan["entries"]), 200)
            self.assertEqual(len({e["parent_label"] for e in plan["entries"]}), 200)
            self.assertEqual([plan["entries"][0]["number"], plan["entries"][-1]["number"]], [1301, 1500])
            self.assertEqual(plan["cumulative_correction_family_size"], 30000)
            for e in plan["entries"]:
                original = PaperConfig(**json.loads(Path(e["parent_config_path"]).read_text())).to_dict()
                changed = e["config"]
                self.assertEqual({k for k in original if original[k] != changed[k]}, {"label", "size_variable"})
                self.assertEqual(changed["size_variable"], "assets_annual")
                self.assertEqual(changed["size_threshold"], "median")
                self.assertEqual(changed["size_missing_policy"], "legacy")
                self.assertIsNone(changed["size_years"])
            search_paper_batch.validate_plan(plan)

    def test_previous_retained_folder_alone_does_not_prove_gate_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "plan.json", dict(source_sha256="previous"))
            folder = root / "retained/paper_search_1"
            write_json(folder / "config.json", PaperConfig(label="paper_search_1").to_dict())
            t = passed_table()
            t.loc[(t.year == 2020) & (t.effect == "sme"), "p_value"] = .01
            t.to_csv(folder / "estimates.csv", index=False)
            self.assertEqual(planner.load_parents([root]), [])


if __name__ == "__main__":
    unittest.main()
