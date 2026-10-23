import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import report_paper_search
import search_paper_more
import search_paper_next
from labor_did.io import write_json


class NextSearchTests(unittest.TestCase):
    def previous_fixture(self, root):
        previous = root / "previous"
        plan = search_paper_more.make_plan(previous)
        by_number = {e["number"]: e for e in plan["entries"]}
        for number in [212, 242]:
            write_json(
                previous / "retained" / f"paper_search_{number}" / "config.json",
                by_number[number]["config"],
            )
        write_json(
            previous / "summary.json",
            dict(
                attempted=200,
                cumulative_attempted=400,
                retained=5,
                cumulative_retained=12,
                pattern_matches=0,
                cumulative_raw_pattern_matches=0,
            ),
        )
        return previous, plan

    def test_401_600_deduplicates_against_both_previous_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous, prior = self.previous_fixture(root)
            plan = search_paper_next.make_plan(root / "next", previous)
            first = json.loads((search_paper_more.PREVIOUS / "plan.json").read_text())
            old_keys = {
                search_paper_more.effective_fingerprint(e["config"])
                for p in [first, prior]
                for e in p["entries"]
            }
            new_keys = {
                search_paper_more.effective_fingerprint(e["config"])
                for e in plan["entries"]
            }
            self.assertEqual(len(new_keys), 200)
            self.assertFalse(old_keys & new_keys)
            self.assertEqual(
                [plan["entries"][0]["number"], plan["entries"][-1]["number"]],
                [401, 600],
            )
            self.assertEqual(plan["cumulative_correction_family_size"], 12000)

    def test_worker_reads_cumulative_family_size_from_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            folder = dest / "retained/case"
            folder.mkdir(parents=True)
            write_json(
                dest / "plan.json", dict(cumulative_correction_family_size=12000)
            )
            pd.DataFrame(
                dict(
                    year=[2023],
                    estimand=["finite_difference_ACRT"],
                    effect=["total"],
                    estimate=[-0.1],
                    p_value=[0.00001],
                )
            ).to_csv(folder / "estimates.csv", index=False)
            with patch(
                "search_paper_more.engine.run_family",
                return_value=dict(retained=["case"], counts={"retained": 1}, family=0),
            ):
                search_paper_more.run_family([], dest)
            t = pd.read_csv(folder / "estimates.csv")
            self.assertAlmostEqual(t.iloc[0].p_bonferroni_cumulative, 0.12)

    def test_cumulative_report_uses_previous_cumulative_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous, _ = self.previous_fixture(root)
            dest = root / "audit/results/paper_search_401_600"
            search_paper_next.make_plan(dest, previous)
            write_json(
                dest / "progress.json",
                dict(
                    status="completed",
                    attempted=200,
                    counts={"initial_2023": 200},
                    retained=[],
                ),
            )
            with (
                patch("report_paper_search.ROOT", root),
                patch("report_paper_search.subprocess.run"),
            ):
                summary = report_paper_search.report(dest)
            self.assertEqual(summary["cumulative_retained"], 12)
            self.assertEqual(summary["cumulative_attempted"], 600)
            self.assertTrue((root / "audit/Paper_Search_401_600_ko.md").exists())


if __name__ == "__main__":
    unittest.main()
