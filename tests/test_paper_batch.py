import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import report_focused_history as focused
import report_paper_search as report
import search_paper_batch as batch
import search_paper_fifth as planner
from labor_did.io import write_json
from paper_did.config import PaperConfig
from search_paper_more import effective_fingerprint


class BatchTests(unittest.TestCase):
    def fixture(self, root):
        prior, all_entries = [], []
        for i in range(4):
            folder = root / str(i)
            entries = [
                dict(
                    number=n,
                    config=PaperConfig(seed=n, label=f"paper_search_{n}").to_dict(),
                )
                for n in range(i * 200 + 1, i * 200 + 201)
            ]
            write_json(
                folder / "plan.json",
                dict(
                    entries=entries,
                    hypothesis_estimand="finite_difference_ACRT",
                    alpha=0.05,
                    retention={},
                    criteria={},
                    focused_filter={},
                ),
            )
            prior.append(folder)
            all_entries += entries
        write_json(
            prior[-1] / "summary.json",
            dict(
                cumulative_attempted=800,
                retained=99,
                cumulative_retained=207,
                pattern_matches=0,
                cumulative_raw_pattern_matches=0,
            ),
        )
        return prior, all_entries

    def test_300_unique_against_800_and_scientific_invariants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior, old = self.fixture(root)
            plan = planner.make_plan(root / "new", prior)
            keys = {e["effective_fingerprint"] for e in plan["entries"]}
            self.assertEqual(len(keys), 300)
            self.assertFalse(keys & {effective_fingerprint(e["config"]) for e in old})
            self.assertEqual(
                [plan["entries"][0]["number"], plan["entries"][-1]["number"]],
                [801, 1100],
            )
            self.assertEqual(plan["correction_family_size"], 6000)
            self.assertEqual(plan["cumulative_correction_family_size"], 22000)
            for e in plan["entries"]:
                c = e["config"]
                self.assertEqual(c["covariate_time"], "baseline")
                self.assertEqual(c["base_year"], 2022)
                self.assertEqual(c["diff_step"], 0.0005)
            batch.validate_plan(plan)

    def test_worker_replaces_both_corrections_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            folder = dest / "retained/case"
            folder.mkdir(parents=True)
            write_json(
                dest / "plan.json",
                dict(
                    correction_family_size=6000, cumulative_correction_family_size=22000
                ),
            )
            rows = []
            levels = {
                "total": [-0.3, -0.1, 0.1],
                "sme": [-0.3, 0.1, 0.3],
                "large": [-0.4, -0.3, -0.1],
                "moderation": [-0.1, -0.4, -0.4],
            }
            for year in [2020, 2021, 2023, 2024, 2025]:
                for effect in levels:
                    pre = year < 2022
                    rows.append(
                        dict(
                            year=year,
                            effect=effect,
                            estimand="finite_difference_ACRT",
                            estimate=0.0 if pre else levels[effect][year - 2023],
                            p_value=0.5 if pre else 0.00001,
                        )
                    )
            pd.DataFrame(rows).to_csv(folder / "estimates.csv", index=False)
            with patch(
                "search_paper_batch.engine.run_family",
                return_value=dict(retained=["case"], counts={"retained": 1}, family=0),
            ):
                batch.run_family([], dest)
            t = pd.read_csv(folder / "estimates.csv")
            cell = t[(t.year == 2023) & (t.effect == "total")].iloc[0]
            self.assertAlmostEqual(cell.p_bonferroni_search, 0.06)
            self.assertAlmostEqual(cell.p_bonferroni_cumulative, 0.22)
            self.assertFalse(
                json.loads((folder / "adjusted_flags.json").read_text())[
                    "initial_significant_decline"
                ]
            )

    def test_300_report_counts_and_paper_only_empty_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior, _ = self.fixture(root)
            dest = root / "audit/results/new"
            planner.make_plan(dest, prior)
            write_json(
                dest / "progress.json",
                dict(
                    status="completed",
                    attempted=300,
                    counts={"initial_2023": 300},
                    retained=[],
                ),
            )
            with (
                patch("report_paper_search.ROOT", root),
                patch("report_paper_search.subprocess.run"),
            ):
                summary = report.report(dest)
            self.assertEqual(summary["attempted"], 300)
            self.assertEqual(summary["family_size"], 6000)
            self.assertEqual(summary["cumulative_retained"], 207)
            md = (root / "audit/Paper_Search_801_1100_ko.md").read_text()
            self.assertIn("300개×20개", md)
            self.assertIn("6,000", md)
            with patch("report_focused_history.subprocess.run"):
                s = focused.render(
                    root / "focus",
                    roots={"paper": dest},
                    batch=dest,
                    stem="Paper_Only_1100",
                )
            self.assertEqual(s["focused_count"], 0)
            self.assertEqual(set(s["source_roots"]), {"paper"})
            self.assertEqual(len(pd.read_csv(root / "focus/selected_cases.csv")), 0)
            text = (root / "focus/Paper_Only_1100_ko.md").read_text()
            self.assertIn("300/300", text)
            self.assertIn("22,000", text)
            self.assertNotIn("### 이전 방식", text)


if __name__ == "__main__":
    unittest.main()
