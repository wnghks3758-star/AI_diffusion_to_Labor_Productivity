import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from labor_did.io import write_json
from paper_did.config import PaperConfig
from report_paper_search import checkpoint, report
from search_paper import ESTIMAND, evaluate, gate, make_plan, run_family


def annual(year, estimates=None, p_values=None):
    effects = ["total", "sme", "large", "moderation"]
    estimates = estimates or [0.0, 0.0, 0.0, 0.0]
    p_values = p_values or [0.5] * 4
    return pd.DataFrame(
        dict(
            year=year,
            estimand=ESTIMAND,
            effect=effects,
            estimate=estimates,
            p_value=p_values,
        )
    )


class SearchTests(unittest.TestCase):
    def test_empty_retained_report_preserves_counts_without_rejected_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "audit/results/paper_search_200"
            make_plan(dest)
            write_json(
                dest / "progress.json",
                dict(
                    status="completed",
                    attempted=200,
                    retained=[],
                    completed_families=list(range(40)),
                    counts={"initial_2023": 190, "pre_2021": 10},
                ),
            )
            with (
                patch("report_paper_search.ROOT", root),
                patch("report_paper_search.subprocess.run"),
            ):
                summary = report(dest)
            self.assertEqual(summary["retained"], 0)
            self.assertEqual(summary["attempted"], 200)
            self.assertTrue(pd.read_csv(dest / "retained_estimates.csv").empty)
            self.assertEqual(len(pd.read_csv(dest / "planned_settings.csv")), 200)
            self.assertTrue(checkpoint(dest).exists())

    def test_rejected_case_stops_early_and_keeps_no_estimates(self):
        for fail_year in [2020, 2023]:
            seen = []

            def compute(frame, year, config, out):
                seen.append(year)
                out.mkdir(parents=True)
                (out / "estimates.csv").write_text("temporary output")
                if year == 2023 and fail_year == 2020:
                    return (
                        annual(year, [-0.1, 0.0, 0.0, 0.0], [0.01, 0.5, 0.5, 0.5]),
                        None,
                        {},
                    )
                p = [0.01] * 4 if year == fail_year == 2020 else [0.5] * 4
                return annual(year, p_values=p), None, {}

            with (
                tempfile.TemporaryDirectory() as tmp,
                patch("search_paper.load_data", return_value=(None, "hash")),
                patch("search_paper.prepare", return_value=(None, {}, None)),
                patch("search_paper.prepare_year", return_value=None),
                patch("search_paper.compute_year", side_effect=compute),
            ):
                result = run_family(
                    [dict(config=PaperConfig().to_dict(), family=0)], tmp
                )
                self.assertEqual(seen, [2023, 2020] if fail_year == 2020 else [2023])
                self.assertEqual(sum(result["counts"].values()), 1)
                self.assertEqual(result["retained"], [])
                self.assertEqual(list(Path(tmp).rglob("*.csv")), [])
                self.assertEqual(list((Path(tmp) / ".scratch").iterdir()), [])

    def test_pre_gate_requires_all_four_and_valid_p(self):
        t = annual(2020)
        self.assertIsNone(gate(t, 2020))
        t.loc[3, "p_value"] = 0.049
        self.assertEqual(gate(t, 2020), "pre_2020")
        t.loc[3, "p_value"] = np.nan
        self.assertEqual(gate(t, 2020), "numerical_failure")
        self.assertEqual(gate(t.iloc[:3], 2020), "numerical_failure")

    def test_initial_gate_only_requires_total_negative_significant(self):
        self.assertIsNone(
            gate(annual(2023, [-0.1, 0.1, 0.2, 0.1], [0.01, 0.8, 0.8, 0.8]), 2023)
        )
        self.assertEqual(
            gate(annual(2023, [0.1, -0.1, -0.2, -0.1], [0.01] * 4), 2023),
            "initial_2023",
        )
        self.assertEqual(
            gate(annual(2023, [-0.1] * 4, [0.1] * 4), 2023), "initial_2023"
        )

    def test_full_pattern_and_moderation_direction(self):
        t = pd.concat(
            [
                annual(2020),
                annual(2021),
                annual(2023, [-0.3, -0.2, -0.4, -0.2], [0.01] * 4),
                annual(2024, [-0.1, 0.1, -0.2, -0.3], [0.01] * 4),
                annual(2025, [0.1, 0.3, 0.1, -0.2], [0.01] * 4),
            ],
            ignore_index=True,
        )
        self.assertTrue(evaluate(t)["preferred_match"])
        t.loc[t.effect.eq("moderation") & t.year.eq(2025), "estimate"] = 0.2
        self.assertFalse(evaluate(t)["pattern_match"])
        self.assertFalse(evaluate(t[t.year.ne(2021)])["complete"])

    def test_frozen_plan_individual_ablations_and_uniqueness(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            entries = plan["entries"]
            self.assertEqual(len(entries), 200)
            self.assertEqual(len({e["fingerprint"] for e in entries}), 200)
            self.assertEqual(len({e["family"] for e in entries}), 40)
            base = entries[0]["config"]
            ablation = entries[5]["config"]
            self.assertEqual(
                len(ablation["gps_features"]), len(base["gps_features"]) - 1
            )
            self.assertEqual(
                ablation["mu_industry_features"], base["mu_industry_features"]
            )
            self.assertTrue(all(e["config"]["base_year"] == 2022 for e in entries))
            with self.assertRaises(FileExistsError):
                make_plan(Path(tmp))


if __name__ == "__main__":
    unittest.main()
