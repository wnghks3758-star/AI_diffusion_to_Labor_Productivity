import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from labor_did.io import write_json
from paper_did.config import PaperConfig
from report_paper_search import checkpoint, report
from search_paper_more import (
    PREVIOUS,
    effective_fingerprint,
    make_plan,
    run_family,
)


class AdditionalSearchTests(unittest.TestCase):
    def test_followup_report_uses_its_own_paths_and_cumulative_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / "audit/results/paper_search_201_400"
            make_plan(dest)
            write_json(
                dest / "progress.json",
                dict(
                    status="completed",
                    attempted=200,
                    counts={"initial_2023": 200},
                    retained=[],
                ),
            )
            original = root / "audit/Paper_Search_200_ko.md"
            original.write_text("preserve previous report")
            with (
                patch("report_paper_search.ROOT", root),
                patch("report_paper_search.subprocess.run"),
            ):
                summary = report(dest)
            self.assertEqual(original.read_text(), "preserve previous report")
            self.assertTrue((root / "audit/Paper_Search_201_400_ko.md").exists())
            self.assertEqual(summary["cumulative_attempted"], 400)
            self.assertEqual(summary["cumulative_retained"], 5)
            self.assertTrue(
                (
                    root
                    / "audit/exports/paper_search_201_400/Paper_Search_201_400_Results.zip"
                ).exists()
            )

    def test_checkpoint_ignores_incomplete_family_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "batch"
            dest.mkdir()
            write_json(dest / "plan.json", {})
            write_json(dest / "progress.json", dict(attempted=5, retained=[]))
            other = dest / "retained/pending"
            other.mkdir(parents=True)
            (other / "estimates.csv").write_text("not yet checkpointed")
            archive = checkpoint(dest)
            with zipfile.ZipFile(archive) as z:
                self.assertFalse(any(n.startswith("retained/") for n in z.namelist()))

    def test_additional_plan_has_200_unique_new_effective_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(Path(tmp))
            entries = plan["entries"]
            old = json.loads((PREVIOUS / "plan.json").read_text())
            old_keys = {effective_fingerprint(e["config"]) for e in old["entries"]}
            new_keys = {effective_fingerprint(e["config"]) for e in entries}
            self.assertEqual(len(new_keys), 200)
            self.assertFalse(old_keys & new_keys)
            self.assertEqual([entries[0]["number"], entries[-1]["number"]], [201, 400])
            self.assertEqual(len({e["family"] for e in entries}), 40)
            self.assertTrue(
                all(e["config"]["covariate_time"] == "baseline" for e in entries)
            )
            self.assertTrue(all(e["config"]["diff_step"] == 0.0005 for e in entries))
            self.assertEqual(plan["cumulative_correction_family_size"], 8000)

    def test_inactive_firm_lists_do_not_create_effectively_new_settings(self):
        a = PaperConfig(include_firm_controls=False).to_dict()
        b = {**a, "mu_firm_features": ["firm_rnd_ratio"], "label": "different"}
        self.assertEqual(effective_fingerprint(a), effective_fingerprint(b))

    def test_cumulative_correction_uses_planned_8000_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "retained/case"
            folder.mkdir(parents=True)
            t = pd.DataFrame(
                dict(
                    year=[2023, 2023],
                    estimand=["finite_difference_ACRT", "ATET"],
                    effect=["total", "total"],
                    estimate=[-0.1, -0.1],
                    p_value=[0.00001, 0.001],
                )
            )
            t.to_csv(folder / "estimates.csv", index=False)
            with patch(
                "search_paper_more.engine.run_family",
                return_value=dict(retained=["case"], counts={"retained": 1}, family=0),
            ):
                run_family([], tmp)
            result = pd.read_csv(folder / "estimates.csv")
            self.assertAlmostEqual(result.iloc[0].p_bonferroni_cumulative, 0.08)
            self.assertTrue(pd.isna(result.iloc[1].p_bonferroni_cumulative))


if __name__ == "__main__":
    unittest.main()
