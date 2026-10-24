import tempfile
import unittest
from pathlib import Path

import pandas as pd

import report_focused_history as focused
import search_paper_fourth as fourth
from labor_did.io import write_json
from paper_did.config import PaperConfig


def example():
    rows = []
    for year in focused.YEARS:
        for effect in focused.EFFECTS:
            estimate, p = 0.0, 0.5
            if year == 2023 and effect == "total":
                estimate, p = -0.2, 0.01
            if year == 2025 and effect == "sme":
                estimate, p = 0.1, 0.3
            if year == 2025 and effect == "moderation":
                estimate, p = -0.3, 0.02
            rows.append(
                dict(
                    year=year,
                    effect=effect,
                    estimate=estimate,
                    p_value=p,
                    estimand="finite_difference_ACRT",
                )
            )
    return pd.DataFrame(rows)


class FocusedHistoryTests(unittest.TestCase):
    def test_sme_positive_need_not_be_significant(self):
        flags = focused.selection(example())
        self.assertTrue(flags["selected"])
        self.assertFalse(flags["sme2025_positive_significant"])

    def test_requires_negative_moderation_in_2025_and_both_pre_years(self):
        for mask, column, value in [
            (lambda t: (t.year == 2025) & (t.effect == "moderation"), "estimate", 0.3),
            (lambda t: (t.year == 2025) & (t.effect == "moderation"), "p_value", 0.05),
            (lambda t: (t.year == 2020) & (t.effect == "large"), "p_value", 0.01),
            (lambda t: (t.year == 2021) & (t.effect == "moderation"), "p_value", 0.01),
            (lambda t: (t.year == 2023) & (t.effect == "total"), "p_value", 0.2),
        ]:
            t = example()
            t.loc[mask(t), column] = value
            self.assertFalse(focused.selection(t)["selected"])

    def test_incomplete_or_duplicate_cells_rejected_but_atet_ignored(self):
        t = example()
        self.assertIsNone(focused.selection(t.iloc[:-1]))
        self.assertIsNone(focused.selection(pd.concat([t, t.iloc[:1]])))
        self.assertTrue(
            focused.selection(pd.concat([t, t.assign(estimand="ATET")]))["selected"]
        )

    def test_archive_copies_deduplicated_and_sources_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["case", "copy"]:
                folder = root / name
                folder.mkdir()
                example().to_csv(folder / "estimates.csv", index=False)
                write_json(folder / "config.json", {"label": name, "seed": 1})
            cases = focused.collect({"paper": root})
            self.assertEqual(len(cases), 1)
            self.assertEqual(len(cases[0]["aliases"]), 1)

    def test_new_plan_200_unique_and_no_overlap_with_prior_600(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            priors = []
            all_entries = []
            for batch in range(3):
                folder = root / str(batch)
                entries = [
                    dict(
                        number=n,
                        config=PaperConfig(seed=n, label=f"paper_search_{n}").to_dict(),
                    )
                    for n in range(1 + batch * 200, 201 + batch * 200)
                ]
                write_json(
                    folder / "plan.json",
                    dict(
                        entries=entries,
                        hypothesis_estimand="finite_difference_ACRT",
                        alpha=0.05,
                        retention={},
                        criteria={},
                    ),
                )
                priors.append(folder)
                all_entries.extend(entries)
            write_json(priors[-1] / "summary.json", dict(cumulative_attempted=600))
            plan = fourth.make_plan(root / "new", priors)
            old = {
                fourth.runner.effective_fingerprint(e["config"]) for e in all_entries
            }
            new = {e["effective_fingerprint"] for e in plan["entries"]}
            self.assertEqual(len(new), 200)
            self.assertFalse(old & new)
            self.assertEqual(plan["cumulative_correction_family_size"], 16000)
            self.assertEqual(
                [plan["entries"][0]["number"], plan["entries"][-1]["number"]],
                [601, 800],
            )
            with self.assertRaises(FileExistsError):
                fourth.make_plan(root / "new", priors)


if __name__ == "__main__":
    unittest.main()
