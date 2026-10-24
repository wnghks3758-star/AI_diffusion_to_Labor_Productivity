import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import search_paper_size as planner
from labor_did.data import create_large_firm_dummy
from labor_did.io import write_json
from paper_did.config import PaperConfig
from paper_did.size import classify_size
from search_paper_more import effective_fingerprint


def panel():
    sales = {1: [10., 100., 100.], 2: [60., 60., 60.], 3: [1000., 1000., 1000.], 4: [80., 80., 80.], 5: [5., 5., 5.]}
    assets = {1: 500., 2: 200., 3: 10., 4: 100., 5: 5.}
    return pd.DataFrame([dict(firm_code=f, ind_code="a" if f < 5 else "b", Year=y,
                              sale=sales[f][y - 2020], assets_annual=assets[f])
                         for f in sales for y in [2020, 2021, 2022]])


def classes(frame, c):
    result, audit = classify_size(frame, c)
    return result.groupby("firm_code").large_firm.first().to_dict(), audit


class SizeTests(unittest.TestCase):
    def test_default_exactly_preserves_original_including_missing_and_singletons(self):
        frame = panel()
        frame.loc[frame.firm_code.eq(2), "sale"] = np.nan
        original = create_large_firm_dummy(frame)
        current, _ = classify_size(frame, PaperConfig())
        pd.testing.assert_frame_equal(current, original)
        already_classified = original.assign(extra_column=1)
        refreshed, _ = classify_size(already_classified, PaperConfig())
        pd.testing.assert_frame_equal(refreshed, already_classified)

    def test_industry_median_mean_and_annual_assets_change_membership(self):
        f = panel()
        median, _ = classes(f, PaperConfig(size_threshold="median"))
        mean, _ = classes(f, PaperConfig(size_threshold="mean"))
        asset, _ = classes(f, PaperConfig(size_variable="assets_annual"))
        self.assertEqual(median, {1: 0, 2: 0, 3: 1, 4: 1, 5: -1})
        self.assertEqual(mean, {1: 0, 2: 0, 3: 1, 4: 0, 5: -1})
        self.assertEqual(asset, {1: 1, 2: 1, 3: 0, 4: 0, 5: -1})

    def test_time_summary_and_period_are_distinct_and_fixed_across_years(self):
        f = panel()
        temporal, _ = classes(f, PaperConfig(size_time_aggregation="median"))
        early, _ = classes(f, PaperConfig(size_years=[2020]))
        late, _ = classes(f, PaperConfig(size_years=[2022]))
        self.assertEqual(temporal[1], 1)
        self.assertEqual(early[1], 0)
        self.assertEqual(late[1], 1)
        after = f[f.Year.eq(2022)].assign(Year=2025, sale=1e20)
        combined = pd.concat([f, after], ignore_index=True)
        before_map, _ = classes(f, PaperConfig(size_years=[2020]))
        result, _ = classify_size(combined, PaperConfig(size_years=[2020]))
        self.assertEqual(result.groupby("firm_code").large_firm.nunique().max(), 1)
        self.assertEqual(result.groupby("firm_code").large_firm.first().to_dict(), before_map)

    def test_missing_stays_in_total_as_unclassified_without_imputation(self):
        f = panel()
        f.loc[f.firm_code.eq(2) & f.Year.eq(2022), "assets_annual"] = np.nan
        result, audit = classify_size(f, PaperConfig(size_variable="assets_annual", size_years=[2022], size_missing_policy="unclassified"))
        self.assertEqual(len(result), len(f))
        self.assertTrue(result.loc[result.firm_code.eq(2), "large_firm"].eq(-1).all())
        self.assertEqual(audit["missing_size_firms"], 1)
        self.assertEqual(audit["unclassified_firms"], 2)

    def test_post_treatment_duplicate_and_absent_years_rejected(self):
        for kwargs in [dict(size_years=[2023]), dict(size_years=[2020, 2020]), dict(size_variable="ln_firm_prod")]:
            with self.assertRaises(ValueError):
                PaperConfig(**kwargs)
        with self.assertRaises(ValueError):
            classify_size(panel(), PaperConfig(size_years=[2019]))

    def test_200_plan_balances_size_definitions_and_varies_models_and_covariates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior, old = [], []
            start = 1
            for i, count in enumerate([200, 200, 200, 200, 300]):
                p = root / str(i)
                entries = [dict(number=n, config=PaperConfig(seed=n, label=f"paper_search_{n}").to_dict()) for n in range(start, start + count)]
                write_json(p / "plan.json", dict(entries=entries, hypothesis_estimand="finite_difference_ACRT", alpha=.05,
                           retention={}, criteria={}, focused_filter={}, source_sha256="previous"))
                prior.append(p)
                old += entries
                start += count
            write_json(prior[-1] / "summary.json", dict(cumulative_attempted=1100, retained=71, cumulative_retained=278, pattern_matches=0))
            plan = planner.make_plan(root / "new", prior)
            configs = [e["config"] for e in plan["entries"]]
            self.assertEqual(len(configs), 200)
            self.assertEqual(Counter(c["size_variable"] for c in configs), {"sale": 100, "assets_annual": 100})
            self.assertEqual(Counter(c["size_threshold"] for c in configs), {"median": 100, "mean": 100})
            self.assertEqual(set(Counter(tuple(c["size_years"]) for c in configs).values()), {40})
            self.assertEqual(sum(c["size_time_aggregation"] == "median" for c in configs), 40)
            self.assertEqual(len({c["gps_model"] for c in configs}), 6)
            self.assertEqual(len({c["mu_model"] for c in configs}), 6)
            keys = {effective_fingerprint(c) for c in configs}
            self.assertEqual(len(keys), 200)
            self.assertFalse(keys & {effective_fingerprint(e["config"]) for e in old})
            self.assertEqual(plan["cumulative_correction_family_size"], 26000)


if __name__ == "__main__":
    unittest.main()
