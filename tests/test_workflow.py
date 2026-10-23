"""Exercise a real fit, checkpoint recovery and CLI grid failure accounting."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from labor_did import Config, run
from labor_did.config import INDUSTRY_CONTROLS
from labor_did.models import fit_mu
from labor_did.pipeline import holm
from run import main


def synthetic_panel():
    rng = np.random.default_rng(42)
    records = []
    for industry in range(32):
        controls = dict(zip(INDUSTRY_CONTROLS, rng.normal(size=len(INDUSTRY_CONTROLS))))
        for firm in range(4):
            for year in [2022, 2023]:
                records.append(
                    {
                        "firm_code": industry * 4 + firm,
                        "Year": year,
                        "ind_code": 100 + industry,
                        "sale": firm + 1.0,
                        "office_ratio": 0.02 + industry / 80,
                        "ln_firm_prod": 0.2 * firm
                        + rng.normal(scale=0.1)
                        - 0.03 * (year - 2022) * industry,
                        "firm_rnd_ratio": 0.1,
                        "ln_firm_ast": firm + 1.0,
                        "ln_firm_avg_pay": firm / 3,
                        **controls,
                    }
                )
    return pd.DataFrame(records)


class Workflow(unittest.TestCase):
    def test_year_fixed_parameters_skip_random_search(self):
        with TemporaryDirectory() as temp:
            frame = synthetic_panel().query("Year == 2023").copy()
            frame["ind_refined"] = "All"
            frame["Y_diff"] = frame.ln_firm_prod
            config = Config(
                cache_dir=temp,
                mu_features=["firm_rnd_ratio"],
                include_industry_dummies=False,
                n_splits=2,
                n_repeats=1,
                mu_search_n_iter=150,
                mu_params_by_year={"2023": {"n_estimators": 3, "max_depth": 1}},
            )
            with patch(
                "labor_did.models.RandomizedSearchCV",
                side_effect=AssertionError("must not tune"),
            ):
                _, manifest, _ = fit_mu(frame, config)
            self.assertEqual(manifest["settings"]["mu_search_n_iter"], 0)
            self.assertEqual(manifest["settings"]["mu_params"]["n_estimators"], 3)

    def test_real_fit_resume_and_prediction_cache(self):
        with TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = root / "data.xlsx"
            synthetic_panel().to_excel(data, index=False)
            config = Config(
                data_path=str(data),
                cache_dir=str(root / "cache"),
                years=[2023],
                n_splits=2,
                inner_splits=2,
                n_repeats=2,
                n_bootstrap=49,
                mu_search_n_iter=0,
                mu_params={"n_estimators": 8, "max_depth": 2},
                industry_scheme="none",
                include_industry_dummies=False,
                diff_step=0.05,
                ess=0.5,
                trim_threshold=0.5,
            )
            result = run(config, root / "result")
            self.assertTrue(np.isfinite(result.estimate).all())
            self.assertEqual(
                set(result.effect),
                {
                    "total",
                    "sme",
                    "large",
                    "common_sme",
                    "common_large",
                    "moderation_means",
                    "moderation_common",
                    "moderation",
                },
            )
            summary = pd.read_csv(root / "result/summary.csv")
            self.assertEqual(len(summary), 4)
            self.assertTrue((root / "result/event_study.png").stat().st_size > 1000)
            # Completed outputs require no nuisance refit.
            with patch(
                "labor_did.pipeline.estimate_year",
                side_effect=AssertionError("must resume"),
            ):
                repeated = run(config, root / "result")
                np.testing.assert_allclose(repeated.estimate, result.estimate)
                # Recover an interrupted finalization from annual checkpoints.
                (root / "result/estimates.csv").unlink()
                recovered = run(config, root / "result")
                np.testing.assert_allclose(recovered.estimate, result.estimate)

    def test_batch_preserves_failed_attempt_and_adjusts_full_planned_family(self):
        with TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            table = pd.DataFrame(
                {"year": [2023], "effect": ["total"], "p_value": [0.001]}
            )

            def fake_run(config, output):
                if config.bandwidth_scale == 0.5:
                    raise ValueError("deliberate failure")
                return table

            with patch("run.run", side_effect=fake_run):
                status = main(["--grid", "bandwidth_scale=[0.4,0.5]", "--out", temp])
            self.assertEqual(status, 1)
            manifest = json.loads((Path(temp) / "attempts.json").read_text())
            self.assertEqual(
                [a["status"] for a in manifest["attempts"]], ["completed", "failed"]
            )
            comparison = pd.read_csv(Path(temp) / "comparison.csv")
            self.assertAlmostEqual(comparison.p_holm_batch.iloc[0], 0.001 * 2 * 5 * 4)

    def test_holm_keeps_nan_tests_in_planned_family(self):
        np.testing.assert_allclose(
            holm([0.01, np.nan, 0.03], 4), [0.04, np.nan, 0.09], equal_nan=True
        )


if __name__ == "__main__":
    unittest.main()
