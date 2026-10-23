"""Model switches, grouped nuisance fits, exact compression and real checkpoints."""

import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from test_workflow import synthetic_panel

from labor_did.support import averaging_weights
from paper_did.config import PaperConfig
from paper_did.estimation import pair_scores
from paper_did.models import (
    fit_nuisance,
    fit_regression,
    predict_controls,
)
from paper_did.numerics import (
    atet_score,
    gaussian_log_ratio,
    np_kernel,
    paper_bandwidth,
)
from paper_did.pipeline import prepare, run
from paper_did.support import averaging_config, computable_grid


def frame_fixture():
    rng = np.random.default_rng(12)
    records = []
    for j in range(32):
        x = rng.normal()
        D = 0.4 * x + rng.normal()
        for i in range(4):
            z = rng.normal()
            records.append(
                dict(
                    ind_code=j,
                    ind_refined="All",
                    firm_code=j * 4 + i,
                    Year=2023,
                    office_ratio=D,
                    x=x,
                    z=z if j + i else np.nan,
                    large_firm=i % 2,
                    Y_diff=0.3 * D + x + 0.2 * z + rng.normal(scale=0.2),
                )
            )
    return pd.DataFrame(records)


def small_config(cache):
    return PaperConfig(
        cache_dir=str(cache),
        gps_features=["x"],
        mu_industry_features=["x"],
        mu_firm_features=["z"],
        include_industry_dummies=False,
        industry_scheme="none",
        n_splits=2,
        inner_splits=2,
        gps_model="linear",
        mu_model="xgboost",
        mu_params={"n_estimators": 8, "max_depth": 2},
        diff_step=0.1,
        n_bootstrap=29,
        trim=1.0,
        acrt_support="computable",
    )


class PaperWorkflow(unittest.TestCase):
    def test_all_model_switches_produce_predictions_and_impute_training_only(self):
        f = frame_fixture()
        with TemporaryDirectory() as tmp:
            c = small_config(tmp)
            for kind in [
                "spline",
                "ridge",
                "lasso",
                "linear",
                "random_forest",
                "xgboost",
            ]:
                params = (
                    {"alpha": 0.01}
                    if kind in ["spline", "ridge", "lasso"]
                    else (
                        {"n_estimators": 5, "max_depth": 2}
                        if kind in ["random_forest", "xgboost"]
                        else {}
                    )
                )
                train = np.arange(80)
                test = np.arange(80, len(f))
                model, _ = fit_regression(
                    f[["x", "z"]].iloc[train],
                    f.Y_diff.iloc[train],
                    f.ind_code.iloc[train],
                    kind,
                    params,
                    c,
                )
                self.assertTrue(
                    np.isfinite(model.predict(f[["x", "z"]].iloc[test])).all()
                )
                self.assertAlmostEqual(
                    model.named_steps["impute"].statistics_[1], f.z.iloc[train].median()
                )

    def test_fold_exclusion_covariance_and_computability_match_firm_calculation(self):
        f = frame_fixture()
        with TemporaryDirectory() as tmp:
            c = small_config(tmp)
            fit = fit_nuisance(f, c)
            for fold in fit["audit"]["folds"]:
                self.assertFalse(
                    set(fold["train_industry_positions"])
                    & set(fold["test_industry_positions"])
                )
            self.assertTrue(fit_nuisance(f, c)["audit"]["cache_hit"])
            h = paper_bandwidth(f.groupby("ind_code").office_ratio.first())
            grid = np.linspace(f.office_ratio.min(), f.office_ratio.max(), 11)
            flags, _ = computable_grid(f, grid, h, fit, replace(c, trim=0.1))
            for scope, sub in {
                "total": f,
                "sme": f[f.large_firm.eq(0)],
                "large": f[f.large_firm.eq(1)],
            }.items():
                for d, flag in zip(grid, flags[scope]):
                    ix = sub.index.to_numpy()
                    try:
                        atet_score(
                            sub.Y_diff,
                            np.zeros(len(sub)),
                            np_kernel(sub.office_ratio, d, h),
                            np_kernel(sub.office_ratio, d - c.diff_step, h),
                            gaussian_log_ratio(
                                d, d - c.diff_step, fit["mean"][ix], fit["variance"][ix]
                            ),
                            sub.ind_code,
                            trim=0.1,
                        )
                        expected = True
                    except ValueError:
                        expected = False
                    self.assertEqual(bool(flag), expected)
            _, pred = next(predict_controls(f, np.array([0.0]), fit, c))
            results, v = pair_scores(f, 0.5, 0.0, h, pred[0], fit, c)
            np.testing.assert_allclose(v["moderation"], v["large"] - v["sme"])
            self.assertAlmostEqual(
                results["moderation"]["se"], np.linalg.norm(v["large"] - v["sme"])
            )

    def test_interpolate_and_exclude_apply_to_gaps_without_extrapolation(self):
        f = pd.DataFrame(
            dict(
                ind_code=range(4),
                firm_code=range(4),
                office_ratio=[-1.0, -0.5, 0.0, 0.5],
            )
        )
        full = np.array([-1.0, -0.5, 0.0, 0.5])
        supported = np.array([-0.5, 0.5])
        c = small_config("unused")
        _, bridge = averaging_weights(
            f,
            supported,
            full,
            0.8,
            averaging_config(replace(c, gap_policy="interpolate")),
        )
        _, drop = averaging_weights(
            f, supported, full, 0.8, averaging_config(replace(c, gap_policy="exclude"))
        )
        self.assertEqual(bridge["n_used_firms"], 3)
        self.assertEqual(drop["n_used_firms"], 2)
        self.assertEqual(bridge["n_in_range_firms"], 3)  # no extrapolation for -1
        self.assertEqual(bridge["n_gap_firms"], 1)

    def test_validation_outcomes_do_not_enter_their_mu_training(self):
        f = frame_fixture()
        with TemporaryDirectory() as tmp:
            c = small_config(tmp)
            original = fit_nuisance(f, c)
            heldout = original["models"][0]["test"]
            changed = f.copy()
            changed.loc[heldout, "Y_diff"] += 100
            other = fit_nuisance(changed, c)
            first = next(predict_controls(f, np.array([0.0]), original, c))[1][0]
            second = next(predict_controls(changed, np.array([0.0]), other, c))[1][0]
            np.testing.assert_allclose(first[heldout], second[heldout])

    def test_full_run_resumes_and_firm_controls_toggle(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            raw = synthetic_panel()
            raw.to_excel(root / "data.xlsx", index=False)
            c = PaperConfig(
                data_path=str(root / "data.xlsx"),
                cache_dir=str(root / "cache"),
                years=[2023],
                industry_scheme="none",
                gps_features=["ln_ind_avg_pay", "ln_ind_ast"],
                mu_industry_features=["ln_ind_avg_pay"],
                mu_firm_features=["ln_firm_ast"],
                include_industry_dummies=False,
                gps_model="linear",
                mu_model="xgboost",
                mu_params={"n_estimators": 5, "max_depth": 2},
                n_splits=2,
                inner_splits=2,
                diff_step=0.1,
                n_bootstrap=29,
                trim=1.0,
                acrt_support="computable",
            )
            panel, sample, _ = prepare(raw, c)
            self.assertEqual(sample["analysis_firms"], 128)
            baseline = panel[panel.Year.eq(2022)].set_index("firm_code")
            self.assertTrue(
                (
                    panel[panel.Year.eq(2023)].set_index("firm_code").ln_firm_ast
                    == baseline.ln_firm_ast
                ).all()
            )
            without = replace(c, include_firm_controls=False)
            self.assertNotIn("ln_firm_ast", without.mu_features)
            self.assertNotEqual(without.fingerprint, c.fingerprint)
            table = run(c, root / "out")
            self.assertEqual(len(table), 12)
            self.assertTrue(table.status.eq("completed").all())
            self.assertTrue(np.isfinite(table.estimate).all())
            self.assertTrue((root / "out/report.html").exists())
            with patch(
                "paper_did.pipeline.compute_year",
                side_effect=AssertionError("must resume"),
            ):
                again = run(c, root / "out")
            np.testing.assert_allclose(table.estimate, again.estimate)
            audit = json.loads((root / "out/audit.json").read_text())
            self.assertIsNone(audit["years"][0]["ess_filter"])
            self.assertEqual(
                audit["years"][0]["gps"]["variance_source"],
                "outer validation industry residual MSE",
            )


if __name__ == "__main__":
    unittest.main()
