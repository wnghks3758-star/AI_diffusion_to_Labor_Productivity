"""Numerical invariants: derivatives, paired inference, group separation and caching."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from labor_did import Config
from labor_did.aggregation import draw_joint
from labor_did.data import cluster_bandwidth, create_large_firm_dummy
from labor_did.models import make_folds, predict_mu
from labor_did.score import normalized_acr_score
from labor_did.weights import get_trimmed_weights, select_supported_centers


class Numerics(unittest.TestCase):
    def test_ratio_influence_matches_weight_derivative(self):
        r = np.array([0.1, 0.9, 1.3, -0.4])
        high, low = np.array([1.0, 2.0, 3.0, 0.0]), np.array([0.0, 3.0, 1.0, 4.0])
        point, influence, _ = normalized_acr_score(r, high, low, 0.02)
        for j in range(len(r)):
            direction = np.ones(len(r))
            direction[j] += 1e-6
            changed = normalized_acr_score(r, high * direction, low * direction, 0.02)[
                0
            ]
            self.assertAlmostEqual(
                (changed - point) / 1e-6, influence[j] / len(r), places=4
            )

    def test_joint_bootstrap_keeps_large_minus_sme_covariance(self):
        rng = np.random.default_rng(3)
        scores = rng.normal(size=(12, 2))
        c = Config(n_bootstrap=99, bootstrap_batch_size=7)
        draws = draw_joint(
            [1.0, 2.0, 1.0], np.column_stack([scores, scores[:, 1] - scores[:, 0]]), c
        )
        np.testing.assert_allclose(draws[:, 2], draws[:, 1] - draws[:, 0], atol=1e-13)

    def test_industry_compression_equals_firm_weight_calculation(self):
        groups = np.repeat(np.arange(10), [2, 3, 4, 3, 2, 4, 2, 3, 2, 4])
        frame = pd.DataFrame(
            {
                "ind_code": groups,
                "office_ratio": groups / 4 + 0.1,
                "D_pred": groups / 5,
                "sigma_pred": 0.7,
            }
        )
        grid = np.linspace(0.2, 2.0, 21)
        selected, diagnostics = select_supported_centers(
            frame, grid, 0.8, 0.01, min_ess=2
        )
        expected = []
        for d, row in zip(grid, diagnostics.itertuples()):
            _, _, info = get_trimmed_weights(
                frame, d, d - 0.01, frame.D_pred.values, frame.sigma_pred.values, 0.8
            )
            self.assertAlmostEqual(info["ess_d"], row.ess_d, places=10)
            self.assertAlmostEqual(info["ess_dp"], row.ess_dp, places=10)
            if min(info["ess_d"], info["ess_dp"]) > 2:
                expected.append(d)
        np.testing.assert_array_equal(selected, expected)

    def test_industry_folds_do_not_leak(self):
        frame = pd.DataFrame(
            {
                "ind_code": np.repeat(np.arange(12), 2),
                "ind_refined": np.repeat(["a"] * 6 + ["b"] * 6, 2),
            }
        )
        folds = make_folds(frame, Config(n_splits=3))
        for train, test in folds:
            self.assertFalse(
                set(frame.iloc[train].ind_code) & set(frame.iloc[test].ind_code)
            )
        np.testing.assert_array_equal(
            np.sort(np.concatenate([t for _, t in folds])), np.arange(len(frame))
        )

    def test_cached_tree_predictions_equal_direct_repeated_median(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            frame = pd.DataFrame(
                {"x": np.arange(12) / 12, "office_ratio": np.linspace(0.2, 1.3, 12)}
            )
            X = frame.to_numpy()
            y = np.sin(X[:, 0]) + X[:, 1] ** 2
            centers = np.array([0.2, 0.21, 0.22, 0.8, 0.81, 1.2])
            expected = np.empty((2, len(centers), len(frame)))
            entries = []
            for rep in range(2):
                for fold in range(2):
                    test = np.arange(fold, 12, 2)
                    train = np.setdiff1d(np.arange(12), test)
                    model = XGBRegressor(
                        n_estimators=4,
                        max_depth=2,
                        n_jobs=1,
                        tree_method="exact",
                        random_state=rep,
                    ).fit(X[train], y[train])
                    name = f"{rep}_{fold}.joblib"
                    joblib.dump({"model": model, "test": test}, folder / name)
                    entries.append({"file": name, "repetition": rep})
                    for j, d in enumerate(centers):
                        rows = X[test].copy()
                        rows[:, -1] = d
                        expected[rep, j, test] = model.predict(rows)
            c = Config(n_repeats=2, cache_dir=temp, prediction_batch_points=2)
            manifest = {
                "models": entries,
                "features": list(frame.columns),
                "cache_key": "test",
            }
            for hit in [False, True]:
                with predict_mu(frame, centers, c, folder, manifest) as predictions:
                    actual = np.concatenate(
                        [batch for _, _, batch in predictions.batches()]
                    )
                    self.assertEqual(predictions.audit["cache_hit"], hit)
                    np.testing.assert_array_equal(actual, np.median(expected, axis=0))
            self.assertFalse(list(folder.glob("mu_predictions_*")))

    def test_std_bandwidth_differs_only_when_robust_scale_is_smaller(self):
        frame = pd.DataFrame(
            {"ind_code": range(6), "office_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 10.0]}
        )
        sigma = cluster_bandwidth(frame, Config(bandwidth_statistic="sigma"))
        std = cluster_bandwidth(frame, Config(bandwidth_statistic="std"))
        self.assertGreater(std, sigma)

    def test_original_size_missing_value_convention(self):
        frame = pd.DataFrame(
            {
                "firm_code": [1, 2, 3],
                "Year": [2022] * 3,
                "ind_code": [10] * 3,
                "sale": [1.0, 2.0, np.nan],
            }
        )
        size = create_large_firm_dummy(frame).set_index("firm_code")
        self.assertEqual(size.loc[3, "large_firm"], 0)
        self.assertEqual(size.loc[2, "large_firm"], 1)

    def test_invalid_settings_fail_before_estimation(self):
        for change in [
            {"ess": float("nan")},
            {"diff_step": 0},
            {"n_splits": 1},
            {"mu_features": ["ln_firm_prod"]},
            {"bandwidth_statistic": "variance"},
        ]:
            with self.assertRaises(ValueError):
                Config(**change)
        with self.assertRaises(TypeError):
            Config.from_dict({"typo_ess": 0})


if __name__ == "__main__":
    unittest.main()
