"""ATET algebra and training-only GPS checks, independent of derivative aggregation."""

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from labor_did import Config
from labor_did.atet import estimate_atet_point
from labor_did.data import cluster_bandwidth
from labor_did.gps import nested_fold
from labor_did.models import fit_gps
from labor_did.score import normalized_atet_score
from labor_did.weights import (
    epanechnikov_kernel,
    get_trimmed_weights,
    select_supported_centers,
)


class AtetGpsTests(unittest.TestCase):
    def test_point_atet_exposes_ess_support_without_hiding_raw_estimate(self):
        d = np.linspace(0, 1, 20)
        frame = pd.DataFrame(
            {"ind_code": range(20), "office_ratio": d, "Y_diff": d * d}
        )
        common = (frame, 0.6, 0.4, 0.5, np.full(20, 0.5), np.ones(20), np.zeros(20))
        low = estimate_atet_point(*common, Config(ess=0, trim_threshold=1))
        high = estimate_atet_point(*common, Config(ess=100, trim_threshold=1))
        self.assertTrue(low["supported"])
        self.assertFalse(high["supported"])
        self.assertEqual(low["atet"], high["atet"])

    def test_atet_is_two_hajek_residual_means(self):
        r = np.array([0.2, -0.1, 0.4, 1.2])
        a = np.array([1.0, 2.0, 1.0, 0.0])
        b = np.array([0.0, 2.0, 3.0, 4.0])
        expected = (a @ r) / a.sum() - (b @ r) / b.sum()
        actual, _, _ = normalized_atet_score(r, a, b)
        self.assertAlmostEqual(actual, expected)
        # R's separately normalized kernels and the old extra ratio normalization cancel.
        self.assertAlmostEqual(actual, normalized_atet_score(r, a / 17, b / 39)[0])

    def test_two_arm_trimming_and_compressed_scan_agree(self):
        frame = pd.DataFrame(
            {
                "ind_code": np.repeat(range(10), 2),
                "office_ratio": np.repeat(np.linspace(0, 2, 10), 2),
                "D_pred": np.repeat(np.linspace(0, 1, 10), 2),
                "sigma_pred": 0.6,
            }
        )
        grid = np.linspace(0.1, 1.9, 19)
        for mode in ["comparison", "both"]:
            selected, table = select_supported_centers(
                frame, grid, 0.7, 0.2, min_ess=1, trim_threshold=0.16, trim_mode=mode
            )
            expected = []
            for d, row in zip(grid, table.itertuples()):
                _, _, info = get_trimmed_weights(
                    frame,
                    d,
                    d - 0.2,
                    frame.D_pred,
                    frame.sigma_pred,
                    0.7,
                    trim_threshold=0.16,
                    trim_mode=mode,
                )
                self.assertAlmostEqual(info["ess_d"], row.ess_d, places=10)
                self.assertAlmostEqual(info["ess_dp"], row.ess_dp, places=10)
                self.assertEqual(info["n_trimmed"], row.n_trimmed)
                if min(info["ess_d"], info["ess_dp"]) > 1:
                    expected.append(d)
            np.testing.assert_allclose(selected, expected)

    def test_np_kernel_conversion_matches_c_source(self):
        d = np.linspace(-1, 1, 101)
        h = 0.2
        z = d / h
        expected = (
            np.where(
                z * z < 5, 0.33541019662496845446 - 0.067082039324993690892 * z * z, 0
            )
            / h
        )
        actual = epanechnikov_kernel(d, 0, np.sqrt(5) * h)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        frame = pd.DataFrame({"ind_code": range(6), "office_ratio": np.arange(6)})
        np.testing.assert_allclose(
            cluster_bandwidth(frame, Config(kernel_convention="np")),
            np.sqrt(5) * cluster_bandwidth(frame, Config()),
        )

    def test_nested_gps_ignores_outer_test_doses_in_both_models(self):
        rng = np.random.default_rng(2)
        X = pd.DataFrame({"x": rng.normal(size=32), "z": rng.normal(size=32)})
        y = X.x.to_numpy() + rng.normal(scale=0.4, size=32)
        train = np.arange(24)
        test = np.arange(24, 32)
        labels = np.repeat("All", 32)
        for family in ["linear", "random_forest"]:
            c = Config(
                gps_model=family,
                gps_fit_scope="nested",
                inner_splits=3,
                gps_params={"n_estimators": 12, "max_depth": 2, "min_samples_leaf": 4},
            )
            original = nested_fold(X, y, labels, train, test, c)
            changed = y.copy()
            changed[test] += 1000
            perturbed = nested_fold(X, changed, labels, train, test, c)
            np.testing.assert_array_equal(original[0], perturbed[0])
            np.testing.assert_array_equal(original[1], perturbed[1])

    def test_rf_cache_and_mean_variance_predictions_are_finite(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            {
                "ind_code": range(24),
                "ind_refined": "All",
                "office_ratio": rng.normal(size=24),
                "x": rng.normal(size=24),
            }
        )
        with TemporaryDirectory() as temp:
            c = Config(
                gps_model="random_forest",
                gps_features=["x"],
                include_industry_dummies=False,
                n_splits=3,
                n_repeats=1,
                inner_splits=2,
                cache_dir=temp,
                gps_params={"n_estimators": 12, "max_depth": 2, "min_samples_leaf": 3},
            )
            mean, sigma, audit = fit_gps(frame, c)
            self.assertTrue(np.isfinite(mean).all() and np.all(sigma > 0))
            self.assertEqual(audit["fit_scope"], "nested")
            self.assertFalse(audit["cache_hit"])
            frame["Y_diff"] = 9999  # Outcomes are never GPS inputs.
            m2, s2, a2 = fit_gps(frame, c)
            self.assertTrue(a2["cache_hit"])
            np.testing.assert_array_equal(mean, m2)
            np.testing.assert_array_equal(sigma, s2)


if __name__ == "__main__":
    unittest.main()
