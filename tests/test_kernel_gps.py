"""Method-2 cross-fitting, direct-score normalization and overlap failures."""

import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from labor_did import Config
from labor_did.kernel_gps import atet_from_kernel_scores, fit_kernel_gps
from labor_did.weights import epanechnikov_kernel


class KernelGpsTests(unittest.TestCase):
    def test_held_out_dose_never_enters_training_and_one_point_shape(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            dict(
                ind_code=np.arange(40),
                ind_refined="All",
                x=rng.normal(size=40),
                office_ratio=rng.normal(size=40),
            )
        )
        with TemporaryDirectory() as tmp:
            c = Config(
                gps_features=["x"],
                include_industry_dummies=False,
                n_splits=4,
                n_repeats=1,
                gps_params=dict(n_estimators=20, min_samples_leaf=3),
                cache_dir=tmp,
            )
            for mode in ["shared", "separate"]:
                first = fit_kernel_gps(frame, [0.0], 1.0, c, mode=mode)
                self.assertEqual(first.predictions.shape, (1, 40, 1))
                self.assertTrue((first.predictions >= 0).all())
                fold = first.audit["folds"][0]
                test = fold["test_positions"]
                self.assertFalse(set(test) & set(fold["train_positions"]))
                changed = frame.copy()
                changed.loc[test, "office_ratio"] += 100
                second = fit_kernel_gps(changed, [0.0], 1.0, c, mode=mode)
                np.testing.assert_allclose(
                    first.predictions[:, test], second.predictions[:, test]
                )
                cached = fit_kernel_gps(frame, [0.0], 1.0, c, mode=mode)
                self.assertTrue(cached.audit["cache_hit"])

    def test_score_matches_author_formula_and_cluster_se(self):
        D = np.linspace(-1, 1, 30)
        frame = pd.DataFrame(
            dict(ind_code=np.repeat(np.arange(15), 2), office_ratio=D, Y_diff=D**2 + D)
        )
        # Industry-constant scores and arbitrary constants per evaluation point cancel.
        pt = np.repeat(np.linspace(0.2, 0.6, 15), 2)
        pc = np.repeat(np.linspace(0.6, 0.3, 15), 2)
        mu = np.repeat(np.linspace(-0.1, 0.2, 15), 2)
        got = atet_from_kernel_scores(
            frame, 0.4, -0.4, 0.7, pt, pc, mu, trim=1, min_ess=0
        )
        a = epanechnikov_kernel(D, 0.4, 0.7)
        a /= a.sum()
        b = epanechnikov_kernel(D, -0.4, 0.7) * pt / pc
        b /= b.sum()
        r = frame.Y_diff.to_numpy() - mu
        theta = (a - b) @ r
        self.assertAlmostEqual(got["atet"], theta)
        cluster = (a * (r - theta) - b * r).reshape(15, 2).sum(axis=1)
        self.assertAlmostEqual(
            got["se_author_cluster"], np.sqrt((cluster**2).sum() * 15 / 14)
        )
        scaled = atet_from_kernel_scores(
            frame, 0.4, -0.4, 0.7, pt / 17, pc / 39, mu, trim=1, min_ess=0
        )
        self.assertAlmostEqual(scaled["atet"], theta)
        self.assertGreater(got["max_industry_weight"], got["max_firm_weight"])
        fail = pc.copy()
        fail[0] = 0
        with self.assertRaisesRegex(ValueError, "active support"):
            atet_from_kernel_scores(frame, 0.4, -0.4, 0.7, pt, fail, mu)


if __name__ == "__main__":
    unittest.main()
