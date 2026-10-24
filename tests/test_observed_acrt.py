import tempfile
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from test_paper_workflow import frame_fixture, small_config

from observed_acrt import (
    aggregate_results,
    direct_weights,
    fit_kernel_oof,
    kernel_pair_scores,
)
from paper_did.models import industry_folds
from paper_did.numerics import atet_score, np_kernel


class ObservedACRTTests(unittest.TestCase):
    def test_grid_interpolation_and_direct_are_equal_for_linear_curves(self):
        frame = pd.DataFrame(dict(ind_code=[0, 0, 0, 1], firm_code=[0, 1, 2, 3], office_ratio=[.1, .1, .1, .9], large_firm=[0, 1, 0, 1]))
        c = replace(small_config("unused"), diff_step=.1)
        grid = np.array([-.5, 0., .5, 1., 1.5])
        results = {}
        for d in np.unique(np.r_[grid, frame.office_ratio]):
            rs = {s: dict(estimate=c.diff_step*(2+3*d), status="completed") for s in ["total", "sme", "large"]}
            vs = {s: c.diff_step*np.array([.01*d, .02*d]) for s in rs}
            results[(float(d), float(d-c.diff_step))] = rs, vs
        targets = {s: set(frame.office_ratio) for s in ["total", "sme", "large"]}
        direct, vd = aggregate_results(frame, results, c, 2023, .5, targets)
        interpolated, vi = aggregate_results(frame, results, c, 2023, .5, targets, mode="grid", grid=grid, supported={s:grid for s in targets})
        np.testing.assert_allclose(direct[["estimate", "se"]], interpolated[["estimate", "se"]], atol=1e-12)
        np.testing.assert_allclose(vd, vi, atol=1e-12)
        self.assertAlmostEqual(direct.query("effect=='total'").iloc[0].estimate, 2.9)

    def test_direct_weights_are_firm_weighted_and_strict_failures_propagate(self):
        frame = pd.DataFrame(dict(ind_code=[0, 0, 0, 1], firm_code=[0, 1, 2, 3], office_ratio=[0., 0., 0., 1.], large_firm=[0, 1, 0, 1]))
        weights, audit = direct_weights(frame, "office_ratio")
        self.assertEqual(weights, {0.: .75, 1.: .25})
        c = replace(small_config("unused"), diff_step=.1)
        results = {}
        for d in [0., 1.]:
            rs = {s: dict(estimate=d*.1, status="completed") for s in ["total", "sme", "large"]}
            vs = {s: np.array([.01, -.01]) for s in rs}
            results[(d, d-.1)] = rs, vs
        targets = {s: [0., 1.] for s in ["total", "sme", "large"]}
        t, _ = aggregate_results(frame, results, c, 2023, .5, targets)
        self.assertAlmostEqual(t.query("effect=='total'").iloc[0].estimate, .25)
        results[(1., .9)][0]["total"] = dict(estimate=np.nan, status="failed")
        strict, _ = aggregate_results(frame, results, c, 2023, .5, targets)
        self.assertTrue(np.isnan(strict.query("effect=='total'").iloc[0].estimate))
        selected = {s: [0.] for s in targets}
        supported, _ = aggregate_results(frame, results, c, 2023, .5, selected)
        self.assertAlmostEqual(supported.query("effect=='total'").iloc[0].coverage, .75)

    def test_kernel_oof_uses_own_role_folds_and_excludes_validation_dose(self):
        frame = frame_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            c = replace(small_config(tmp), gps_model="random_forest", gps_params=dict(n_estimators=8, max_depth=2, min_samples_leaf=3),
                        gps_n_splits=4, gps_fold_seed=107)
            first = fit_kernel_oof(frame, np.array([-.1, .1]), .8, c, workers=1)
            train, test = industry_folds(np.arange(32), 4, 107)[0]
            changed = frame.copy()
            changed.loc[changed.ind_code.isin(test), c.dose] += 10
            second = fit_kernel_oof(changed, np.array([-.1, .1]), .8, c, workers=1)
            np.testing.assert_array_equal(first["predictions"][test], second["predictions"][test])
            cached = fit_kernel_oof(frame, np.array([-.1, .1]), .8, c, workers=1)
            self.assertTrue(cached["audit"]["cache_hit"])
            self.assertTrue((first["predictions"] >= 0).all())
            self.assertEqual(first["predictions"].shape, (32, 2))
            self.assertFalse(set(train) & set(test))

    def test_kernel_score_matches_author_ratio_and_arbitrary_point_scaling(self):
        frame = frame_fixture()
        c = small_config("unused")
        n = frame.ind_code.nunique()
        kernel = dict(industries=np.arange(n), centers=np.array([-.1, .1]),
                      predictions=np.column_stack([np.linspace(.3, .7, n), np.linspace(.5, .2, n)]))
        mu = np.zeros(len(frame))
        rs, vs = kernel_pair_scores(frame, .1, -.1, 1., mu, kernel, c)
        m = frame.ind_code.to_numpy()
        lr = np.log(kernel["predictions"][m, 1]) - np.log(kernel["predictions"][m, 0])
        raw = atet_score(frame.Y_diff, mu, np_kernel(frame[c.dose], .1, 1.), np_kernel(frame[c.dose], -.1, 1.), lr, frame.ind_code, c.trim)
        self.assertAlmostEqual(rs["total"]["estimate"], raw["atet"])
        scale = dict(kernel, predictions=kernel["predictions"] * np.array([17., 31.]))
        scaled, _ = kernel_pair_scores(frame, .1, -.1, 1., mu, scale, c)
        self.assertAlmostEqual(scaled["total"]["estimate"], raw["atet"])
        self.assertAlmostEqual(np.linalg.norm(vs["total"]), rs["total"]["se"])
        bad = dict(kernel, predictions=np.zeros_like(kernel["predictions"]))
        failed, _ = kernel_pair_scores(frame, .1, -.1, 1., mu, bad, c)
        self.assertEqual(failed["total"]["status"], "failed")
        self.assertTrue(np.isnan(failed["total"]["estimate"]))


if __name__ == "__main__":
    unittest.main()
