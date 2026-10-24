"""Official frozen R score fixtures and numerical stability of paper primitives."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from paper_did.numerics import (
    atet_score,
    gaussian_log_ratio,
    paper_bandwidth,
    reference_fold_variance,
)


class PaperNumerics(unittest.TestCase):
    def test_official_r_score_fixtures_match_including_se(self):
        root = Path(__file__).resolve().parents[1] / "audit/results"
        fixture = pd.read_csv(root / "score_fixture.csv")
        expected = pd.read_csv(root / "score_r.csv").set_index("case")
        for name, g in fixture.groupby("case"):
            result = atet_score(
                g.y,
                g.mu,
                g.high,
                g.low,
                np.log(g.pd / g.pc),
                np.arange(len(g)),
                g.trim.iloc[0],
            )
            self.assertAlmostEqual(
                result["atet"], expected.loc[name, "R_ATET"], places=13
            )
            self.assertEqual(result["n_trimmed"], expected.loc[name, "R_trimmed"])
            self.assertAlmostEqual(result["se"], expected.loc[name, "R_se"], places=13)

    def test_bandwidth_uses_sample_sd_and_outer_fold_mse(self):
        d = np.array([0.0, 0.3, 1.0, 2.0])
        self.assertAlmostEqual(
            paper_bandwidth(d), 2.34 * np.std(d, ddof=1) * 4 ** (-0.25) * 0.7
        )
        self.assertAlmostEqual(reference_fold_variance([1, 3], [0, 1]), 2.5)

    def test_log_ratio_matches_dnorm_without_underflow_floor(self):
        mean = np.array([0.2, 0.4])
        variance = np.array([0.5, 0.8])
        got = gaussian_log_ratio(0.5, -0.3, mean, variance)
        expected = norm.pdf((0.5 - mean) / np.sqrt(variance)) / norm.pdf(
            (-0.3 - mean) / np.sqrt(variance)
        )
        np.testing.assert_allclose(np.exp(got), expected)
        # Both ordinary Gaussian densities underflow here, but their ratio is usable.
        got = gaussian_log_ratio(100.0, 100.000001, np.zeros(20), np.ones(20))
        r = atet_score(
            np.arange(20),
            np.zeros(20),
            np.ones(20),
            np.ones(20),
            got,
            np.arange(20),
            trim=1,
        )
        self.assertAlmostEqual(r["atet"], 0.0, places=10)
        self.assertAlmostEqual(r["ess_treat"], 20.0)
        self.assertNotIn("supported", r)  # no ESS acceptance/rejection rule


if __name__ == "__main__":
    unittest.main()
