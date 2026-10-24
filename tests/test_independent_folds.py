import tempfile
import unittest
from dataclasses import replace

import numpy as np
from test_paper_workflow import frame_fixture, small_config

from paper_did.config import PaperConfig
from paper_did.models import fit_nuisance, predict_controls


class IndependentFoldTests(unittest.TestCase):
    def test_each_nuisance_holds_out_every_industry_exactly_once(self):
        frame = frame_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            c = replace(small_config(tmp), gps_n_splits=3, mu_n_splits=4, gps_fold_seed=11, mu_fold_seed=29)
            fitted = fit_nuisance(frame, c)
            audit = fitted["audit"]
            self.assertEqual(len(audit["gps_folds"]), 3)
            self.assertEqual(len(audit["mu_folds"]), 4)
            self.assertFalse(audit["shared_fold_partition"])
            for role in ["gps", "mu"]:
                heldout = []
                for fold in audit[role + "_folds"]:
                    train, test = set(fold["train_industry_positions"]), set(fold["test_industry_positions"])
                    self.assertFalse(train & test)
                    heldout += list(test)
                self.assertEqual(sorted(heldout), list(range(frame.ind_code.nunique())))
            firms = np.concatenate([entry["test"] for entry in fitted["models"]])
            self.assertEqual(sorted(firms.tolist()), list(range(len(frame))))
            pred = next(predict_controls(frame, np.array([0.]), fitted, c))[1]
            self.assertTrue(np.isfinite(pred).all())

    def test_independent_mu_validation_outcomes_are_excluded_from_training(self):
        frame = frame_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            c = replace(small_config(tmp), gps_n_splits=3, mu_n_splits=4, gps_fold_seed=11, mu_fold_seed=29)
            first = fit_nuisance(frame, c)
            heldout = first["models"][0]["test"]
            changed = frame.copy()
            changed.loc[heldout, "Y_diff"] += 100
            second = fit_nuisance(changed, c)
            self.assertTrue(second["audit"]["gps_cache_hit"])
            a = next(predict_controls(frame, np.array([0.]), first, c))[1][0]
            b = next(predict_controls(changed, np.array([0.]), second, c))[1][0]
            np.testing.assert_allclose(a[heldout], b[heldout], rtol=0, atol=1e-12)

    def test_role_defaults_match_explicit_common_partition_and_gps_cache_ignores_outcome(self):
        frame = frame_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            c = small_config(tmp)
            first = fit_nuisance(frame, c)
            explicit = replace(c, gps_n_splits=c.n_splits, mu_n_splits=c.n_splits,
                               gps_fold_seed=c.seed, mu_fold_seed=c.seed)
            second = fit_nuisance(frame, explicit)
            np.testing.assert_array_equal(first["mean"], second["mean"])
            np.testing.assert_array_equal(first["variance"], second["variance"])
            self.assertTrue(second["audit"]["gps_cache_hit"])
            self.assertTrue(second["audit"]["shared_fold_partition"])
            a = next(predict_controls(frame, np.array([0.]), first, c))[1]
            b = next(predict_controls(frame, np.array([0.]), second, explicit))[1]
            np.testing.assert_array_equal(a, b)
            changed = frame.copy()
            changed["Y_diff"] += np.arange(len(frame))
            third = fit_nuisance(changed, explicit)
            self.assertTrue(third["audit"]["gps_cache_hit"])
            np.testing.assert_array_equal(first["mean"], third["mean"])

    def test_invalid_role_fold_settings_rejected(self):
        for values in [dict(gps_n_splits=1), dict(mu_n_splits=True), dict(gps_fold_seed=-1), dict(mu_fold_seed=True)]:
            with self.assertRaises(ValueError):
                PaperConfig(**values)


if __name__ == "__main__":
    unittest.main()
