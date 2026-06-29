import unittest

import numpy as np

from thesis.stat_tests import (
    bootstrap_metric_ci,
    bootstrap_mean_difference_ci,
    delong_roc_test,
    holm_bonferroni,
    mcnemar_exact,
    paired_correctness,
    roc_auc,
)


class StatisticalTestsTest(unittest.TestCase):
    def test_paired_correctness_counts_discordant_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        pred_a = np.array([0, 1, 1, 0])
        pred_b = np.array([0, 0, 0, 0])

        result = paired_correctness(y_true, pred_a, pred_b)

        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["a_correct_b_wrong"], 1)
        self.assertEqual(result["a_wrong_b_correct"], 1)
        self.assertEqual(result["both_wrong"], 1)

    def test_mcnemar_exact_uses_two_sided_binomial_tail(self):
        result = mcnemar_exact(a_correct_b_wrong=4, a_wrong_b_correct=0)

        self.assertEqual(result["discordant"], 4)
        self.assertAlmostEqual(result["p_value"], 0.125)

    def test_delong_roc_test_detects_large_auc_difference(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        strong = np.array([0.05, 0.10, 0.20, 0.80, 0.90, 0.95])
        weak = np.array([0.45, 0.50, 0.55, 0.40, 0.60, 0.65])

        result = delong_roc_test(y_true, strong, weak)

        self.assertGreater(result["auc_a"], result["auc_b"])
        self.assertGreater(result["auc_difference"], 0)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_bootstrap_metric_ci_is_reproducible_and_contains_point_estimate(self):
        y_true = np.array([0, 0, 1, 1, 1, 0])
        scores = np.array([0.10, 0.20, 0.80, 0.70, 0.60, 0.30])

        result = bootstrap_metric_ci(y_true, scores, roc_auc, n_bootstraps=50, seed=7)

        self.assertEqual(result["n_bootstraps"], 50)
        self.assertLessEqual(result["ci_low"], result["point"])
        self.assertGreaterEqual(result["ci_high"], result["point"])
        self.assertEqual(result, bootstrap_metric_ci(y_true, scores, roc_auc, n_bootstraps=50, seed=7))

    def test_holm_bonferroni_adjusts_p_values_monotonically(self):
        adjusted = holm_bonferroni([0.01, 0.04, 0.001])

        self.assertEqual(adjusted, [0.02, 0.04, 0.003])

    def test_bootstrap_mean_difference_ci_uses_paired_differences(self):
        before = np.array([0.60, 0.70, 0.80])
        after = np.array([0.70, 0.80, 0.90])

        result = bootstrap_mean_difference_ci(before, after, n_bootstraps=50, seed=3)

        self.assertAlmostEqual(result["point"], 0.10)
        self.assertLessEqual(result["ci_low"], result["point"])
        self.assertGreaterEqual(result["ci_high"], result["point"])
        self.assertEqual(result["n_bootstraps"], 50)


if __name__ == "__main__":
    unittest.main()
