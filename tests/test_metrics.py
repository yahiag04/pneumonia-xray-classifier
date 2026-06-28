import unittest

from thesis.metrics import compute_binary_metrics


class MetricsTest(unittest.TestCase):
    def test_compute_binary_metrics_from_probabilities(self):
        labels = [0, 0, 1, 1]
        probs = [0.1, 0.8, 0.9, 0.4]

        metrics = compute_binary_metrics(labels, probs, threshold=0.5)

        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["f1_pneumonia"], 0.5)


if __name__ == "__main__":
    unittest.main()
