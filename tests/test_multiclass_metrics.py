import unittest

from thesis.metrics import compute_multiclass_metrics


class MulticlassMetricsTest(unittest.TestCase):
    def test_compute_multiclass_metrics_returns_macro_and_per_class_values(self):
        labels = [0, 1, 2, 2]
        probabilities = [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.5, 0.4, 0.1],
            [0.1, 0.2, 0.7],
        ]

        metrics = compute_multiclass_metrics(
            labels,
            probabilities,
            class_names=["normal", "lung_opacity", "not_normal_no_lung_opacity"],
        )

        self.assertEqual(metrics["num_samples"], 4)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (1.0 + 1.0 + 0.5) / 3)
        self.assertAlmostEqual(metrics["recall_normal"], 1.0)
        self.assertAlmostEqual(metrics["recall_lung_opacity"], 1.0)
        self.assertAlmostEqual(metrics["recall_not_normal_no_lung_opacity"], 0.5)
        self.assertAlmostEqual(metrics["f1_macro"], (2 / 3 + 1.0 + 2 / 3) / 3)


if __name__ == "__main__":
    unittest.main()
