import math
import random
import unittest

import numpy as np
import torch

from scripts.run_multiseed_evaluation import (
    aggregate_metric_rows,
    build_per_seed_row,
    build_seed_list,
    parse_manifest_arg,
    parse_nonnegative_int,
    sample_std,
    seed_run_id,
)
from thesis.reproducibility import set_global_seed


class MultiSeedUtilityTest(unittest.TestCase):
    def test_seed_run_id_keeps_outputs_isolated_by_seed(self):
        self.assertEqual(seed_run_id(42), "seed_42")

    def test_build_seed_list_defaults_to_seed_42(self):
        self.assertEqual(build_seed_list(None), [42])

    def test_build_seed_list_rejects_duplicates_to_avoid_overwrites(self):
        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            build_seed_list([1, 1])

    def test_parse_nonnegative_int_rejects_negative_seed(self):
        with self.assertRaisesRegex(Exception, "seed must be non-negative"):
            parse_nonnegative_int("-1")

    def test_parse_manifest_arg_returns_name_and_path(self):
        name, path = parse_manifest_arg("kermany=outputs/kermany.csv")

        self.assertEqual(name, "kermany")
        self.assertEqual(str(path), "outputs/kermany.csv")

    def test_build_per_seed_row_flattens_training_and_metric_data(self):
        row = build_per_seed_row(
            seed=7,
            model_name="pneumonia_net",
            dataset_name="rsna",
            checkpoint="/tmp/best.pt",
            summary={
                "best_epoch": 3,
                "best_val_loss": 0.25,
                "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            },
            complexity={"params": 300161, "gmac": 0.527458368},
            metrics={
                "balanced_accuracy": 0.91,
                "sensitivity": 0.90,
                "specificity": 0.92,
                "roc_auc": 0.97,
                "tn": 460,
                "fp": 40,
                "fn": 50,
                "tp": 450,
            },
        )

        self.assertEqual(row["seed"], 7)
        self.assertEqual(row["model_name"], "pneumonia_net")
        self.assertEqual(row["dataset"], "rsna")
        self.assertEqual(row["epochs_run"], 3)
        self.assertEqual(row["params"], 300161)
        self.assertAlmostEqual(row["balanced_accuracy"], 0.91)
        self.assertEqual(row["tp"], 450)


class AggregationTest(unittest.TestCase):
    def test_sample_std_uses_unbiased_estimator(self):
        self.assertAlmostEqual(sample_std([1.0, 2.0, 3.0]), 1.0)

    def test_sample_std_is_zero_for_single_seed(self):
        self.assertEqual(sample_std([0.9]), 0.0)

    def test_aggregate_metric_rows_computes_mean_std_and_ci95(self):
        rows = [
            {"model_name": "m", "dataset": "rsna", "balanced_accuracy": 0.80, "roc_auc": 0.90},
            {"model_name": "m", "dataset": "rsna", "balanced_accuracy": 0.90, "roc_auc": 0.94},
            {"model_name": "m", "dataset": "kermany", "balanced_accuracy": 0.70, "roc_auc": None},
        ]

        aggregate = aggregate_metric_rows(rows, ["balanced_accuracy", "roc_auc"])
        by_key = {
            (row["model_name"], row["dataset"], row["metric"]): row
            for row in aggregate
        }

        rsna_ba = by_key[("m", "rsna", "balanced_accuracy")]
        self.assertEqual(rsna_ba["n"], 2)
        self.assertAlmostEqual(rsna_ba["mean"], 0.85)
        self.assertAlmostEqual(rsna_ba["std"], math.sqrt(0.005))
        margin = 1.96 * math.sqrt(0.005) / math.sqrt(2)
        self.assertAlmostEqual(rsna_ba["ci95_low"], 0.85 - margin)
        self.assertAlmostEqual(rsna_ba["ci95_high"], 0.85 + margin)

        kermany_ba = by_key[("m", "kermany", "balanced_accuracy")]
        self.assertEqual(kermany_ba["n"], 1)
        self.assertEqual(kermany_ba["std"], 0.0)
        self.assertEqual(kermany_ba["ci95_low"], kermany_ba["mean"])
        self.assertEqual(kermany_ba["ci95_high"], kermany_ba["mean"])
        self.assertNotIn(("m", "kermany", "roc_auc"), by_key)


class ReproducibilityTest(unittest.TestCase):
    def test_set_global_seed_controls_python_numpy_and_torch_rngs(self):
        first_report = set_global_seed(123)
        first = (random.random(), np.random.rand(), torch.rand(1).item())

        second_report = set_global_seed(123)
        second = (random.random(), np.random.rand(), torch.rand(1).item())

        self.assertEqual(first, second)
        self.assertEqual(first_report.seed, 123)
        self.assertFalse(first_report.deterministic)
        self.assertEqual(second_report.seed, 123)

    def test_set_global_seed_rejects_negative_seed(self):
        with self.assertRaisesRegex(ValueError, "seed must be non-negative"):
            set_global_seed(-1)


if __name__ == "__main__":
    unittest.main()
