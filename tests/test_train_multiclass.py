import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from thesis.train_multiclass import (
    collect_multiclass_predictions,
    evaluate_multiclass_loader,
    load_matching_checkpoint_weights,
)


class IdentityMulticlassModel(nn.Module):
    def forward(self, x):
        return x


class MulticlassPredictionCollectionTest(unittest.TestCase):
    def test_collect_multiclass_predictions_returns_probabilities_and_loss(self):
        logits = torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 4.0],
            ]
        )
        labels = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        criterion = nn.CrossEntropyLoss()

        result = collect_multiclass_predictions(
            IdentityMulticlassModel(),
            loader,
            criterion,
            torch.device("cpu"),
        )

        expected_probabilities = torch.softmax(logits, dim=1).tolist()
        expected_loss = criterion(logits, labels).item()
        self.assertEqual(result["labels"], labels.tolist())
        self.assertEqual(result["num_samples"], 3)
        self.assertAlmostEqual(result["loss"], expected_loss)
        for actual_row, expected_row in zip(result["probabilities"], expected_probabilities):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected)

    def test_evaluate_multiclass_loader_returns_macro_metrics(self):
        logits = torch.tensor(
            [
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 4.0],
            ]
        )
        labels = torch.tensor([0, 1, 2, 2])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

        metrics = evaluate_multiclass_loader(
            IdentityMulticlassModel(),
            loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"], (1.0 + 1.0 + 0.5) / 3)
        self.assertIn("seconds_per_image", metrics)


class MatchingCheckpointLoadTest(unittest.TestCase):
    def test_load_matching_checkpoint_weights_skips_shape_mismatches(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 3))
        checkpoint = {
            "model_state": {
                "0.weight": torch.full((3, 2), 2.0),
                "0.bias": torch.full((3,), 3.0),
                "1.weight": torch.ones(1, 3),
                "1.bias": torch.ones(1),
            }
        }

        summary = load_matching_checkpoint_weights(model, checkpoint)

        self.assertEqual(summary["loaded"], ["0.bias", "0.weight"])
        self.assertEqual(summary["skipped_shape_mismatch"], ["1.bias", "1.weight"])
        self.assertTrue(torch.equal(model[0].weight, torch.full((3, 2), 2.0)))
        self.assertTrue(torch.equal(model[0].bias, torch.full((3,), 3.0)))


if __name__ == "__main__":
    unittest.main()
