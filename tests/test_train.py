import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from thesis.train import collect_predictions


class IdentityLogitModel(nn.Module):
    def forward(self, x):
        return x[:, :1]


class PredictionCollectionTest(unittest.TestCase):
    def test_collect_predictions_returns_reusable_probabilities_and_loss(self):
        inputs = torch.tensor([[-2.0], [0.0], [2.0]])
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)
        criterion = nn.BCEWithLogitsLoss()

        result = collect_predictions(
            IdentityLogitModel(),
            loader,
            criterion,
            torch.device("cpu"),
        )

        expected_probabilities = torch.sigmoid(inputs.squeeze(1)).tolist()
        expected_loss = criterion(inputs.squeeze(1), labels.float()).item()
        self.assertEqual(result["labels"], labels.tolist())
        self.assertEqual(result["num_samples"], 3)
        self.assertAlmostEqual(result["loss"], expected_loss)
        for actual, expected in zip(result["probabilities"], expected_probabilities):
            self.assertAlmostEqual(actual, expected)
        self.assertGreaterEqual(result["elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
