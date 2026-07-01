import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from thesis.model_registry import build_model
from thesis.train import collect_predictions, evaluate_checkpoint, keep_frozen_modules_eval


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


class FrozenModuleTrainingTest(unittest.TestCase):
    def test_keep_frozen_modules_eval_preserves_batchnorm_stats(self):
        model = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Linear(2, 1),
        )
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        model.train()

        keep_frozen_modules_eval(model)
        before = model[0].running_mean.clone()
        model(torch.ones(4, 2))

        self.assertFalse(model[0].training)
        self.assertTrue(model[1].training)
        self.assertTrue(torch.equal(model[0].running_mean, before))


class CheckpointMetadataTest(unittest.TestCase):
    def test_evaluate_checkpoint_rebuilds_pneumonia_net_with_saved_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_root = root / "images"
            rows = ["path,label"]
            for label in ["normal", "pneumonia"]:
                label_dir = test_root / label
                label_dir.mkdir(parents=True)
                image_path = label_dir / f"{label}.png"
                Image.new("L", (32, 32), color=128).save(image_path)
                rows.append(f"{image_path},{label}")
            manifest = root / "manifest.csv"
            manifest.write_text("\n".join(rows) + "\n")
            checkpoint = root / "best.pt"
            model = build_model("pneumonia_net", pretrained=False, width=0.5)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": "pneumonia_net",
                    "model_width": 0.5,
                    "image_size": 32,
                    "threshold": 0.5,
                },
                checkpoint,
            )

            result = evaluate_checkpoint(checkpoint, manifest_csv=manifest, device="cpu")

        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["model_name"], "pneumonia_net")


if __name__ == "__main__":
    unittest.main()
