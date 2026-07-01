import unittest

import torch
import torch.nn as nn

from thesis.model_complexity import build_complexity_row, profile_model


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ModelComplexityTest(unittest.TestCase):
    def test_profile_model_counts_parameters_and_conv_linear_macs(self):
        model = TinyConvNet()

        profile = profile_model(model, input_shape=(1, 8, 8))

        expected_conv_macs = 2 * 8 * 8 * 1 * 3 * 3
        expected_linear_macs = 2 * 1
        self.assertEqual(profile.parameters, 23)
        self.assertEqual(profile.trainable_parameters, 23)
        self.assertEqual(profile.macs, expected_conv_macs + expected_linear_macs)
        self.assertAlmostEqual(profile.gmac, 0.000001154)

    def test_build_complexity_row_adds_balanced_accuracy_per_gmac(self):
        profile = profile_model(TinyConvNet(), input_shape=(1, 8, 8))

        row = build_complexity_row(
            "tiny",
            profile,
            input_shape=(1, 8, 8),
            performance={"rsna_balanced_accuracy": 0.8},
        )

        self.assertEqual(row["model_name"], "tiny")
        self.assertEqual(row["input_shape"], "1x8x8")
        self.assertEqual(row["parameters"], 23)
        self.assertEqual(row["macs"], 1154)
        self.assertAlmostEqual(row["rsna_balanced_accuracy_per_gmac"], 0.8 / profile.gmac)


if __name__ == "__main__":
    unittest.main()
