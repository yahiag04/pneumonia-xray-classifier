import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.run_scaling_study import build_scaling_specs, find_saturation_knee, metric_columns, scaling_run_id
from thesis.data import BinaryImageDataset, make_balanced_subset


class BalancedTrainSubsetTest(unittest.TestCase):
    def test_make_balanced_subset_selects_equal_labels_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ["normal", "pneumonia"]:
                label_dir = root / label
                label_dir.mkdir()
                for index in range(5):
                    Image.new("L", (4, 4), color=128).save(label_dir / f"{label}_{index}.png")
            dataset = BinaryImageDataset(root)

            first = make_balanced_subset(dataset, size=6, seed=123)
            second = make_balanced_subset(dataset, size=6, seed=123)

        first_labels = [dataset.samples[index][1] for index in first.indices]
        self.assertEqual(first_labels.count(0), 3)
        self.assertEqual(first_labels.count(1), 3)
        self.assertEqual(first.indices, second.indices)


class ScalingStudyUtilityTest(unittest.TestCase):
    def test_build_scaling_specs_can_select_only_efficientnet_family(self):
        specs = build_scaling_specs(family="efficientnet")

        self.assertEqual(
            [spec["model_name"] for spec in specs],
            ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"],
        )

    def test_scaling_run_id_identifies_family_variant_and_width(self):
        self.assertEqual(
            scaling_run_id({"family": "pneumonia_net", "width": 1.5}),
            "pneumonia_net_width_1_5",
        )
        self.assertEqual(
            scaling_run_id({"family": "efficientnet", "variant": "efficientnet_b2"}),
            "efficientnet_b2",
        )

    def test_metric_columns_expand_requested_dataset_names(self):
        self.assertEqual(
            metric_columns(["rsna", "kermany"]),
            [
                "rsna_balanced_accuracy",
                "rsna_sensitivity",
                "rsna_specificity",
                "rsna_roc_auc",
                "kermany_balanced_accuracy",
                "kermany_sensitivity",
                "kermany_specificity",
                "kermany_roc_auc",
            ],
        )

    def test_find_saturation_knee_marks_first_small_gain_after_parameter_doubling(self):
        points = [
            {"parameters": 100, "balanced_accuracy": 0.70, "variant": "a"},
            {"parameters": 210, "balanced_accuracy": 0.735, "variant": "b"},
            {"parameters": 430, "balanced_accuracy": 0.742, "variant": "c"},
        ]

        knee = find_saturation_knee(points)

        self.assertEqual(knee["variant"], "c")


if __name__ == "__main__":
    unittest.main()
