import unittest

from scripts.evaluate_torchxrayvision import build_result_payload, find_pneumonia_index


class TorchXRayVisionEvalTest(unittest.TestCase):
    def test_find_pneumonia_index_requires_exact_pathology_name(self):
        pathologies = ["Atelectasis", "Pneumonia", "Lung Opacity"]

        self.assertEqual(find_pneumonia_index(pathologies), 1)

        with self.assertRaises(ValueError):
            find_pneumonia_index(["Atelectasis", "Lung Opacity"])

    def test_build_result_payload_adds_model_manifest_and_sample_count(self):
        payload = build_result_payload(
            model_name="densenet121-res224-rsna",
            manifest_csv="outputs/example_manifest.csv",
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.8, 0.9, 0.4],
            threshold=0.5,
            elapsed_seconds=2.0,
        )

        self.assertEqual(payload["model_name"], "densenet121-res224-rsna")
        self.assertEqual(payload["manifest"], "outputs/example_manifest.csv")
        self.assertEqual(payload["num_samples"], 4)
        self.assertEqual(payload["tn"], 1)
        self.assertEqual(payload["fp"], 1)
        self.assertEqual(payload["fn"], 1)
        self.assertEqual(payload["tp"], 1)
        self.assertAlmostEqual(payload["seconds_per_image"], 0.5)


if __name__ == "__main__":
    unittest.main()
