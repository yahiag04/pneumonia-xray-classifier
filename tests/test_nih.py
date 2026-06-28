import csv
import tempfile
import unittest
from pathlib import Path

from thesis.nih import build_nih_manifest, map_nih_binary_label


class NihMappingTest(unittest.TestCase):
    def test_map_nih_binary_label_excludes_non_pneumonia_pathologies(self):
        self.assertEqual(map_nih_binary_label("No Finding"), "normal")
        self.assertEqual(map_nih_binary_label("Pneumonia"), "pneumonia")
        self.assertEqual(map_nih_binary_label("Pneumonia|Infiltration"), "pneumonia")
        self.assertIsNone(map_nih_binary_label("Infiltration"))
        self.assertIsNone(map_nih_binary_label("Effusion|Atelectasis"))

    def test_map_nih_binary_label_can_require_exclusive_pneumonia(self):
        self.assertEqual(
            map_nih_binary_label("Pneumonia|Infiltration", exclusive_pneumonia=True),
            None,
        )
        self.assertEqual(
            map_nih_binary_label("Pneumonia", exclusive_pneumonia=True),
            "pneumonia",
        )

    def test_build_nih_manifest_resolves_images_and_excludes_other_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_root.mkdir()
            for name in ["normal.png", "pneumonia.png", "effusion.png"]:
                (image_root / name).write_bytes(b"fake")

            csv_path = root / "Data_Entry_2017.csv"
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Image Index", "Finding Labels"])
                writer.writeheader()
                writer.writerow({"Image Index": "normal.png", "Finding Labels": "No Finding"})
                writer.writerow({"Image Index": "pneumonia.png", "Finding Labels": "Pneumonia"})
                writer.writerow({"Image Index": "effusion.png", "Finding Labels": "Effusion"})

            manifest = build_nih_manifest(csv_path, image_root)

            self.assertEqual(len(manifest), 2)
            self.assertEqual([row.label for row in manifest], ["normal", "pneumonia"])
            self.assertTrue(all(row.path.exists() for row in manifest))


if __name__ == "__main__":
    unittest.main()
