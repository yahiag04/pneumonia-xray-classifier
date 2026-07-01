import tempfile
import unittest
from pathlib import Path

from PIL import Image

from thesis.data import (
    MULTICLASS_INDEX_TO_LABEL,
    MULTICLASS_LABEL_TO_INDEX,
    MultiClassImageDataset,
    MultiClassManifestImageDataset,
)


class MultiClassDataTest(unittest.TestCase):
    def test_multiclass_constants_define_rsna_label_order(self):
        self.assertEqual(
            MULTICLASS_LABEL_TO_INDEX,
            {
                "normal": 0,
                "lung_opacity": 1,
                "not_normal_no_lung_opacity": 2,
            },
        )
        self.assertEqual(MULTICLASS_INDEX_TO_LABEL[2], "not_normal_no_lung_opacity")

    def test_multiclass_image_dataset_reads_three_class_imagefolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in MULTICLASS_LABEL_TO_INDEX:
                label_dir = root / label
                label_dir.mkdir()
                Image.new("L", (4, 4), color=128).save(label_dir / f"{label}.png")

            dataset = MultiClassImageDataset(root)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            sorted(label for _, label in dataset.samples),
            [0, 1, 2],
        )

    def test_multiclass_manifest_dataset_accepts_image_path_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "sample.png"
            Image.new("L", (4, 4), color=128).save(image_path)
            manifest = root / "metadata.csv"
            manifest.write_text(
                "image_path,label\n"
                f"{image_path},not_normal_no_lung_opacity\n"
            )

            dataset = MultiClassManifestImageDataset(manifest)

        self.assertEqual(
            dataset.samples,
            [(image_path, MULTICLASS_LABEL_TO_INDEX["not_normal_no_lung_opacity"])],
        )


if __name__ == "__main__":
    unittest.main()
