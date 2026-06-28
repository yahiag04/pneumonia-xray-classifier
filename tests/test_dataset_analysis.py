import tempfile
import unittest
from pathlib import Path

from PIL import Image

from thesis.dataset_analysis import (
    infer_binary_label,
    scan_imagefolder,
    summarize_records,
    write_analysis_outputs,
)


class DatasetAnalysisTest(unittest.TestCase):
    def test_infer_binary_label_from_folder_or_filename(self):
        self.assertEqual(infer_binary_label(Path("train/NORMAL/NORMAL-1.png")), "normal")
        self.assertEqual(infer_binary_label(Path("train/PNEUMONIA/BACTERIA-1.png")), "pneumonia")
        self.assertEqual(infer_binary_label(Path("train/PNEUMONIA/VIRUS-1.png")), "pneumonia")
        self.assertEqual(infer_binary_label(Path("train/PNEUMONIA/PNEUMONIA-1.png")), "pneumonia")
        self.assertIsNone(infer_binary_label(Path("train/OTHER/foo.png")))

    def test_scan_imagefolder_counts_binary_classes_and_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_image(root / "train" / "NORMAL" / "NORMAL-1.png", size=(11, 13))
            self._write_image(root / "train" / "PNEUMONIA" / "BACTERIA-1.png", size=(17, 19))
            self._write_image(root / "test" / "PNEUMONIA" / "VIRUS-1.jpeg", size=(23, 29))

            records = scan_imagefolder(root)
            summary = summarize_records(records)

            self.assertEqual(summary["total_images"], 3)
            self.assertEqual(summary["splits"]["train"]["normal"], 1)
            self.assertEqual(summary["splits"]["train"]["pneumonia"], 1)
            self.assertEqual(summary["splits"]["test"]["pneumonia"], 1)
            self.assertEqual(summary["image_sizes"][(11, 13)], 1)
            self.assertEqual(summary["image_sizes"][(17, 19)], 1)
            self.assertEqual(summary["image_sizes"][(23, 29)], 1)

    def test_write_analysis_outputs_can_skip_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "outputs"
            self._write_image(root / "train" / "NORMAL" / "NORMAL-1.png")
            records = scan_imagefolder(root)

            write_analysis_outputs(records, output_dir, make_plots=False)

            self.assertTrue((output_dir / "dataset_summary.csv").exists())
            self.assertTrue((output_dir / "split_class_counts.csv").exists())
            self.assertFalse((output_dir / "class_distribution.png").exists())

    def _write_image(self, path: Path, size=(8, 8)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", size, color=128).save(path)


if __name__ == "__main__":
    unittest.main()
