import tempfile
import unittest
from pathlib import Path

from scripts.prepare_rsna_multiclass import (
    RsnaMulticlassRecord,
    build_output_image_path,
    collect_processed_png_records,
    map_rsna_class,
    prepare_rsna_multiclass,
    split_balanced_records,
)


class PrepareRsnaMulticlassTest(unittest.TestCase):
    def test_map_rsna_class_keeps_all_three_rsna_classes(self):
        self.assertEqual(map_rsna_class("Normal"), "normal")
        self.assertEqual(map_rsna_class("Lung Opacity"), "lung_opacity")
        self.assertEqual(
            map_rsna_class("No Lung Opacity / Not Normal"),
            "not_normal_no_lung_opacity",
        )
        self.assertIsNone(map_rsna_class("Other"))

    def test_collect_processed_png_records_reads_all_classes_and_collapses_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "Training" / "Images"
            images.mkdir(parents=True)
            for patient_id in ["n1", "p1", "x1"]:
                (images / f"{patient_id}.png").write_bytes(b"fake")
            metadata = root / "stage2_train_metadata.csv"
            metadata.write_text(
                "patientId,x,y,width,height,Target,class,age,sex,modality,position\n"
                "n1,,,,,0,Normal,51,F,CR,PA\n"
                "p1,1,2,3,4,1,Lung Opacity,62,M,CR,AP\n"
                "p1,5,6,7,8,1,Lung Opacity,62,M,CR,AP\n"
                "x1,,,,,0,No Lung Opacity / Not Normal,44,M,CR,PA\n"
            )

            records = collect_processed_png_records(metadata, images)

        self.assertEqual(
            [(record.patient_id, record.label) for record in records],
            [
                ("n1", "normal"),
                ("p1", "lung_opacity"),
                ("x1", "not_normal_no_lung_opacity"),
            ],
        )

    def test_split_balanced_records_returns_requested_counts_for_three_classes(self):
        records = []
        for label in ["normal", "lung_opacity", "not_normal_no_lung_opacity"]:
            records.extend(
                RsnaMulticlassRecord(f"{label}-{i}", label, Path(f"{label}-{i}.png"), label)
                for i in range(5)
            )

        splits = split_balanced_records(
            records,
            train_per_class=2,
            val_per_class=1,
            test_per_class=1,
            seed=7,
        )

        self.assertEqual(len(splits["train"]), 6)
        self.assertEqual(len(splits["val"]), 3)
        self.assertEqual(len(splits["test"]), 3)
        for split_records in splits.values():
            labels = [record.label for record in split_records]
            self.assertEqual(labels.count("normal"), labels.count("lung_opacity"))
            self.assertEqual(
                labels.count("normal"),
                labels.count("not_normal_no_lung_opacity"),
            )

    def test_build_output_image_path_uses_multiclass_imagefolder_layout(self):
        record = RsnaMulticlassRecord(
            "abc123",
            "not_normal_no_lung_opacity",
            Path("abc123.png"),
            "No Lung Opacity / Not Normal",
        )

        path = build_output_image_path(Path("data/rsna_multi"), "train", record)

        self.assertEqual(
            path,
            Path("data/rsna_multi/train/not_normal_no_lung_opacity/abc123.png"),
        )

    def test_prepare_rsna_multiclass_supports_processed_png_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "rsna" / "Training" / "Images"
            images.mkdir(parents=True)
            rows = []
            for label, rsna_class in [
                ("n", "Normal"),
                ("p", "Lung Opacity"),
                ("x", "No Lung Opacity / Not Normal"),
            ]:
                for i in range(3):
                    patient_id = f"{label}{i}"
                    rows.append((patient_id, rsna_class))
                    (images / f"{patient_id}.png").write_bytes(b"fake-png")
            metadata_lines = [
                "patientId,x,y,width,height,Target,class,age,sex,modality,position"
            ]
            for patient_id, rsna_class in rows:
                target = "1" if rsna_class == "Lung Opacity" else "0"
                metadata_lines.append(
                    f"{patient_id},,,,,{target},{rsna_class},60,F,CR,PA"
                )
            (root / "rsna" / "stage2_train_metadata.csv").write_text(
                "\n".join(metadata_lines) + "\n"
            )

            summary = prepare_rsna_multiclass(
                rsna_root=root / "rsna",
                output_root=root / "out",
                train_per_class=1,
                val_per_class=1,
                test_per_class=1,
                link_mode="copy",
            )

            self.assertEqual(summary["total_multiclass_records_available"], 9)
            self.assertEqual(summary["source_layout"], "processed_png")
            self.assertEqual(
                summary["splits"]["train"],
                {
                    "total": 3,
                    "normal": 1,
                    "lung_opacity": 1,
                    "not_normal_no_lung_opacity": 1,
                },
            )
            self.assertTrue((root / "out" / "metadata.csv").is_file())
            self.assertEqual(len(list((root / "out" / "train" / "normal").glob("*.png"))), 1)
            self.assertEqual(
                len(list((root / "out" / "train" / "lung_opacity").glob("*.png"))),
                1,
            )
            self.assertEqual(
                len(
                    list(
                        (
                            root
                            / "out"
                            / "train"
                            / "not_normal_no_lung_opacity"
                        ).glob("*.png")
                    )
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
