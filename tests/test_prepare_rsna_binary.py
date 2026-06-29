import tempfile
import unittest
from pathlib import Path

from scripts.prepare_rsna_binary import (
    RsnaRecord,
    build_output_image_path,
    collect_processed_png_records,
    collect_binary_records,
    map_rsna_class,
    prepare_rsna_binary,
    split_balanced_records,
)


class PrepareRsnaBinaryTest(unittest.TestCase):
    def test_map_rsna_class_keeps_only_normal_and_lung_opacity(self):
        self.assertEqual(map_rsna_class("Normal"), "normal")
        self.assertEqual(map_rsna_class("Lung Opacity"), "pneumonia")
        self.assertIsNone(map_rsna_class("No Lung Opacity / Not Normal"))

    def test_collect_binary_records_collapses_duplicate_patient_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "stage_2_train_images"
            images.mkdir()
            (images / "p1.dcm").write_bytes(b"fake")
            (images / "p2.dcm").write_bytes(b"fake")
            (images / "p3.dcm").write_bytes(b"fake")
            details = root / "stage_2_detailed_class_info.csv"
            details.write_text(
                "patientId,class\n"
                "p1,Lung Opacity\n"
                "p1,Lung Opacity\n"
                "p2,Normal\n"
                "p3,No Lung Opacity / Not Normal\n"
            )

            records = collect_binary_records(details, images)

        self.assertEqual(
            [(record.patient_id, record.label) for record in records],
            [("p1", "pneumonia"), ("p2", "normal")],
        )

    def test_collect_processed_png_records_reads_metadata_and_collapses_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "Training" / "Images"
            images.mkdir(parents=True)
            for patient_id in ["p1", "p2", "p3"]:
                (images / f"{patient_id}.png").write_bytes(b"fake")
            metadata = root / "stage2_train_metadata.csv"
            metadata.write_text(
                "patientId,x,y,width,height,Target,class,age,sex,modality,position\n"
                "p1,1,2,3,4,1,Lung Opacity,62,M,CR,AP\n"
                "p1,5,6,7,8,1,Lung Opacity,62,M,CR,AP\n"
                "p2,,,,,0,Normal,51,F,CR,PA\n"
                "p3,,,,,0,No Lung Opacity / Not Normal,44,M,CR,PA\n"
            )

            records = collect_processed_png_records(metadata, images)

        self.assertEqual(
            [(record.patient_id, record.label) for record in records],
            [("p1", "pneumonia"), ("p2", "normal")],
        )
        self.assertEqual(records[0].dicom_path, images / "p1.png")
        self.assertEqual(records[0].patient_age, "62")
        self.assertEqual(records[0].patient_sex, "M")
        self.assertEqual(records[0].view_position, "AP")
        self.assertEqual(records[0].source_format, "png")

    def test_split_balanced_records_returns_requested_counts_per_class(self):
        records = [
            RsnaRecord(f"n{i}", "normal", Path(f"n{i}.dcm"), "Normal")
            for i in range(6)
        ] + [
            RsnaRecord(f"p{i}", "pneumonia", Path(f"p{i}.dcm"), "Lung Opacity")
            for i in range(6)
        ]

        splits = split_balanced_records(
            records,
            train_per_class=2,
            val_per_class=1,
            test_per_class=1,
            seed=7,
        )

        self.assertEqual(len(splits["train"]), 4)
        self.assertEqual(len(splits["val"]), 2)
        self.assertEqual(len(splits["test"]), 2)
        for split_records in splits.values():
            labels = [record.label for record in split_records]
            self.assertEqual(labels.count("normal"), labels.count("pneumonia"))

    def test_build_output_image_path_uses_imagefolder_layout(self):
        record = RsnaRecord("abc123", "pneumonia", Path("abc123.dcm"), "Lung Opacity")

        path = build_output_image_path(Path("data/rsna"), "train", record)

        self.assertEqual(path, Path("data/rsna/train/pneumonia/abc123.png"))

    def test_prepare_rsna_binary_supports_processed_png_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "rsna" / "Training" / "Images"
            images.mkdir(parents=True)
            rows = [
                ("n0", "Normal", "51", "F", "PA"),
                ("n1", "Normal", "52", "M", "AP"),
                ("n2", "Normal", "53", "F", "PA"),
                ("p0", "Lung Opacity", "61", "M", "AP"),
                ("p1", "Lung Opacity", "62", "F", "PA"),
                ("p2", "Lung Opacity", "63", "M", "AP"),
                ("x0", "No Lung Opacity / Not Normal", "70", "F", "PA"),
            ]
            metadata_lines = [
                "patientId,x,y,width,height,Target,class,age,sex,modality,position"
            ]
            for patient_id, rsna_class, age, sex, position in rows:
                (images / f"{patient_id}.png").write_bytes(b"fake-png")
                target = "1" if rsna_class == "Lung Opacity" else "0"
                metadata_lines.append(
                    f"{patient_id},,,,,{target},{rsna_class},{age},{sex},CR,{position}"
                )
            (root / "rsna" / "stage2_train_metadata.csv").write_text(
                "\n".join(metadata_lines) + "\n"
            )

            summary = prepare_rsna_binary(
                rsna_root=root / "rsna",
                output_root=root / "out",
                train_per_class=1,
                val_per_class=1,
                test_per_class=1,
                link_mode="copy",
            )

            metadata_csv = root / "out" / "metadata.csv"
            self.assertTrue(metadata_csv.is_file())
            self.assertEqual(summary["total_binary_records_available"], 6)
            self.assertEqual(summary["source_layout"], "processed_png")
            self.assertEqual(summary["splits"]["train"], {"total": 2, "normal": 1, "pneumonia": 1})
            self.assertEqual(len(list((root / "out" / "train" / "normal").glob("*.png"))), 1)
            self.assertEqual(len(list((root / "out" / "train" / "pneumonia").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
