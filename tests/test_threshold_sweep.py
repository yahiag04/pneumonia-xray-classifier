import argparse
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from thesis.threshold_sweep import (
    compute_threshold_rows,
    select_best_rows,
    select_threshold,
    write_sweep_outputs,
)
from scripts.select_thresholds import (
    _build_dataset,
    build_selection_payload,
    evaluate_selected_threshold,
    summarize_threshold_selection,
    validate_args,
    write_selection_outputs,
)
from scripts.sweep_nih_thresholds import build_model_rows


class ThresholdSweepTest(unittest.TestCase):
    def setUp(self):
        self.rows = compute_threshold_rows(
            model_name="resnet18",
            checkpoint="outputs/resnet18/best.pt",
            labels=[0, 0, 1, 1],
            probabilities=[0.2, 0.65, 0.55, 0.8],
            thresholds=[0.7, 0.5, 0.6],
            loss=0.25,
            seconds_per_image=0.01,
        )

    def test_compute_threshold_rows_sorts_thresholds_and_reuses_scores(self):
        self.assertEqual([row["threshold"] for row in self.rows], [0.5, 0.6, 0.7])
        self.assertTrue(all(row["model_name"] == "resnet18" for row in self.rows))
        self.assertTrue(
            all(row["checkpoint"] == "outputs/resnet18/best.pt" for row in self.rows)
        )
        predicted_positives = [row["tp"] + row["fp"] for row in self.rows]
        self.assertEqual(predicted_positives, sorted(predicted_positives, reverse=True))

    def test_select_best_rows_breaks_balanced_accuracy_ties_at_lower_threshold(self):
        best = select_best_rows(self.rows)

        self.assertEqual(best["resnet18"]["threshold"], 0.5)
        self.assertAlmostEqual(best["resnet18"]["balanced_accuracy"], 0.75)

    def test_write_sweep_outputs_writes_json_and_csv(self):
        metadata = {
            "manifest": "outputs/nih/manifest.csv",
            "thresholds": [0.5, 0.6, 0.7],
        }
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "sweep.json"
            csv_path = Path(tmp) / "sweep.csv"

            write_sweep_outputs(
                self.rows,
                json_path=json_path,
                csv_path=csv_path,
                metadata=metadata,
            )

            payload = json.loads(json_path.read_text())
            csv_bytes = csv_path.read_bytes()
            with csv_path.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(len(payload["rows"]), 3)
        self.assertEqual(payload["best_by_model"]["resnet18"]["threshold"], 0.5)
        self.assertEqual(len(csv_rows), 3)
        self.assertIn("balanced_accuracy", csv_rows[0])
        self.assertIn("specificity", csv_rows[0])
        self.assertEqual(csv_rows[0]["model_name"], "resnet18")
        self.assertNotIn(b"\r\n", csv_bytes)

    def test_compute_threshold_rows_rejects_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            compute_threshold_rows(
                model_name="resnet18",
                checkpoint="best.pt",
                labels=[0, 1],
                probabilities=[0.1, 0.9],
                thresholds=[],
                loss=0.1,
                seconds_per_image=0.01,
            )

        with self.assertRaises(ValueError):
            compute_threshold_rows(
                model_name="resnet18",
                checkpoint="best.pt",
                labels=[0, 1],
                probabilities=[0.1, 0.9],
                thresholds=[1.1],
                loss=0.1,
                seconds_per_image=0.01,
            )

    def test_build_model_rows_collects_predictions_once_for_all_thresholds(self):
        calls = []

        def collector():
            calls.append("called")
            return {
                "labels": [0, 1],
                "probabilities": [0.2, 0.8],
                "loss": 0.1,
                "num_samples": 2,
                "elapsed_seconds": 0.04,
            }

        rows = build_model_rows(
            model_name="resnet18",
            checkpoint="best.pt",
            thresholds=[0.5, 0.6, 0.65, 0.7],
            prediction_collector=collector,
        )

        self.assertEqual(calls, ["called"])
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["seconds_per_image"] == 0.02 for row in rows))


class ThresholdSelectionTest(unittest.TestCase):
    def test_select_threshold_maximizes_metric(self):
        rows = [
            {
                "model_name": "resnet18",
                "threshold": 0.3,
                "balanced_accuracy": 0.70,
                "sensitivity": 0.98,
            },
            {
                "model_name": "resnet18",
                "threshold": 0.5,
                "balanced_accuracy": 0.78,
                "sensitivity": 0.94,
            },
            {
                "model_name": "resnet18",
                "threshold": 0.7,
                "balanced_accuracy": 0.76,
                "sensitivity": 0.90,
            },
        ]

        selected = select_threshold(rows, metric="balanced_accuracy")

        self.assertEqual(selected["threshold"], 0.5)
        self.assertAlmostEqual(selected["balanced_accuracy"], 0.78)

    def test_select_threshold_respects_minimum_sensitivity(self):
        rows = [
            {
                "model_name": "resnet18",
                "threshold": 0.3,
                "balanced_accuracy": 0.70,
                "sensitivity": 0.98,
            },
            {
                "model_name": "resnet18",
                "threshold": 0.5,
                "balanced_accuracy": 0.78,
                "sensitivity": 0.94,
            },
            {
                "model_name": "resnet18",
                "threshold": 0.7,
                "balanced_accuracy": 0.76,
                "sensitivity": 0.90,
            },
        ]

        selected = select_threshold(
            rows, metric="balanced_accuracy", min_sensitivity=0.95
        )

        self.assertEqual(selected["threshold"], 0.3)
        self.assertAlmostEqual(selected["sensitivity"], 0.98)

    def test_select_threshold_rejects_unknown_metric(self):
        rows = [
            {
                "model_name": "resnet18",
                "threshold": 0.5,
                "balanced_accuracy": 0.78,
                "sensitivity": 0.94,
            },
        ]

        with self.assertRaises(ValueError):
            select_threshold(rows, metric="missing_metric")


class ThresholdSelectionSummaryTest(unittest.TestCase):
    def _valid_cli_args(self, **overrides):
        args = {
            "val_manifest": "val.csv",
            "val_data_root": None,
            "test_manifest": None,
            "test_data_root": None,
            "thresholds": [0.5],
            "min_sensitivity": None,
            "metric": "balanced_accuracy",
            "batch_size": 32,
            "num_workers": 0,
        }
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_summarize_threshold_selection_returns_selected_row_and_rows(self):
        summary = summarize_threshold_selection(
            model_name="efficientnet_b0",
            checkpoint="outputs/runs_fair/efficientnet_b0/best.pt",
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.6, 0.7, 0.9],
            thresholds=[0.5, 0.65],
            loss=0.2,
            seconds_per_image=0.01,
            metric="balanced_accuracy",
            min_sensitivity=None,
        )

        self.assertEqual(summary["selected"]["model_name"], "efficientnet_b0")
        self.assertEqual(summary["selected"]["threshold"], 0.65)
        self.assertEqual(len(summary["rows"]), 2)

    def test_build_selection_payload_preserves_best_by_model(self):
        rows = [
            {
                "model_name": "efficientnet_b0",
                "checkpoint": "best.pt",
                "threshold": 0.5,
                "balanced_accuracy": 0.75,
                "sensitivity": 0.8,
            },
            {
                "model_name": "efficientnet_b0",
                "checkpoint": "best.pt",
                "threshold": 0.65,
                "balanced_accuracy": 0.8,
                "sensitivity": 0.7,
            },
        ]
        summary = {"selected": rows[1], "rows": rows}
        metadata = {"selection_metric": "balanced_accuracy"}

        payload = build_selection_payload(
            metadata,
            summary,
            test_metrics={"accuracy": 0.9},
        )

        self.assertEqual(payload["metadata"], metadata)
        self.assertEqual(payload["selected"], rows[1])
        self.assertEqual(payload["rows"], rows)
        self.assertEqual(
            payload["best_by_model"]["efficientnet_b0"]["threshold"],
            0.65,
        )
        self.assertEqual(payload["test_metrics_at_selected_threshold"]["accuracy"], 0.9)

    def test_build_selection_payload_uses_selection_metric_for_best_by_model(self):
        rows = [
            {
                "model_name": "efficientnet_b0",
                "checkpoint": "best.pt",
                "threshold": 0.35,
                "balanced_accuracy": 0.80,
                "f1_pneumonia": 0.70,
                "sensitivity": 0.99,
            },
            {
                "model_name": "efficientnet_b0",
                "checkpoint": "best.pt",
                "threshold": 0.65,
                "balanced_accuracy": 0.75,
                "f1_pneumonia": 0.90,
                "sensitivity": 0.96,
            },
        ]
        summary = {"selected": rows[1], "rows": rows}
        metadata = {"selection_metric": "f1_pneumonia", "min_sensitivity": 0.95}

        payload = build_selection_payload(metadata, summary)

        self.assertEqual(
            payload["best_by_model"]["efficientnet_b0"]["threshold"],
            0.65,
        )

    def test_write_selection_outputs_writes_custom_json_and_csv(self):
        rows = compute_threshold_rows(
            model_name="resnet18",
            checkpoint="best.pt",
            labels=[0, 1],
            probabilities=[0.2, 0.8],
            thresholds=[0.5],
            loss=0.1,
            seconds_per_image=0.02,
        )
        summary = {"selected": rows[0], "rows": rows}
        metadata = {"selection_metric": "balanced_accuracy"}

        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "selection.json"
            csv_path = Path(tmp) / "selection.csv"

            write_selection_outputs(
                summary,
                metadata=metadata,
                output_json=json_path,
                output_csv=csv_path,
            )

            payload = json.loads(json_path.read_text())
            with csv_path.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertIn("best_by_model", payload)
        self.assertIn("selected", payload)
        self.assertEqual(payload["selected"]["threshold"], 0.5)
        self.assertEqual(len(csv_rows), 1)

    def test_build_dataset_uses_checkpoint_split_settings(self):
        splits = SimpleNamespace(val="validation-dataset", test="test-dataset")
        with mock.patch(
            "scripts.select_thresholds.build_transforms",
            return_value="transform",
        ) as build_transforms, mock.patch(
            "scripts.select_thresholds.build_internal_splits",
            return_value=splits,
        ) as build_splits:
            dataset = _build_dataset(
                model_name="resnet18",
                image_size=320,
                manifest=None,
                data_root="data/chest_xray",
                split="val",
                checkpoint_config={"val_fraction": 0.2, "seed": 7},
            )

        self.assertEqual(dataset, "validation-dataset")
        build_transforms.assert_not_called()
        build_splits.assert_called_once_with(
            "data/chest_xray",
            "resnet18",
            image_size=320,
            val_fraction=0.2,
            seed=7,
        )

    def test_build_test_dataset_uses_checkpoint_split_settings(self):
        splits = SimpleNamespace(val="validation-dataset", test="test-dataset")
        with mock.patch(
            "scripts.select_thresholds.build_transforms",
            return_value="transform",
        ) as build_transforms, mock.patch(
            "scripts.select_thresholds.build_internal_splits",
            return_value=splits,
        ) as build_splits:
            dataset = _build_dataset(
                model_name="resnet18",
                image_size=320,
                manifest=None,
                data_root="data/chest_xray",
                split="test",
                checkpoint_config={"val_fraction": 0.2, "seed": 7},
            )

        self.assertEqual(dataset, "test-dataset")
        build_transforms.assert_not_called()
        build_splits.assert_called_once_with(
            "data/chest_xray",
            "resnet18",
            image_size=320,
            val_fraction=0.2,
            seed=7,
        )

    def test_evaluate_selected_threshold_uses_selected_threshold(self):
        model = object()
        dataset = [("image-a", 0), ("image-b", 1)]
        criterion = object()
        device = object()

        with mock.patch(
            "scripts.select_thresholds.DataLoader",
            return_value="loader",
        ) as data_loader, mock.patch(
            "scripts.select_thresholds.collect_predictions",
            return_value={
                "labels": [0, 1],
                "probabilities": [0.4, 0.6],
                "loss": 0.2,
                "num_samples": 2,
                "elapsed_seconds": 0.04,
            },
        ) as collect, mock.patch(
            "scripts.select_thresholds.compute_binary_metrics",
            return_value={"accuracy": 1.0},
        ) as compute_metrics:
            metrics = evaluate_selected_threshold(
                model=model,
                dataset=dataset,
                criterion=criterion,
                device=device,
                checkpoint="best.pt",
                model_name="resnet18",
                threshold=0.65,
                batch_size=16,
                num_workers=2,
            )

        data_loader.assert_called_once_with(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=2,
        )
        collect.assert_called_once_with(model, "loader", criterion, device)
        compute_metrics.assert_called_once_with(
            [0, 1],
            [0.4, 0.6],
            threshold=0.65,
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["checkpoint"], "best.pt")
        self.assertEqual(metrics["model_name"], "resnet18")
        self.assertEqual(metrics["num_samples"], 2)
        self.assertEqual(metrics["loss"], 0.2)
        self.assertEqual(metrics["seconds_per_image"], 0.02)

    def test_validate_args_rejects_invalid_cli_inputs_before_inference(self):
        parser = argparse.ArgumentParser()
        args = self._valid_cli_args(
            val_manifest="val.csv",
            val_data_root="data",
        )

        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(
            SystemExit
        ):
            validate_args(parser, args)

    def test_validate_args_rejects_threshold_independent_metrics(self):
        parser = argparse.ArgumentParser()

        for metric in ("roc_auc", "pr_auc"):
            with self.subTest(metric=metric):
                args = self._valid_cli_args(metric=metric)
                with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    validate_args(parser, args)

    def test_validate_args_rejects_invalid_loader_settings(self):
        parser = argparse.ArgumentParser()
        cases = [
            {"batch_size": 0},
            {"batch_size": -1},
            {"num_workers": -1},
        ]

        for overrides in cases:
            with self.subTest(overrides=overrides):
                args = self._valid_cli_args(**overrides)
                with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    validate_args(parser, args)


if __name__ == "__main__":
    unittest.main()
