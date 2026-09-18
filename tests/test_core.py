from __future__ import annotations

import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.landmark_dataframe import prepare_landmark_dataframe
from preprocessing.landmark_subsets import SUBSETS
from training.lopo_dataset import LopoDataset


class _DummyRepresentation:
    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> np.ndarray:
        return np.zeros((16, 16, 3), dtype=np.float32)


class CorePipelineTests(unittest.TestCase):
    def test_landmark_subset_counts(self) -> None:
        expected = {
            "all": 543,
            "1st": 118,
            "2nd": 80,
            "laines": 67,
            "arcanjo": 75,
        }
        self.assertEqual(
            {name: len(builder()) for name, builder in SUBSETS.items()},
            expected,
        )

    def test_laines_adds_synthetic_chest_midpoint(self) -> None:
        rows: list[dict] = []
        for frame_id in range(2):
            row = {
                "sign_id": 0,
                "sequence_id": "video_0",
                "frame_id": frame_id,
                "split": "train",
            }
            for face_index in range(468):
                for axis in "xyz":
                    row[f"face_{face_index}_{axis}"] = 0.1
            for hand in (0, 1):
                for landmark_index in range(21):
                    for axis in "xyz":
                        row[f"hand_{hand}_{landmark_index}_{axis}"] = 0.2
            for landmark_index in range(33):
                for axis in "xyz":
                    row[f"pose_{landmark_index}_{axis}"] = 0.3
            row["pose_11_x"] = 0.2
            row["pose_12_x"] = 0.6
            rows.append(row)

        prepared = prepare_landmark_dataframe(
            pd.DataFrame(rows),
            subset="laines",
            use_imputation=False,
            person_col="split",
            category_col="sign_id",
            video_col="sequence_id",
            frame_col="frame_id",
            lowercase_person=True,
        )

        coordinate_columns = [
            column
            for column in prepared.columns
            if column.endswith(("_x", "_y", "_z"))
        ]
        self.assertEqual(len(coordinate_columns), 68 * 3)
        self.assertTrue(
            np.allclose(prepared["pose_middle_chest_x"].to_numpy(), 0.4)
        )

    def test_fixed_split_preparation_and_video_dataset(self) -> None:
        rows: list[dict] = []
        for split in ("train", "val", "test"):
            for category in (0, 1):
                video_name = f"{split}_{category}"
                for frame_id in range(3):
                    row = {
                        "sign_id": category,
                        "sequence_id": video_name,
                        "frame_id": frame_id,
                        "split": split,
                    }
                    for hand in (0, 1):
                        for landmark_index in range(21):
                            for axis in "xyz":
                                row[
                                    f"hand_{hand}_{landmark_index}_{axis}"
                                ] = 0.25
                    for landmark_index in range(33):
                        for axis in "xyz":
                            row[f"pose_{landmark_index}_{axis}"] = 0.5
                    rows.append(row)

        prepared = prepare_landmark_dataframe(
            pd.DataFrame(rows),
            subset="arcanjo",
            use_imputation=False,
            person_col="split",
            category_col="sign_id",
            video_col="sequence_id",
            frame_col="frame_id",
            lowercase_person=True,
            allowed_person_values={"train", "val", "test"},
        )

        dataset = LopoDataset(
            prepared,
            _DummyRepresentation(),
            transforms=None,
            augment=False,
            person_in=["test"],
        )

        self.assertEqual(len(dataset), 2)
        image, label = dataset[0]
        self.assertEqual(image.size, (16, 16))
        self.assertIn(int(label), (0, 1))

    def test_interpolation_uses_temporal_frame_order(self) -> None:
        rows: list[dict] = []
        for frame_id in (2, 0, 1):
            row = {
                "sign_id": 0,
                "sequence_id": "video_0",
                "frame_id": frame_id,
                "split": "train",
            }
            for hand in (0, 1):
                for landmark_index in range(21):
                    for axis in "xyz":
                        row[f"hand_{hand}_{landmark_index}_{axis}"] = 0.25
            for landmark_index in range(33):
                for axis in "xyz":
                    row[f"pose_{landmark_index}_{axis}"] = 0.5

            row["hand_0_0_x"] = {
                0: 0.0,
                1: np.nan,
                2: 2.0,
            }[frame_id]
            rows.append(row)

        prepared = prepare_landmark_dataframe(
            pd.DataFrame(rows),
            subset="arcanjo",
            use_imputation=True,
            person_col="split",
            category_col="sign_id",
            video_col="sequence_id",
            frame_col="frame_id",
        )

        self.assertEqual(prepared["frame"].tolist(), [0, 1, 2])
        self.assertAlmostEqual(float(prepared.loc[1, "hand_0_00_x"]), 1.0)

    def test_publication_schema_metadata_is_supported(self) -> None:
        rows = []
        for frame_id in range(2):
            row = {
                "dataset": "ksl",
                "class_id": "ksl::01",
                "sample_id": "ksl::00::00_01.MP4",
                "sequence_id": "ksl::00::00_01.MP4",
                "signer_id": "ksl::00",
                "frame_id": frame_id,
                "split": "",
            }
            for hand in (0, 1):
                for landmark_index in range(21):
                    for axis in "xyz":
                        row[f"hand_{hand}_{landmark_index}_{axis}"] = 0.25
            for landmark_index in range(33):
                for axis in "xyz":
                    row[f"pose_{landmark_index}_{axis}"] = 0.5
            rows.append(row)

        prepared = prepare_landmark_dataframe(
            pd.DataFrame(rows), subset="arcanjo", use_imputation=False
        )
        self.assertEqual(prepared["person"].unique().tolist(), ["ksl::00"])
        self.assertEqual(prepared["video_name"].unique().tolist(), ["ksl::00::00_01.MP4"])
        self.assertEqual(prepared["frame"].tolist(), [0, 1])
        self.assertEqual(prepared["category"].unique().tolist(), [0])

    def test_alves_ksl_folds_reproduce_16_4_same_validation_and_test(self) -> None:
        script = PROJECT_ROOT / "scripts" / "batch" / "run_ksl_alves_split.py"
        spec = importlib.util.spec_from_file_location("ksl_alves_split", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for fold in module.FOLDS:
            train, validation, test = module.fold_partition(fold)
            self.assertEqual((len(train), len(validation), len(test)), (16, 4, 4))
            self.assertFalse(set(train) & set(validation))
            self.assertFalse(set(train) & set(test))
            self.assertEqual(set(validation), set(test))

    def test_duplicate_frame_key_is_rejected(self) -> None:
        row = {
            "class_id": "ksl::01", "sample_id": "ksl::sample",
            "signer_id": "ksl::00", "frame_id": 0, "dataset": "ksl",
        }
        with self.assertRaisesRegex(ValueError, "Duplicate frame keys"):
            prepare_landmark_dataframe(
                pd.DataFrame([row, row]), subset="arcanjo", use_imputation=False
            )

    def test_missing_landmark_column_is_rejected(self) -> None:
        row = {
            "class_id": "ksl::01", "sample_id": "ksl::sample",
            "signer_id": "ksl::00", "frame_id": 0, "dataset": "ksl",
        }
        with self.assertRaisesRegex(ValueError, "required landmark coordinate"):
            prepare_landmark_dataframe(
                pd.DataFrame([row]), subset="arcanjo", use_imputation=False
            )


if __name__ == "__main__":
    unittest.main()
