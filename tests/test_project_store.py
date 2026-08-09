"""Focused durability and path-safety tests for the local project repository."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.project_store import (  # noqa: E402
    InvalidProjectIdError,
    ProjectAlreadyExistsError,
    ProjectRepository,
    ProjectStoreCorruptionError,
    SnapshotChecksumError,
)


def _snapshot(
    filename: str,
    values: list[int],
    *,
    trained: bool = False,
    predictions: int = 0,
) -> dict:
    return {
        "version": 1,
        "filename": filename,
        "df": pd.DataFrame({"value": values, "group": ["a"] * len(values)}),
        "profile": {"rows": len(values)},
        "workflow_result": {"_pipeline": "fitted"} if trained else None,
        "predictions": [{"prediction": index} for index in range(predictions)],
        "report_bundle": {"status": "completed"} if trained else None,
    }


class ProjectRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.storage = Path(self.tempdir.name) / "storage"
        self.projects_root = self.storage / "projects"
        self.repository = ProjectRepository(
            self.projects_root,
            max_projects=64,
            max_snapshot_bytes=8 * 1024 * 1024,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_two_projects_remain_isolated_across_save_and_load(self) -> None:
        first = self.repository.create_project(
            _snapshot("first.csv", [1, 2, 3])
        )
        second = self.repository.create_project(
            _snapshot("second.csv", [80, 90])
        )

        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(self.repository.active_project_id, second["id"])
        self.assertEqual(first["rows"], 3)
        self.assertEqual(first["cols"], 2)
        self.assertEqual(second["rows"], 2)

        updated_first = self.repository.save_project(
            first["id"],
            _snapshot(
                "first-renamed.csv",
                [4, 5, 6, 7],
                trained=True,
                predictions=3,
            ),
            activate=False,
        )
        self.assertTrue(updated_first["trained"])
        self.assertTrue(updated_first["report_ready"])
        self.assertEqual(updated_first["prediction_count"], 3)

        loaded_second = self.repository.load_project(second["id"])
        self.assertEqual(loaded_second["filename"], "second.csv")
        self.assertEqual(loaded_second["df"]["value"].tolist(), [80, 90])
        loaded_first = self.repository.load_project(first["id"])
        self.assertEqual(loaded_first["df"]["value"].tolist(), [4, 5, 6, 7])
        self.assertEqual(loaded_first["project_id"], first["id"])

    def test_restart_lists_projects_and_restores_active_pointer(self) -> None:
        first = self.repository.create_project(_snapshot("one.csv", [1]))
        second = self.repository.create_project(_snapshot("two.csv", [2]))
        self.repository.activate_project(first["id"])

        restarted = ProjectRepository(
            self.projects_root,
            max_projects=64,
            max_snapshot_bytes=8 * 1024 * 1024,
        )
        self.assertEqual(restarted.active_project_id, first["id"])
        self.assertEqual(
            {item["id"] for item in restarted.list_projects()},
            {first["id"], second["id"]},
        )
        self.assertEqual(
            restarted.load_project()["df"]["value"].tolist(),
            [1],
        )
        restarted.activate_project(None)
        self.assertIsNone(
            ProjectRepository(self.projects_root).active_project_id
        )

    def test_snapshot_tamper_is_rejected_before_joblib_load(self) -> None:
        project = self.repository.create_project(_snapshot("safe.csv", [1, 2]))
        project_dir = self.projects_root / project["id"]
        manifest = json.loads(
            (project_dir / "manifest.json").read_text(encoding="utf-8")
        )
        snapshot_path = project_dir / manifest["snapshot_file"]
        with snapshot_path.open("ab") as tampered:
            tampered.write(b"tampered")

        with self.assertRaises(SnapshotChecksumError):
            self.repository.load_project(project["id"])

    def test_invalid_ids_and_manifest_paths_fail_closed(self) -> None:
        invalid_ids = (
            "../outside000",
            "ABCDEF123456",
            "abcdef12345",
            "abcdef1234567",
            "abcde/123456",
            "",
        )
        for project_id in invalid_ids:
            with self.subTest(project_id=project_id):
                with self.assertRaises(InvalidProjectIdError):
                    self.repository.load_project(project_id)
                with self.assertRaises(InvalidProjectIdError):
                    self.repository.create_project(
                        _snapshot("x.csv", [1]), project_id=project_id
                    )

        fixed_id = "0123456789ab"
        self.repository.create_project(
            _snapshot("owned.csv", [1]), project_id=fixed_id
        )
        with self.assertRaises(ProjectAlreadyExistsError):
            self.repository.create_project(
                _snapshot("overwrite.csv", [999]), project_id=fixed_id
            )

        # Even a tampered app manifest cannot redirect a load outside its
        # exact project directory.  The index/manifest binding detects this
        # before joblib is called.
        manifest_path = self.projects_root / fixed_id / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["snapshot_file"] = "../../outside.joblib"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ProjectStoreCorruptionError):
            self.repository.load_project(fixed_id)
        self.assertFalse((self.storage.parent / "outside.joblib").exists())

    def test_deletion_is_exact_reported_and_non_recursive(self) -> None:
        first = self.repository.create_project(_snapshot("delete.csv", [1]))
        second = self.repository.create_project(_snapshot("keep.csv", [2]))
        self.repository.activate_project(first["id"])

        result = self.repository.delete_project(first["id"])
        self.assertTrue(result["deleted"])
        self.assertEqual(result["failures"], [])
        self.assertIn("manifest.json", result["deleted_files"])
        self.assertFalse((self.projects_root / first["id"]).exists())
        self.assertTrue((self.projects_root / second["id"]).is_dir())
        self.assertIsNone(self.repository.active_project_id)

        guarded = self.repository.create_project(_snapshot("guarded.csv", [3]))
        guarded_dir = self.projects_root / guarded["id"]
        nested = guarded_dir / "unexpected-directory"
        nested.mkdir()
        guarded_result = self.repository.delete_project(guarded["id"])
        self.assertFalse(guarded_result["deleted"])
        self.assertTrue(guarded_result["failures"])
        self.assertTrue((guarded_dir / "manifest.json").exists())
        self.assertTrue(nested.exists())

    def test_concurrent_creates_and_saves_keep_index_consistent(self) -> None:
        def create(index: int) -> dict:
            return self.repository.create_project(
                _snapshot(f"concurrent-{index}.csv", [index, index + 1]),
                activate=False,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            created = list(executor.map(create, range(16)))
        ids = [item["id"] for item in created]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(self.repository.list_projects()), 16)

        target = ids[0]

        def save(marker: int) -> None:
            self.repository.save_project(
                target,
                _snapshot("shared.csv", [marker]),
                activate=False,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(save, range(12)))

        restarted = ProjectRepository(
            self.projects_root,
            max_projects=64,
            max_snapshot_bytes=8 * 1024 * 1024,
        )
        self.assertEqual(len(restarted.list_projects()), 16)
        self.assertIn(
            restarted.load_project(target, activate=False)["df"]["value"].iloc[0],
            range(12),
        )
        self.assertEqual(list(self.projects_root.glob(".*.tmp")), [])
        self.assertEqual(
            list((self.projects_root / target).glob(".snapshot-*.tmp")), []
        )

    def test_explicit_legacy_import_checks_source_and_checksum(self) -> None:
        legacy_path = self.storage / "workspace.joblib"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy = _snapshot("legacy.csv", [7, 8, 9])
        legacy["project_id"] = "a1b2c3d4e5f6"
        joblib.dump(legacy, legacy_path, compress=3)
        checksum = hashlib.sha256(legacy_path.read_bytes()).hexdigest()

        imported = self.repository.import_legacy_snapshot(
            legacy_path, expected_sha256=checksum
        )
        self.assertEqual(imported["id"], "a1b2c3d4e5f6")
        self.assertEqual(imported["rows"], 3)
        self.assertEqual(
            self.repository.load_project(imported["id"])["df"]["value"].tolist(),
            [7, 8, 9],
        )

        other_legacy = self.storage / "other.joblib"
        joblib.dump(_snapshot("other.csv", [4]), other_legacy, compress=3)
        with self.assertRaises(SnapshotChecksumError):
            self.repository.import_legacy_snapshot(
                other_legacy, expected_sha256="0" * 64
            )


if __name__ == "__main__":
    unittest.main()
