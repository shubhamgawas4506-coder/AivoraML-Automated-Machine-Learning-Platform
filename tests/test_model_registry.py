"""Focused tests for immutable local model version storage."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.model_registry import (  # noqa: E402
    DuplicateModelVersionError,
    InvalidModelIdentifierError,
    ModelArtifactIntegrityError,
    ModelRegistry,
    ModelVersionNotFoundError,
)


class _FittedPredictor:
    """Small serializable prediction protocol used instead of a heavy model."""

    def __init__(self, label: int = 1) -> None:
        self.label = label

    def predict(self, rows):
        return np.full(len(rows), self.label)


class ModelRegistryTest(unittest.TestCase):
    PROJECT_ID = "a1b2c3d4e5f6"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "model_versions"
        self.registry = ModelRegistry(self.root, retention_limit=20)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_register_list_resolve_and_load(self) -> None:
        first = self.registry.register(
            self.PROJECT_ID,
            "run-001",
            _FittedPredictor(7),
            {
                "best_model": "Tiny Predictor",
                "score": np.float64(0.91),
                "not_finite": float("nan"),
                "_private": "not public",
            },
        )
        second = self.registry.register(
            self.PROJECT_ID,
            "run-002",
            _FittedPredictor(9),
            {"best_model": "New Predictor", "score": 0.93},
        )

        self.assertEqual(first["version_id"], "run-001")
        self.assertRegex(first["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(first["size_bytes"], 0)
        self.assertIsNone(first["metadata"]["not_finite"])
        self.assertNotIn("_private", first["metadata"])
        json.dumps(first)

        versions = self.registry.list_versions(self.PROJECT_ID)
        self.assertEqual(
            [item["version_id"] for item in versions],
            ["run-002", "run-001"],
        )
        self.assertEqual(
            self.registry.get_metadata(self.PROJECT_ID, "run-001"),
            first,
        )
        artifact_path = self.registry.resolve_artifact(
            self.PROJECT_ID,
            "run-001",
        )
        self.assertEqual(
            artifact_path,
            self.root / self.PROJECT_ID / "run-001.joblib",
        )
        artifact = self.registry.load_artifact(self.PROJECT_ID, "run-001")
        self.assertEqual(artifact["version_id"], "run-001")
        self.assertEqual(artifact["public_metadata"], first["metadata"])
        predictions = artifact["pipeline"].predict([[1], [2]])
        self.assertEqual(predictions.tolist(), [7, 7])
        self.assertGreater(
            len(self.registry.read_artifact_bytes(self.PROJECT_ID, "run-001")),
            0,
        )

    def test_checksum_tampering_blocks_resolve_load_and_download(self) -> None:
        self.registry.register(
            self.PROJECT_ID,
            "tamper-test",
            _FittedPredictor(),
            {"score": 0.8},
        )
        artifact_path = self.root / self.PROJECT_ID / "tamper-test.joblib"
        original = artifact_path.read_bytes()
        artifact_path.write_bytes(original + b"tampered")

        for operation in (
            self.registry.verify_artifact,
            self.registry.resolve_artifact,
            self.registry.load_artifact,
            self.registry.read_artifact_bytes,
        ):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(ModelArtifactIntegrityError):
                    operation(self.PROJECT_ID, "tamper-test")

    def test_path_safety_validation(self) -> None:
        invalid_projects = (
            "../escape",
            "a1b2c3",
            "a1b2c3d4e5fg",
            " a1b2c3d4e5f6",
            "a1b2c3d4e5f6/other",
        )
        for project_id in invalid_projects:
            with self.subTest(project_id=project_id):
                with self.assertRaises(InvalidModelIdentifierError):
                    self.registry.list_versions(project_id)

        invalid_versions = (
            "../escape",
            "folder/model",
            r"folder\model",
            ".hidden",
            "has space",
            "con",
            "v" * 65,
            "version ",
        )
        for version_id in invalid_versions:
            with self.subTest(version_id=version_id):
                with self.assertRaises(InvalidModelIdentifierError):
                    self.registry.register(
                        self.PROJECT_ID,
                        version_id,
                        _FittedPredictor(),
                    )
        self.assertFalse((Path(self.temporary.name) / "escape.joblib").exists())

    def test_duplicate_is_rejected_without_changing_original(self) -> None:
        original = self.registry.register(
            self.PROJECT_ID,
            "immutable",
            _FittedPredictor(1),
            {"score": 0.5},
        )
        original_bytes = self.registry.read_artifact_bytes(
            self.PROJECT_ID,
            "immutable",
        )
        with self.assertRaises(DuplicateModelVersionError):
            self.registry.register(
                self.PROJECT_ID,
                "immutable",
                _FittedPredictor(2),
                {"score": 0.99},
            )
        self.assertEqual(
            self.registry.get_metadata(self.PROJECT_ID, "immutable"),
            original,
        )
        self.assertEqual(
            self.registry.read_artifact_bytes(self.PROJECT_ID, "immutable"),
            original_bytes,
        )

    def test_retention_prunes_artifacts_but_tombstones_ids(self) -> None:
        registry = ModelRegistry(self.root, retention_limit=2)
        for index in range(3):
            registry.register(
                self.PROJECT_ID,
                f"version-{index}",
                _FittedPredictor(index),
                {"order": index},
            )
        self.assertEqual(
            [item["version_id"] for item in registry.list_versions(self.PROJECT_ID)],
            ["version-2", "version-1"],
        )
        self.assertFalse(
            (self.root / self.PROJECT_ID / "version-0.joblib").exists()
        )
        with self.assertRaises(ModelVersionNotFoundError):
            registry.load_artifact(self.PROJECT_ID, "version-0")
        with self.assertRaises(DuplicateModelVersionError):
            registry.register(
                self.PROJECT_ID,
                "version-0",
                _FittedPredictor(99),
            )

    def test_concurrent_registration_keeps_manifest_complete(self) -> None:
        version_ids = [f"parallel-{index:02d}" for index in range(16)]

        def register(version_id: str) -> str:
            # Separate instances verify that the shared root lock, rather than
            # only one object's lock, protects atomic manifest updates.
            registry = ModelRegistry(self.root, retention_limit=20)
            return registry.register(
                self.PROJECT_ID.upper(),
                version_id,
                _FittedPredictor(int(version_id[-2:])),
                {"version": version_id},
            )["version_id"]

        with ThreadPoolExecutor(max_workers=8) as executor:
            registered = list(executor.map(register, version_ids))
        self.assertCountEqual(registered, version_ids)

        versions = self.registry.list_versions(self.PROJECT_ID)
        self.assertEqual(len(versions), len(version_ids))
        self.assertCountEqual(
            [item["version_id"] for item in versions],
            version_ids,
        )
        manifest_path = self.root / self.PROJECT_ID / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["versions"]), len(version_ids))
        self.assertFalse(list(manifest_path.parent.glob(".*.tmp")))
        for version in versions:
            self.registry.verify_artifact(self.PROJECT_ID, version["version_id"])

        def register_duplicate(_index: int) -> str:
            registry = ModelRegistry(self.root, retention_limit=20)
            try:
                registry.register(
                    self.PROJECT_ID,
                    "one-concurrent-id",
                    _FittedPredictor(),
                )
                return "registered"
            except DuplicateModelVersionError:
                return "duplicate"

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(register_duplicate, range(8)))
        self.assertEqual(outcomes.count("registered"), 1)
        self.assertEqual(outcomes.count("duplicate"), 7)


if __name__ == "__main__":
    unittest.main()
