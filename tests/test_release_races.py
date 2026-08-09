"""Focused release regressions for project isolation and single-writer jobs.

These tests intentionally use small in-memory frames and controlled events.  They
exercise the application coordination layer without running a real ML workflow
or generating a notebook.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("AIVORAML_DISABLE_PERSISTENCE", "1")
os.environ.setdefault("AIVORAML_ENABLE_REMOTE_AI", "false")

import backend.main as main_module  # noqa: E402
from backend.services.model_registry import ModelRegistry  # noqa: E402
from backend.services.project_store import ProjectRepository  # noqa: E402
from backend.services.training_jobs import TrainingJobManager  # noqa: E402


STORE_FIELDS = (
    "df",
    "filename",
    "profile",
    "workflow_result",
    "project_id",
    "created_at",
    "updated_at",
    "predictions",
    "report_bundle",
    "persistence_error",
)


def _set_project(
    project_id: str,
    *,
    marker: str,
    report_bundle: dict | None = None,
) -> None:
    """Install one tiny active workspace in the process-global local store."""
    main_module.store.df = pd.DataFrame(
        {"feature": [0.0, 1.0, 2.0, 3.0], "target": [0, 0, 1, 1]}
    )
    main_module.store.filename = f"{marker}.csv"
    main_module.store.profile = {
        "filename": f"{marker}.csv",
        "dataset_fingerprint": f"fingerprint-{marker}",
    }
    main_module.store.workflow_result = {"marker": marker}
    main_module.store.project_id = project_id
    main_module.store.created_at = "2026-01-01T00:00:00+00:00"
    main_module.store.updated_at = "2026-01-02T00:00:00+00:00"
    main_module.store.predictions = [{"id": f"prediction-{marker}"}]
    main_module.store.report_bundle = report_bundle
    main_module.store.persistence_error = None


def _fake_workflow_result() -> dict:
    """Return a serializable fitted result suitable for coordination tests."""
    estimator = DummyClassifier(strategy="most_frequent")
    estimator.fit([[0.0], [1.0]], [0, 1])
    run_id = uuid.uuid4().hex
    return {
        "_pipeline": estimator,
        "_preprocessor": None,
        "task_type": "classification",
        "target_column": "target",
        "features": {"numeric": ["feature"], "categorical": [], "dropped": []},
        "models_trained": [
            {
                "name": "Dummy Baseline",
                "is_best": True,
                "cv_score_mean": 0.5,
                "cv_score_std": 0.0,
                "selection_score": 0.5,
                "metrics": {"f1_score": 0.5},
            }
        ],
        "best_model": "Dummy Baseline",
        "best_score": 0.5,
        "run_manifest": {
            "run_id": run_id,
            "dataset_fingerprint": "fingerprint-alpha",
            "trained_at_utc": "2026-01-03T00:00:00+00:00",
        },
        "warnings": [],
    }


class _PauseAfterFirstStoreSnapshot:
    """RLock wrapper that pauses immediately after the first guarded snapshot."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._exit_count = 0
        self.snapshot_released = threading.Event()
        self.resume = threading.Event()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lock.release()
        self._exit_count += 1
        if self._exit_count == 1:
            self.snapshot_released.set()
            if not self.resume.wait(3):
                raise TimeoutError("Test did not resume persistence after snapshot capture.")
        return False


class _RecordingRepository:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def save_project(self, project_id, snapshot, *, filename, activate):
        self.calls.append(
            {
                "project_id": project_id,
                "snapshot": snapshot,
                "filename": filename,
                "activate": activate,
            }
        )
        return {
            "created_at": "2030-01-01T00:00:00+00:00",
            "updated_at": "2030-01-02T00:00:00+00:00",
        }


class ReleaseRaceRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_store = {
            field: getattr(main_module.store, field) for field in STORE_FIELDS
        }
        self.temporary = tempfile.TemporaryDirectory()
        self.temp_root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        with main_module.STORE_LOCK:
            for field, value in self.saved_store.items():
                setattr(main_module.store, field, value)
        self.temporary.cleanup()

    def test_persistence_binds_snapshot_to_captured_project_during_switch(self):
        """A save begun for A must never target or update newly active B."""
        project_a = "a1a2a3a4a5a6"
        project_b = "b1b2b3b4b5b6"
        _set_project(project_a, marker="alpha")
        expected_frame = main_module.store.df.copy()

        pausing_lock = _PauseAfterFirstStoreSnapshot()
        repository = _RecordingRepository()
        worker = threading.Thread(target=main_module._persist_workspace, daemon=True)

        with (
            patch.object(main_module, "STORE_LOCK", pausing_lock),
            patch.object(main_module, "project_repository", repository),
            patch.object(main_module, "_persistence_enabled", return_value=True),
        ):
            worker.start()
            self.assertTrue(
                pausing_lock.snapshot_released.wait(2),
                "Persistence never completed its guarded snapshot.",
            )
            _set_project(project_b, marker="beta")
            beta_created_at = main_module.store.created_at
            beta_updated_at = main_module.store.updated_at
            pausing_lock.resume.set()
            worker.join(3)

        self.assertFalse(worker.is_alive(), "Persistence worker did not finish.")
        self.assertEqual(len(repository.calls), 1)
        call = repository.calls[0]
        self.assertEqual(call["project_id"], project_a)
        self.assertEqual(call["filename"], "alpha.csv")
        self.assertEqual(call["snapshot"]["project_id"], project_a)
        self.assertEqual(call["snapshot"]["filename"], "alpha.csv")
        pd.testing.assert_frame_equal(call["snapshot"]["df"], expected_frame)

        self.assertEqual(main_module.store.project_id, project_b)
        self.assertEqual(main_module.store.filename, "beta.csv")
        self.assertEqual(main_module.store.created_at, beta_created_at)
        self.assertEqual(main_module.store.updated_at, beta_updated_at)

    def test_malformed_prediction_sidecar_does_not_discard_valid_snapshot(self):
        """Prediction history is optional; a bad sidecar must not brick restore."""
        project_id = "c1c2c3c4c5c6"
        repository = ProjectRepository(self.temp_root / "projects")
        snapshot_predictions = [{"id": "from-checksummed-snapshot"}]
        snapshot = {
            "version": 1,
            "project_id": project_id,
            "filename": "recoverable.csv",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-02T00:00:00+00:00",
            "df": pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
            "profile": {"dataset_fingerprint": "recoverable"},
            "workflow_result": None,
            "predictions": snapshot_predictions,
            "report_bundle": None,
        }
        repository.create_project(
            snapshot,
            project_id=project_id,
            filename="recoverable.csv",
            activate=True,
        )
        sidecar = repository.root / project_id / "predictions.json"
        sidecar.write_text("{not valid json", encoding="utf-8")
        main_module._clear_active_store()

        with (
            patch.object(main_module, "project_repository", repository),
            patch.object(main_module, "_persistence_enabled", return_value=True),
        ):
            main_module._restore_workspace()

        self.assertEqual(main_module.store.project_id, project_id)
        self.assertIsNotNone(main_module.store.df)
        self.assertEqual(main_module.store.predictions, snapshot_predictions)

    def test_legacy_workflow_routes_are_gone_or_preserve_release_invariants(self):
        """Any retained synchronous route must version and invalidate its result."""
        registry = ModelRegistry(self.temp_root / "model_versions")
        client = TestClient(main_module.app)
        try:
            for endpoint in ("/api/workflow/run", "/api/workflow/run-simple"):
                with self.subTest(endpoint=endpoint):
                    _set_project(
                        "a1a2a3a4a5a6",
                        marker="alpha",
                        report_bundle={
                            "status": "completed",
                            "run_id": "stale-report",
                            "artifacts": {"html_report": "reports/stale.html"},
                        },
                    )
                    with (
                        patch.object(
                            main_module,
                            "run_enhanced_workflow",
                            side_effect=lambda *args, **kwargs: _fake_workflow_result(),
                        ),
                        patch.object(main_module, "model_registry", registry),
                        patch.object(main_module, "_persist_workspace", return_value=None),
                    ):
                        response = client.post(
                            endpoint,
                            json={"target_column": "target", "task_type": "auto"},
                        )

                    if response.status_code in {404, 410}:
                        continue
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIsNone(main_module.store.report_bundle)
                    active_version = (
                        main_module.store.workflow_result or {}
                    ).get("model_version")
                    self.assertTrue(active_version)
                    versions = registry.list_versions(main_module.store.project_id)
                    self.assertTrue(
                        any(item["version_id"] == active_version for item in versions)
                    )
        finally:
            client.close()

    def test_legacy_workflow_cannot_commit_into_project_activated_mid_run(self):
        """A workflow begun for A may fail, but it may not overwrite active B."""
        registry = ModelRegistry(self.temp_root / "race-model-versions")

        for endpoint in ("/api/workflow/run", "/api/workflow/run-simple"):
            with self.subTest(endpoint=endpoint):
                _set_project("a1a2a3a4a5a6", marker="alpha")
                engine_started = threading.Event()
                release_engine = threading.Event()
                response_box: dict = {}

                def blocked_workflow(*args, **kwargs):
                    engine_started.set()
                    if not release_engine.wait(3):
                        raise TimeoutError("Test did not release the workflow.")
                    return _fake_workflow_result()

                def request_route(client):
                    try:
                        response_box["response"] = client.post(
                            endpoint,
                            json={"target_column": "target", "task_type": "auto"},
                        )
                    except BaseException as exc:  # captured for the test thread
                        response_box["exception"] = exc

                with (
                    TestClient(main_module.app) as client,
                    patch.object(
                        main_module,
                        "run_enhanced_workflow",
                        side_effect=blocked_workflow,
                    ),
                    patch.object(main_module, "model_registry", registry),
                    patch.object(main_module, "_persist_workspace", return_value=None),
                ):
                    request = threading.Thread(
                        target=request_route,
                        args=(client,),
                        daemon=True,
                    )
                    request.start()
                    if engine_started.wait(2):
                        _set_project("b1b2b3b4b5b6", marker="beta")
                    release_engine.set()
                    request.join(4)

                self.assertFalse(request.is_alive(), "Legacy request did not finish.")
                self.assertNotIn("exception", response_box)
                response = response_box["response"]
                if not engine_started.is_set():
                    self.assertIn(response.status_code, {404, 410})
                    continue
                self.assertIn(response.status_code, {200, 409, 410})
                self.assertEqual(main_module.store.project_id, "b1b2b3b4b5b6")
                self.assertEqual(
                    main_module.store.workflow_result,
                    {"marker": "beta"},
                    "A completed legacy run overwrote the newly active project.",
                )

    def test_background_and_sync_report_paths_have_one_generator(self):
        """A queued background report and sync request must share one writer."""
        manager = TrainingJobManager(
            max_workers=1,
            max_history=20,
            max_outstanding_jobs=8,
            thread_name_prefix="release-race",
        )
        project_id = "d1d2d3d4d5d6"
        _set_project(project_id, marker="report")
        main_module.store.workflow_result = {
            "_pipeline": object(),
            "run_manifest": {"run_id": "report-run"},
            "warnings": [],
        }

        blocker_started = threading.Event()
        release_worker = threading.Event()
        first_writer = threading.Event()
        second_writer = threading.Event()
        release_generation = threading.Event()
        calls_lock = threading.Lock()
        generation_calls = 0
        sync_box: dict = {}
        artifact_content = b"verified report artifact"
        artifact = self.temp_root / "reports" / "release-race.html"

        def occupy_only_worker(report_progress, is_cancelled):
            blocker_started.set()
            if not release_worker.wait(4):
                raise TimeoutError("Test did not release the occupied worker.")
            return {"released": True}

        def generate(**kwargs):
            nonlocal generation_calls
            with calls_lock:
                generation_calls += 1
                current_call = generation_calls
            if current_call == 1:
                first_writer.set()
            else:
                second_writer.set()
            if not release_generation.wait(4):
                raise TimeoutError("Test did not release report generation.")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(artifact_content)
            return {
                "status": "completed",
                "run_id": "report-run",
                "artifacts": {"html_report": "reports/release-race.html"},
                "checksums_sha256": {
                    "html_report": hashlib.sha256(artifact_content).hexdigest()
                },
            }

        def request_sync_report(client):
            try:
                sync_box["response"] = client.post(
                    f"/api/reports/generate?project_id={project_id}"
                )
            except BaseException as exc:  # captured for the test thread
                sync_box["exception"] = exc

        blocker = manager.submit("eeeeeeeeeeee", occupy_only_worker)
        self.assertTrue(blocker_started.wait(2))

        sync_request = None
        client = TestClient(main_module.app)
        try:
            with (
                patch.object(main_module, "training_jobs", manager),
                patch.object(main_module, "PROJECT_ROOT", self.temp_root),
                patch.object(main_module, "generate_report_bundle", side_effect=generate),
                patch.object(main_module, "_persist_workspace", return_value=None),
            ):
                created = client.post(
                    f"/api/reports/jobs?project_id={project_id}"
                )
                self.assertEqual(created.status_code, 202, created.text)
                report_job_id = created.json()["job_id"]

                sync_request = threading.Thread(
                    target=request_sync_report,
                    args=(client,),
                    daemon=True,
                )
                sync_request.start()

                deadline = time.monotonic() + 2
                while (
                    not first_writer.is_set()
                    and sync_request.is_alive()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)

                # Let the queued background operation compete while any sync
                # generator is deliberately still in its critical section.
                release_worker.set()
                if sync_request.is_alive():
                    self.assertTrue(first_writer.wait(2))
                    self.assertFalse(
                        second_writer.wait(0.35),
                        "Background and synchronous report paths both entered the generator.",
                    )

                release_generation.set()
                sync_request.join(4)
                report_job = manager.wait(report_job_id, timeout=4)
                manager.wait(blocker["job_id"], timeout=4)

                self.assertNotIn("exception", sync_box)
                self.assertFalse(sync_request.is_alive())
                self.assertIn(sync_box["response"].status_code, {200, 409})
                self.assertIn(
                    report_job["state"],
                    {"succeeded", "failed", "cancelled"},
                )
                self.assertEqual(generation_calls, 1)
        finally:
            release_worker.set()
            release_generation.set()
            if sync_request is not None:
                sync_request.join(1)
            client.close()
            manager.shutdown(wait=True, cancel_futures=True)

    def test_privacy_label_reflects_remote_ai_capability(self):
        """The data-source UI must not always claim an all-local data path."""
        source = (REPO_ROOT / "frontend" / "app_premium.js").read_text(
            encoding="utf-8"
        )
        privacy_lines = [
            line for line in source.splitlines() if "<span>Privacy</span>" in line
        ]
        self.assertEqual(len(privacy_lines), 1)
        self.assertNotIn(
            "<strong>Local only</strong>",
            privacy_lines[0],
            "Privacy remains hardcoded even when remote profile advice is enabled.",
        )
        self.assertRegex(
            source,
            r"capabilities\??\.remote_ai\??\.enabled",
            "The privacy label is not derived from the backend capability state.",
        )


if __name__ == "__main__":
    unittest.main()
