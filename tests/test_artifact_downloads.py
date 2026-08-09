"""Focused tests for project-scoped generated-artifact downloads."""

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ["AIVORAML_DISABLE_PERSISTENCE"] = "1"
os.environ["AIVORAML_ENABLE_REMOTE_AI"] = "false"

import backend.main as main_module  # noqa: E402
from backend.services.report_generator import (  # noqa: E402
    NotebookExecutionError,
    ReportGenerationError,
    _execute_notebook,
    _pdf_text_has_required_sections,
    _report_workdir,
)


class ArtifactDownloadTest(unittest.TestCase):
    project_id = "a1b2c3d4e5f6"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name).resolve()
        self.root_patch = patch.object(
            main_module,
            "PROJECT_ROOT",
            self.project_root,
        )
        self.root_patch.start()

        self.saved_store = {
            name: getattr(main_module.store, name)
            for name in (
                "df",
                "filename",
                "profile",
                "workflow_result",
                "project_id",
                "created_at",
                "updated_at",
                "predictions",
                "report_bundle",
            )
        }
        main_module.store.df = pd.DataFrame({"feature": [1]})
        main_module.store.filename = "download-test.csv"
        main_module.store.profile = {}
        main_module.store.workflow_result = {"_pipeline": object()}
        main_module.store.project_id = self.project_id
        main_module.store.report_bundle = None
        self.client = TestClient(main_module.app)

    def tearDown(self):
        self.client.close()
        for name, value in self.saved_store.items():
            setattr(main_module.store, name, value)
        self.root_patch.stop()
        self.temp_dir.cleanup()

    def _bundle(self, relative_path: str) -> dict:
        return {
            "status": "completed",
            "artifacts": {"notebook": relative_path},
        }

    def _download_url(self, project_id: str | None = None) -> str:
        return (
            "/api/notebook/download?project_id="
            + (project_id or self.project_id)
        )

    def test_successful_download_has_content_headers_and_filename(self):
        notebook_path = self.project_root / "notebooks" / "final.ipynb"
        notebook_path.parent.mkdir(parents=True)
        expected = b'{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}'
        notebook_path.write_bytes(expected)

        with patch.object(
            main_module,
            "_ensure_report_bundle",
            new=AsyncMock(return_value=self._bundle("notebooks/final.ipynb")),
        ):
            response = self.client.get(self._download_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, expected)
        self.assertEqual(
            response.headers["content-type"],
            "application/x-ipynb+json",
        )
        disposition = response.headers["content-disposition"]
        self.assertIn("attachment", disposition.lower())
        self.assertIn("final.ipynb", disposition)

    def test_missing_file_returns_structured_404(self):
        with patch.object(
            main_module,
            "_ensure_report_bundle",
            new=AsyncMock(return_value=self._bundle("notebooks/missing.ipynb")),
        ):
            response = self.client.get(self._download_url())

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["detail"]["code"],
            "ARTIFACT_FILE_MISSING",
        )

    def test_empty_file_never_returns_success(self):
        notebook_path = self.project_root / "notebooks" / "empty.ipynb"
        notebook_path.parent.mkdir(parents=True)
        notebook_path.touch()

        with patch.object(
            main_module,
            "_ensure_report_bundle",
            new=AsyncMock(return_value=self._bundle("notebooks/empty.ipynb")),
        ):
            response = self.client.get(self._download_url())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"]["code"], "ARTIFACT_EMPTY")

    def test_invalid_project_id_is_rejected(self):
        response = self.client.get(
            "/api/notebook/download?project_id=not-valid"
        )
        self.assertEqual(response.status_code, 422)

    def test_non_active_project_access_is_forbidden(self):
        response = self.client.get(self._download_url("b1b2c3d4e5f6"))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            response.json()["detail"]["code"],
            "PROJECT_ACCESS_DENIED",
        )

    def test_report_generation_failure_is_safe_and_structured(self):
        with patch.object(
            main_module,
            "generate_report_bundle",
            side_effect=ReportGenerationError("internal diagnostic"),
        ):
            response = self.client.post(
                "/api/reports/generate"
                f"?project_id={self.project_id}&force=true"
            )

        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]
        self.assertEqual(detail["code"], "REPORT_GENERATION_FAILED")
        self.assertNotIn("internal diagnostic", detail["message"])
        self.assertEqual(main_module.store.report_bundle["status"], "failed")

    def test_wrapped_pdf_title_passes_semantic_verification(self):
        extracted = (
            "Tesla Csv Tesla Csv Machine Learning\n"
            "Report\nFinal Evaluation"
        )
        self.assertTrue(_pdf_text_has_required_sections(extracted))

    def test_kernel_startup_timeout_becomes_structured_execution_error(self):
        executor = Mock()
        executor.preprocess.side_effect = RuntimeError(
            "Kernel didn't respond in 60 seconds"
        )

        with self.assertRaises(NotebookExecutionError) as context:
            _execute_notebook(executor, {"cells": []}, self.project_root)

        self.assertIsInstance(context.exception.__cause__, RuntimeError)

    def test_report_workdir_is_removed_without_exit_time_cleanup(self):
        work_root = self.project_root / "outputs" / "report-work"
        work_root.mkdir(parents=True)

        with _report_workdir(work_root, prefix="cleanup-") as work_dir:
            (work_dir / "kernel-file.txt").write_text("temporary", encoding="utf-8")
            self.assertTrue(work_dir.is_dir())

        self.assertFalse(work_dir.exists())

    def test_kernel_failure_resets_report_state(self):
        with patch.object(
            main_module,
            "generate_report_bundle",
            side_effect=NotebookExecutionError("kernel startup failed"),
        ):
            response = self.client.post(
                "/api/reports/generate"
                f"?project_id={self.project_id}&force=true"
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"]["code"],
            "NOTEBOOK_EXECUTION_FAILED",
        )
        self.assertEqual(main_module.store.report_bundle["status"], "failed")

    def test_background_report_job_exposes_real_progress_and_result(self):
        def generate(**kwargs):
            callback = kwargs["progress_callback"]
            callback("Executing notebook", ["Notebook generated"])
            callback(
                "Verifying artifacts",
                [
                    "Notebook generated",
                    "All cells executed",
                    "Outputs embedded",
                    "Pipeline reloaded",
                    "HTML rendered",
                    "PDF rendered and verified",
                    "Metadata and metrics saved",
                ],
            )
            return {
                "status": "completed",
                "artifacts": {"notebook": "notebooks/generated/test.ipynb"},
                "checksums_sha256": {"notebook": "a" * 64},
            }

        with patch.object(main_module, "generate_report_bundle", side_effect=generate):
            created = self.client.post(
                f"/api/reports/jobs?project_id={self.project_id}"
            )
            self.assertEqual(created.status_code, 202, created.text)
            job = created.json()
            deadline = time.monotonic() + 5
            while job["state"] in {"queued", "running", "cancel_requested"}:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.02)
                job = self.client.get(f"/api/jobs/{job['job_id']}").json()

        self.assertEqual(job["state"], "succeeded", job)
        self.assertEqual(job["progress"], 100.0)
        self.assertEqual(job["result"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
