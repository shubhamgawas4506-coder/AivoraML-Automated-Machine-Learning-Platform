"""Release checks for jobs, model versions, batch inference, and local security."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ["AIVORAML_DISABLE_PERSISTENCE"] = "1"
os.environ["AIVORAML_ENABLE_REMOTE_AI"] = "false"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import backend.main as main_module  # noqa: E402
from backend.services.ai_advisor import get_ai_advisor  # noqa: E402
from backend.services.model_registry import ModelRegistry  # noqa: E402


class ProductUpgradeApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.original_model_registry = main_module.model_registry
        main_module.model_registry = ModelRegistry(
            Path(cls.temp_dir.name) / "model_versions",
        )
        cls.client = TestClient(main_module.app)
        demo = cls.client.post("/api/demo")
        if demo.status_code != 200:
            raise AssertionError(demo.text)
        cls.profile = demo.json()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        main_module.model_registry = cls.original_model_registry
        cls.temp_dir.cleanup()

    def test_01_health_capabilities_security_headers_and_origin_boundary(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        self.assertTrue(response.headers["x-request-id"])

        capabilities = self.client.get("/api/capabilities")
        self.assertEqual(capabilities.status_code, 200)
        self.assertTrue(capabilities.json()["training"]["background_jobs"])
        self.assertTrue(capabilities.json()["batch_prediction"]["enabled"])
        self.assertFalse(capabilities.json()["remote_ai"]["enabled"])
        self.assertFalse(capabilities.json()["prediction_history"]["retains_inputs"])

        blocked = self.client.post(
            "/api/demo",
            headers={"Origin": "https://attacker.example"},
        )
        self.assertEqual(blocked.status_code, 403)
        self.assertIsNone(get_ai_advisor())

    def test_01b_project_assistant_uses_bounded_metadata_context(self):
        captured = {}

        class FakeAssistant:
            def answer(self, message, context, history):
                captured.update({
                    "message": message,
                    "context": context,
                    "history": history,
                })
                return {
                    "answer": "Use holdout evidence.",
                    "provider": "Gemini",
                    "model": "test-model",
                    "fallback_used": False,
                    "remote": True,
                }

        with patch.object(
            main_module,
            "get_project_assistant",
            return_value=FakeAssistant(),
        ):
            response = self.client.post(
                "/api/ai/chat",
                json={
                    "message": "Is this model reliable?",
                    "view": "evaluate",
                    "strategy": "balanced",
                    "target_column": "churn",
                    "history": [{"role": "user", "content": "Earlier question"}],
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["provider"], "Gemini")
        self.assertEqual(captured["context"]["dataset"]["rows"], 2500)
        self.assertNotIn("preview", captured["context"]["dataset"])
        self.assertNotIn("sample", captured["context"]["dataset"])

    def test_02_background_training_versions_and_batch_prediction(self):
        created = self.client.post(
            "/api/training/jobs",
            json={
                "target_column": "churn",
                "approach": "fast",
                "include_eda": False,
            },
        )
        self.assertEqual(created.status_code, 202, created.text)
        job = created.json()
        self.assertIn(job["state"], {"queued", "running"})
        self.assertIn("starting_model_version", job["metadata"])

        deadline = time.monotonic() + 180
        while job["state"] in {"queued", "running", "cancel_requested"}:
            if time.monotonic() >= deadline:
                self.fail("Background training did not settle within 180 seconds.")
            time.sleep(0.15)
            response = self.client.get(f"/api/training/jobs/{job['job_id']}")
            self.assertEqual(response.status_code, 200, response.text)
            job = response.json()

        self.assertEqual(job["state"], "succeeded", job)
        self.assertEqual(job["progress"], 100.0)
        self.assertTrue(job["result"]["model_version"])
        self.assertGreaterEqual(job["result"]["total_models_compared"], 4)

        versions = self.client.get("/api/models/versions")
        self.assertEqual(versions.status_code, 200, versions.text)
        version_payload = versions.json()
        self.assertGreaterEqual(len(version_payload["versions"]), 1)
        active = next(item for item in version_payload["versions"] if item["active"])
        self.assertRegex(active["sha256"], r"^[0-9a-f]{64}$")
        download = self.client.get(
            f"/api/models/versions/{active['version_id']}/download"
        )
        self.assertEqual(download.status_code, 200)
        self.assertGreater(len(download.content), 100)

        schema = self.client.get("/api/predict/schema")
        self.assertEqual(schema.status_code, 200, schema.text)
        fields = schema.json()["fields"]
        rows = []
        for _ in range(3):
            rows.append({
                field["name"]: field.get("recommended_value")
                for field in fields
            })
        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        batch = self.client.post(
            "/api/predict/batch",
            files={"file": ("batch.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        self.assertEqual(batch.status_code, 200, batch.text)
        self.assertEqual(batch.headers["x-aivoraml-rows"], "3")
        output = pd.read_csv(io.BytesIO(batch.content))
        self.assertIn("aivoraml_prediction", output.columns)
        self.assertIn("aivoraml_model_version", output.columns)
        self.assertEqual(len(output), 3)

        malformed = self.client.post(
            "/api/predict/batch",
            files={"file": ("bad.csv", b"unknown\n1\n", "text/csv")},
        )
        self.assertEqual(malformed.status_code, 422)

    def test_03_prediction_history_can_be_purged(self):
        schema = self.client.get("/api/predict/schema").json()
        values = {
            field["name"]: field.get("recommended_value")
            for field in schema["fields"]
        }
        prediction = self.client.post("/api/predict", json={"values": values})
        self.assertEqual(prediction.status_code, 200, prediction.text)
        history = self.client.get("/api/predictions").json()["predictions"]
        self.assertTrue(history)
        self.assertNotIn("inputs", history[0])

        cleared = self.client.delete("/api/predictions")
        self.assertEqual(cleared.status_code, 200, cleared.text)
        self.assertGreaterEqual(cleared.json()["deleted"], 1)
        self.assertEqual(
            self.client.get("/api/predictions").json()["predictions"],
            [],
        )


if __name__ == "__main__":
    unittest.main()
