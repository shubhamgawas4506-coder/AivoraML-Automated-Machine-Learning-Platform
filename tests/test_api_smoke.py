"""End-to-end smoke tests for the local AivoraML workspace."""

import ast
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import nbformat
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ["AIVORAML_DISABLE_PERSISTENCE"] = "1"
os.environ["AIVORAML_ENABLE_REMOTE_AI"] = "false"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import backend.main as main_module  # noqa: E402
from backend.services.model_registry import ModelRegistry  # noqa: E402


class AivoraMLLifecycleSmokeTest(unittest.TestCase):
    """Exercise demo -> train -> predict -> export once for the test class."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root_patch = patch.object(
            main_module,
            "PROJECT_ROOT",
            Path(cls.temp_dir.name).resolve(),
        )
        cls.root_patch.start()
        cls.original_model_registry = main_module.model_registry
        main_module.model_registry = ModelRegistry(
            Path(cls.temp_dir.name) / "model_versions",
        )
        cls.client = TestClient(main_module.app)
        cls.demo_response = cls.client.post("/api/demo")
        cls.project_id = cls.client.get("/api/project/current").json()["id"]
        cls.training_response = cls.client.post(
            "/api/ml/professional-workflow",
            json={"target_column": "churn", "approach": "fast"},
        )

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        main_module.model_registry = cls.original_model_registry
        cls.root_patch.stop()
        cls.temp_dir.cleanup()

    def test_01_profile_contains_real_distributions(self):
        self.assertEqual(self.demo_response.status_code, 200)
        profile = self.demo_response.json()
        self.assertEqual(profile["n_rows"], 2500)
        numeric = next(
            column
            for column in profile["columns"]
            if column["name"] == "tenure_months"
        )
        self.assertEqual(
            sum(numeric["histogram"]["counts"]),
            profile["n_rows"],
        )
        self.assertGreater(len(numeric["histogram"]["bin_edges"]), 1)

    def test_02_training_uses_cv_selection_and_untouched_holdout(self):
        self.assertEqual(
            self.training_response.status_code,
            200,
            self.training_response.text,
        )
        result = self.training_response.json()
        self.assertGreaterEqual(len(result["models_trained"]), 4)
        self.assertTrue(result["best_model"])
        self.assertIsInstance(result["best_score"], (int, float))
        self.assertFalse(result["model_selection"]["holdout_used_for_selection"])
        self.assertGreaterEqual(result["model_selection"]["folds"], 2)
        self.assertTrue(result["evaluation_details"]["metrics"])
        self.assertEqual(result["models_trained"][0]["name"], "Dummy Baseline")
        self.assertIn("run_manifest", result)
        self.assertTrue(result["run_manifest"]["run_id"])
        self.assertEqual(result["run_manifest"]["random_seed"], 42)
        self.assertFalse(
            result["threshold_selection"]["holdout_used_for_selection"]
        )
        self.assertIn(
            "positive_f1",
            result["evaluation_details"]["metrics"],
        )

    def test_03_prediction_uses_fitted_schema(self):
        schema_response = self.client.get("/api/predict/schema")
        self.assertEqual(schema_response.status_code, 200)
        schema = schema_response.json()
        self.assertGreater(len(schema["fields"]), 1)
        self.assertTrue(
            all(
                field.get("auto_strategy") in {"median", "most_frequent"}
                for field in schema["fields"]
            )
        )
        self.assertTrue(
            any(field.get("recommended_value") is not None for field in schema["fields"])
        )
        values = {
            field["name"]: field["sample_value"]
            for field in schema["fields"]
            if field.get("sample_value") is not None
        }
        prediction = self.client.post("/api/predict", json={"values": values})
        self.assertEqual(prediction.status_code, 200, prediction.text)
        payload = prediction.json()
        self.assertIn("prediction", payload)
        self.assertEqual(payload["target_column"], "churn")

    def test_04_model_and_notebook_exports_are_valid(self):
        model_response = self.client.get("/api/model/download")
        self.assertEqual(model_response.status_code, 200)
        artifact = joblib.load(io.BytesIO(model_response.content))
        self.assertEqual(artifact["schema_version"], 1)
        self.assertIn("pipeline", artifact)
        self.assertTrue(artifact["feature_schema"]["ordered"])
        self.assertTrue(artifact["metrics"])

        notebook_response = self.client.get(
            f"/api/notebook/download?project_id={self.project_id}"
        )
        self.assertEqual(notebook_response.status_code, 200)
        notebook = nbformat.reads(notebook_response.text, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        self.assertGreaterEqual(len(code_cells), 10)
        for cell in code_cells:
            ast.parse(cell.source)
            self.assertNotIn("Add model here", cell.source)
            self.assertIsNotNone(cell.execution_count)
            self.assertFalse(
                any(output.output_type == "error" for output in cell.outputs)
            )
        notebook_text = "\n".join(
            "".join(output.get("text", ""))
            for cell in code_cells
            for output in cell.outputs
            if output.output_type == "stream"
        )
        self.assertIn(
            f"Selected champion: {self.training_response.json()['best_model']}",
            notebook_text,
        )

        report_status = self.client.get(
            f"/api/reports/current?project_id={self.project_id}"
        )
        self.assertEqual(report_status.status_code, 200)
        self.assertEqual(report_status.json()["status"], "completed")
        self.assertGreaterEqual(
            len(report_status.json()["completed_sections"]),
            7,
        )
        report_endpoints = {
            "/api/reports/html": b"<html",
            "/api/reports/pdf": b"%PDF",
            "/api/reports/summary": b'"run_manifest"',
            "/api/reports/metrics": b"model,",
            "/api/reports/feature-schema": b'"ordered"',
        }
        for endpoint, marker in report_endpoints.items():
            response = self.client.get(
                f"{endpoint}?project_id={self.project_id}"
            )
            self.assertEqual(response.status_code, 200, endpoint)
            self.assertIn(marker.lower(), response.content.lower(), endpoint)

        summary = self.client.get(
            f"/api/reports/summary?project_id={self.project_id}"
        ).json()
        self.assertEqual(
            summary["run_id"],
            self.training_response.json()["run_manifest"]["run_id"],
        )
        self.assertEqual(
            set(summary["checksums_sha256"]),
            {
                "notebook",
                "html_report",
                "pdf_report",
                "metrics",
                "feature_schema",
                "pipeline",
            },
        )

        pipeline_response = self.client.get(
            f"/api/reports/pipeline?project_id={self.project_id}"
        )
        self.assertEqual(pipeline_response.status_code, 200)
        report_artifact = joblib.load(io.BytesIO(pipeline_response.content))
        self.assertEqual(
            report_artifact["best_model"],
            self.training_response.json()["best_model"],
        )
        self.assertEqual(
            report_artifact["decision_threshold"],
            self.training_response.json()["decision_threshold"],
        )

    def test_05_project_exports_and_validation_errors(self):
        for endpoint in (
            "/api/preprocessor/download",
            "/api/metrics/download",
            "/api/project/download",
            "/api/dataset/download",
        ):
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200, endpoint)

        unknown = self.client.post(
            "/api/predict",
            json={"values": {"not_a_feature": 1}},
        )
        self.assertEqual(unknown.status_code, 422)

        unsupported = self.client.post(
            "/api/upload",
            files={"file": ("dataset.txt", b"a,b\n1,2\n", "text/plain")},
        )
        self.assertEqual(unsupported.status_code, 400)


if __name__ == "__main__":
    unittest.main()
