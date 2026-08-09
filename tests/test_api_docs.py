"""Contract tests for the first-party AivoraML API documentation portal."""

import os
import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ["AIVORAML_DISABLE_PERSISTENCE"] = "1"

from backend.main import app  # noqa: E402


class ApiDocumentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_docs_route_serves_the_first_party_portal(self):
        response = self.client.get("/docs")

        self.assertEqual(response.status_code, 200)
        self.assertIn("AivoraML API", response.text)
        self.assertIn("Live contract analytics", response.text)
        self.assertNotIn("swagger-ui", response.text.lower())
        self.assertNotIn("redoc", response.text.lower())

    def test_docs_stylesheet_and_script_are_available(self):
        stylesheet = self.client.get("/static/api_docs.css")
        script = self.client.get("/static/api_docs.js")

        self.assertEqual(stylesheet.status_code, 200)
        self.assertEqual(script.status_code, 200)
        self.assertIn(".endpoint-card", stylesheet.text)
        self.assertIn("loadContract", script.text)

    def test_openapi_contract_uses_the_new_product_name(self):
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        contract = response.json()
        self.assertEqual(contract["info"]["title"], "AivoraML")
        self.assertNotIn("/docs", contract["paths"])
        self.assertGreater(len(contract["paths"]), 10)


if __name__ == "__main__":
    unittest.main()
