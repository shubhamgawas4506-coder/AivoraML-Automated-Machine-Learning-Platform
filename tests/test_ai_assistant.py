"""Focused tests for secure dual-provider assistant routing."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.ai_assistant import (  # noqa: E402
    ProjectAssistant,
    assistant_status,
    get_project_assistant,
)


class ProjectAssistantTest(unittest.TestCase):
    def test_disabled_remote_ai_never_initializes_a_provider(self):
        with patch.dict(
            os.environ,
            {
                "AIVORAML_ENABLE_REMOTE_AI": "false",
                "GEMINI_API_KEY": "configured-but-disabled",
                "MISTRAL_API_KEY": "configured-but-disabled",
            },
            clear=False,
        ):
            self.assertIsNone(get_project_assistant())
            status = assistant_status()
            self.assertFalse(status["enabled"])
            self.assertNotIn("key", str(status).lower())

    def test_gemini_is_primary_without_calling_mistral(self):
        with patch.dict(
            os.environ,
            {
                "AIVORAML_ENABLE_REMOTE_AI": "true",
                "AIVORAML_AI_PROVIDER": "auto",
                "GEMINI_API_KEY": "test-gemini",
                "MISTRAL_API_KEY": "test-mistral",
            },
            clear=False,
        ):
            assistant = ProjectAssistant()
            with (
                patch.object(assistant, "_ask_gemini", return_value="Evidence-based answer."),
                patch.object(assistant, "_ask_mistral") as mistral,
            ):
                reply = assistant.answer("How is the model?", {"dataset": {"rows": 100}})
            self.assertEqual(reply["provider"], "Gemini")
            self.assertFalse(reply["fallback_used"])
            mistral.assert_not_called()

    def test_mistral_is_used_when_gemini_fails(self):
        with patch.dict(
            os.environ,
            {
                "AIVORAML_ENABLE_REMOTE_AI": "true",
                "AIVORAML_AI_PROVIDER": "auto",
                "GEMINI_API_KEY": "test-gemini",
                "MISTRAL_API_KEY": "test-mistral",
            },
            clear=False,
        ):
            assistant = ProjectAssistant()
            with (
                patch.object(assistant, "_ask_gemini", side_effect=RuntimeError("offline")),
                patch.object(assistant, "_ask_mistral", return_value="Fallback answer."),
            ):
                reply = assistant.answer("Explain recall", {}, [])
            self.assertEqual(reply["provider"], "Mistral")
            self.assertTrue(reply["fallback_used"])


if __name__ == "__main__":
    unittest.main()
