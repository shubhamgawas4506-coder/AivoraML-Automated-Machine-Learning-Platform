"""Focused tests for bounded, evaluation-only ML diagnostics."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.ml_engine_enhanced import run_enhanced_workflow  # noqa: E402


class EnhancedMLDiagnosticsTest(unittest.TestCase):
    def test_event_label_and_selection_metric_are_explicit(self):
        generator = np.random.default_rng(19)
        rows = 180
        signal = generator.normal(size=rows)
        event_probability = 1.0 / (1.0 + np.exp(-(-1.8 + 1.5 * signal)))
        labels = np.where(
            generator.random(rows) < event_probability,
            "Fraud",
            "Legit",
        )
        frame = pd.DataFrame({
            "signal": signal,
            "noise": generator.normal(size=rows),
            "outcome": labels,
        })

        result = run_enhanced_workflow(
            frame,
            "outcome",
            task_type="classification",
            max_models=1,
            tune_best=False,
        )

        self.assertEqual(result["positive_label"], "Fraud")
        self.assertEqual(
            result["model_selection"]["metric"],
            "positive_class_f1",
        )
        self.assertFalse(result["model_selection"]["holdout_used_for_selection"])
        self.assertEqual(
            result["threshold_selection"]["positive_label"],
            "Fraud",
        )
        self.assertFalse(
            result["probability_calibration_training"]["holdout_used"]
        )
        metrics = result["evaluation_details"]["metrics"]
        self.assertIn("positive_f1", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertEqual(result["best_score"], metrics["positive_f1"])

    def test_binary_classification_diagnostics_and_permutation_fallback(self):
        generator = np.random.default_rng(42)
        rows = 160
        feature_a = generator.normal(size=rows)
        feature_b = generator.normal(size=rows)
        target = (feature_a + 0.25 * feature_b > 0).astype(int)
        frame = pd.DataFrame({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "target": target,
        })

        # max_models=0 deliberately leaves the Dummy baseline, which has no
        # native coefficients/importances and therefore exercises the bounded
        # permutation-importance fallback.
        result = run_enhanced_workflow(
            frame,
            "target",
            task_type="classification",
            max_models=0,
            tune_best=False,
        )

        details = result["evaluation_details"]
        self.assertFalse(result["model_selection"]["holdout_used_for_selection"])
        self.assertIn("classification_report", details)
        self.assertEqual(len(details["confusion_matrix"]), 2)
        self.assertIn("false_positives", details["error_counts"])
        self.assertIn("false_negatives", details["error_counts"])
        self.assertIn("brier_score", details["probability_calibration"])
        self.assertIn(
            "expected_calibration_error",
            details["probability_calibration"],
        )
        self.assertEqual(
            result["feature_importance_diagnostics"]["method"],
            "holdout_permutation_importance",
        )
        self.assertEqual(
            result["feature_importance_diagnostics"]["status"],
            "computed",
        )

    def test_regression_error_quantiles_and_normalization(self):
        generator = np.random.default_rng(7)
        rows = 180
        feature_a = generator.normal(size=rows)
        feature_b = generator.uniform(-2, 2, size=rows)
        target = 4.0 * feature_a - 1.5 * feature_b + generator.normal(
            scale=0.4,
            size=rows,
        )
        frame = pd.DataFrame({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "target": target,
        })

        result = run_enhanced_workflow(
            frame,
            "target",
            task_type="regression",
            max_models=1,
            tune_best=False,
        )

        details = result["evaluation_details"]
        self.assertFalse(result["model_selection"]["holdout_used_for_selection"])
        self.assertIn("p95", details["absolute_error_quantiles"])
        self.assertIn("maximum", details["absolute_error_quantiles"])
        self.assertIsNotNone(
            details["normalized_error"]["mae_over_target_iqr"]
        )
        self.assertIsNotNone(
            details["normalized_error"]["rmse_over_target_std"]
        )
        self.assertIn("normalized_mae_iqr", details["metrics"])
        self.assertEqual(
            result["diagnostic_safeguards"][
                "holdout_used_for_model_or_tuning_selection"
            ],
            False,
        )


if __name__ == "__main__":
    unittest.main()
