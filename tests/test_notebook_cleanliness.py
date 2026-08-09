"""Regression tests for clean, warning-free generated notebook model code."""

import json
import unittest
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from backend.services.ml_engine_enhanced import (
    _model_parameters,
    get_classification_models,
)
from backend.services.notebook_generator_pro import generate_professional_notebook


class NotebookCleanlinessTest(unittest.TestCase):
    def test_svm_candidate_uses_supported_probability_calibration(self):
        models = dict(get_classification_models())
        candidate = models["Support Vector Machine"]

        self.assertIsInstance(candidate, CalibratedClassifierCV)
        self.assertIsInstance(candidate.estimator, SVC)
        self.assertIsNot(candidate.estimator.probability, True)

        features = np.array([
            [-2.0],
            [-1.6],
            [-1.2],
            [-0.8],
            [-0.4],
            [-0.1],
            [0.1],
            [0.4],
            [0.8],
            [1.2],
            [1.6],
            [2.0],
        ])
        target = np.array([0] * 6 + [1] * 6)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r".*`probability` parameter was deprecated.*",
                category=FutureWarning,
            )
            candidate.fit(features, target)
            probabilities = candidate.predict_proba(features[:2])

        self.assertEqual(probabilities.shape, (2, 2))
        json.dumps(_model_parameters(candidate))

    def test_generated_notebook_contains_no_deprecated_svc_probability_flag(self):
        notebook = generate_professional_notebook(
            {
                "target_column": "outcome",
                "task_type": "classification",
                "best_model": "Support Vector Machine",
                "best_score": 0.8,
                "features": {
                    "numeric": ["feature"],
                    "categorical": [],
                    "dropped": [],
                },
                "models_trained": [
                    {
                        "name": "Support Vector Machine",
                        "cv_score_mean": 0.8,
                        "cv_score_std": 0.02,
                    }
                ],
                "model_selection": {"folds": 3},
            },
            filename="clean_notebook.csv",
        )
        source = "\n".join(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
        )

        self.assertNotIn("probability=True", source)
        self.assertIn("from sklearn.calibration import CalibratedClassifierCV", source)
        self.assertIn("'Support Vector Machine': CalibratedClassifierCV(", source)

    def test_generated_notebook_handles_an_all_numeric_dataset(self):
        notebook = generate_professional_notebook(
            {
                "target_column": "outcome",
                "task_type": "classification",
                "best_model": "Logistic Regression",
                "features": {
                    "numeric": ["signal", "age"],
                    "categorical": [],
                    "dropped": [],
                },
                "models_trained": [{"name": "Logistic Regression"}],
            },
            filename="numeric_only.csv",
        )
        source = "\n".join(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "code"
        )

        self.assertIn("if not categorical_columns.empty", source)
        self.assertIn("else pd.DataFrame()", source)


if __name__ == "__main__":
    unittest.main()
