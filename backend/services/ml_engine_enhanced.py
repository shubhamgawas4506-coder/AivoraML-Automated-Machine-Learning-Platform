"""
Enhanced ML Engine - Multiple algorithms, better metrics, cross-validation
"""
import time
import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, List, Optional
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier, DummyRegressor

# Regression models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error,
    r2_score, classification_report, confusion_matrix,
    average_precision_score, brier_score_loss, log_loss,
    median_absolute_error, balanced_accuracy_score, make_scorer,
)

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available - install with: pip install xgboost")

RANDOM_SEED = 42
PERMUTATION_MAX_ROWS = 1000
PERMUTATION_MAX_FEATURES = 200
THRESHOLD_OPTIMIZATION_MAX_ROWS = 10000
CALIBRATION_MAX_BINS = 10


def _parallel_jobs(sample_count: Optional[int] = None) -> int:
    """Return a conservative parallelism budget for independent validation work.

    Estimators themselves remain single-threaded, which avoids nested CPU and
    memory amplification. Small datasets stay serial because process startup
    can cost more than the work being parallelized.
    """
    configured = os.getenv(
        "AIVORAML_ML_JOBS",
        os.getenv("REDIMENT_ML_JOBS", "2"),
    )
    try:
        requested = int(configured)
    except (TypeError, ValueError):
        requested = 2
    available = os.cpu_count() or 1
    jobs = max(1, min(requested, available, 4))
    if sample_count is not None and sample_count < 5000:
        return 1
    return jobs

POSITIVE_LABEL_TOKENS = {
    "1", "yes", "y", "true", "positive", "pos", "event",
    "churn", "churned", "fraud", "fraudulent", "default", "defaulted",
    "fail", "failed", "failure", "cancel", "canceled", "cancelled",
    "attrition", "attrited", "anomaly", "abnormal", "disease",
    "diseased", "malignant",
}


class WorkflowCancelled(RuntimeError):
    """Raised cooperatively between candidate fits when cancellation is requested."""


def _safe(v: Any) -> Any:
    """Make value JSON-safe"""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(float(v), 6)
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def _model_parameters(model: Any) -> Dict[str, Any]:
    """Return JSON-safe public parameters without mutating fitted estimators."""
    def public_value(value: Any) -> Any:
        if hasattr(value, "get_params"):
            return repr(value)
        if isinstance(value, dict):
            return {str(key): public_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [public_value(item) for item in value]
        return _safe(value)

    parameters = {}
    for key, value in model.get_params(deep=False).items():
        parameters[str(key)] = public_value(value)
    return parameters


def _choose_positive_label(y: Any) -> Optional[Any]:
    """Choose the event label explicitly for a binary classification target."""
    counts = pd.Series(y).value_counts(dropna=False)
    if len(counts) != 2:
        return None

    for label in counts.index:
        normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in POSITIVE_LABEL_TOKENS:
            return label

    # When labels are domain-neutral, treating the rarer outcome as the event
    # is the safest default for anomaly/churn-style workflows. Ties use a
    # stable representation so row order never changes the selected class.
    ranked = sorted(
        counts.items(),
        key=lambda item: (int(item[1]), type(item[0]).__name__, repr(item[0])),
    )
    return ranked[0][0]


def _class_index(labels: List[Any], target_label: Any) -> int:
    for index, label in enumerate(labels):
        try:
            if bool(label == target_label):
                return index
        except (TypeError, ValueError):
            continue
    raise ValueError(f"Positive label {target_label!r} is not present in fitted classes.")


def _selection_scorer(task_type: str, y: Any, positive_label: Any = None):
    if task_type == "regression":
        return "r2", "r2"
    if pd.Series(y).nunique(dropna=True) == 2 and positive_label is not None:
        return (
            make_scorer(f1_score, pos_label=positive_label, zero_division=0),
            "positive_class_f1",
        )
    return "f1_macro", "macro_f1"


def _classification_diagnostics(
    y_true: Any,
    y_pred: Any,
    labels: List[Any],
    y_pred_proba: Optional[np.ndarray] = None,
    positive_label: Any = None,
) -> Dict[str, Any]:
    """Return bounded holdout diagnostics without influencing model selection."""
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    matrix = confusion_matrix(truth, predictions, labels=labels)
    total = int(matrix.sum())
    per_class = {}

    for index, label in enumerate(labels):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = int(total - true_positive - false_negative - false_positive)
        per_class[str(label)] = {
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "true_negatives": true_negative,
        }

    error_counts: Dict[str, Any] = {
        "total_errors": int(np.sum(truth != predictions)),
        "correct_predictions": int(np.sum(truth == predictions)),
        "per_class": per_class,
    }
    if len(labels) == 2:
        positive_label = positive_label if positive_label is not None else labels[-1]
        positive_counts = per_class[str(positive_label)]
        error_counts.update({
            "positive_label": _safe(positive_label),
            "false_positives": positive_counts["false_positives"],
            "false_negatives": positive_counts["false_negatives"],
        })

    diagnostics: Dict[str, Any] = {
        "labels": _safe(labels),
        "confusion_matrix": _safe(matrix),
        "classification_report": _safe(classification_report(
            truth,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )),
        "error_counts": error_counts,
    }

    probabilities = None if y_pred_proba is None else np.asarray(y_pred_proba)
    if (
        len(labels) == 2
        and probabilities is not None
        and probabilities.ndim == 2
        and probabilities.shape[0] == len(truth)
        and probabilities.shape[1] >= 2
    ):
        positive_label = positive_label if positive_label is not None else labels[-1]
        positive_index = _class_index(labels, positive_label)
        positive_probabilities = np.clip(
            probabilities[:, positive_index].astype(float),
            0.0,
            1.0,
        )
        binary_truth = (truth == positive_label).astype(int)
        sample_size = len(binary_truth)
        # Keep roughly ten observations per bin where possible, while still
        # producing a useful two-bin preview for very small holdouts.
        bin_count = min(
            CALIBRATION_MAX_BINS,
            max(2, sample_size // 10),
        )
        bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
        bin_ids = np.digitize(
            positive_probabilities,
            bin_edges[1:-1],
            right=False,
        )
        calibration_bins = []
        weighted_gap = 0.0
        for bin_index in range(bin_count):
            mask = bin_ids == bin_index
            count = int(mask.sum())
            if not count:
                continue
            predicted_rate = float(np.mean(positive_probabilities[mask]))
            observed_rate = float(np.mean(binary_truth[mask]))
            gap = abs(predicted_rate - observed_rate)
            weighted_gap += (count / max(sample_size, 1)) * gap
            calibration_bins.append({
                "lower_bound": round(float(bin_edges[bin_index]), 4),
                "upper_bound": round(float(bin_edges[bin_index + 1]), 4),
                "mean_predicted_probability": round(predicted_rate, 6),
                "observed_positive_rate": round(observed_rate, 6),
                "calibration_gap": round(gap, 6),
                "count": count,
            })

        positive_count = int(binary_truth.sum())
        negative_count = int(sample_size - positive_count)
        diagnostics["probability_calibration"] = {
            "positive_label": _safe(positive_label),
            "brier_score": round(float(brier_score_loss(
                binary_truth,
                positive_probabilities,
            )), 6),
            "expected_calibration_error": round(weighted_gap, 6),
            "sample_size": sample_size,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "bins_requested": bin_count,
            "bins_populated": len(calibration_bins),
            "reliability": (
                "descriptive_only_small_holdout"
                if sample_size < 100 or min(positive_count, negative_count) < 20
                else "adequate_sample"
            ),
            "bins": calibration_bins,
        }

    return diagnostics


def _regression_diagnostics(y_true: Any, y_pred: Any) -> Dict[str, Any]:
    """Summarize signed, absolute, and scale-normalized holdout errors."""
    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    residuals = truth - predictions
    absolute_errors = np.abs(residuals)
    target_iqr = float(np.quantile(truth, 0.75) - np.quantile(truth, 0.25))
    target_std = float(np.std(truth))
    mean_absolute_target = float(np.mean(np.abs(truth)))
    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(np.mean(np.square(residuals))))

    def quantiles(values: np.ndarray, points: List[float]) -> Dict[str, float]:
        return {
            f"p{int(point * 100):02d}": round(float(np.quantile(values, point)), 6)
            for point in points
        }

    return {
        "residual_summary": {
            "mean": round(float(np.mean(residuals)), 6),
            "std": round(float(np.std(residuals)), 6),
            "q05": round(float(np.quantile(residuals, 0.05)), 6),
            "median": round(float(np.median(residuals)), 6),
            "q95": round(float(np.quantile(residuals, 0.95)), 6),
            **quantiles(residuals, [0.05, 0.25, 0.50, 0.75, 0.95]),
        },
        "absolute_error_quantiles": {
            **quantiles(absolute_errors, [0.50, 0.75, 0.90, 0.95, 0.99]),
            "maximum": round(float(np.max(absolute_errors)), 6),
        },
        "normalized_error": {
            "mae_over_target_iqr": (
                round(mae / target_iqr, 6) if target_iqr > 0 else None
            ),
            "rmse_over_target_std": (
                round(rmse / target_std, 6) if target_std > 0 else None
            ),
            "weighted_absolute_percentage_error": (
                round(mae / mean_absolute_target * 100.0, 4)
                if mean_absolute_target > 0
                else None
            ),
            "target_iqr": round(target_iqr, 6),
            "target_std": round(target_std, 6),
            "normalization_note": (
                "Lower is better. IQR and standard-deviation denominators are "
                "reported so normalized errors remain interpretable."
            ),
        },
    }


def _bounded_permutation_importance(
    pipeline: Any,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    feature_cols: List[str],
    task_type: str,
    positive_label: Any = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compute a deterministic, resource-bounded fallback importance."""
    feature_count = len(feature_cols)
    holdout_rows = len(X_holdout)
    metadata: Dict[str, Any] = {
        "method": "holdout_permutation_importance",
        "used_for_model_selection": False,
        "available_holdout_rows": holdout_rows,
        "feature_count": feature_count,
        "max_rows": PERMUTATION_MAX_ROWS,
        "max_features": PERMUTATION_MAX_FEATURES,
    }
    if holdout_rows < 8:
        metadata.update({
            "status": "skipped",
            "reason": "At least 8 holdout rows are required for stable permutation importance.",
        })
        return [], metadata
    if feature_count > PERMUTATION_MAX_FEATURES:
        metadata.update({
            "status": "skipped",
            "reason": (
                f"The {feature_count}-feature dataset exceeds the bounded "
                f"{PERMUTATION_MAX_FEATURES}-feature diagnostic limit."
            ),
        })
        return [], metadata

    sample_size = min(holdout_rows, PERMUTATION_MAX_ROWS)
    if sample_size < holdout_rows:
        random_state = np.random.RandomState(RANDOM_SEED)
        positions = np.sort(random_state.choice(
            holdout_rows,
            size=sample_size,
            replace=False,
        ))
        diagnostic_X = X_holdout.iloc[positions]
        diagnostic_y = y_holdout.iloc[positions]
    else:
        diagnostic_X = X_holdout
        diagnostic_y = y_holdout

    repeats = 5 if feature_count <= 20 and sample_size <= 500 else 3
    if feature_count > 100 or sample_size > 750:
        repeats = 2
    scoring, scoring_name = _selection_scorer(
        task_type,
        y_holdout,
        positive_label,
    )
    diagnostic_jobs = _parallel_jobs(sample_size)
    permutation = permutation_importance(
        pipeline,
        diagnostic_X,
        diagnostic_y,
        scoring=scoring,
        n_repeats=repeats,
        random_state=RANDOM_SEED,
        n_jobs=diagnostic_jobs,
    )
    pairs = sorted(
        zip(
            feature_cols,
            permutation.importances_mean,
            permutation.importances_std,
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    metadata.update({
        "status": "computed",
        "sample_rows": sample_size,
        "repeats": repeats,
        "scoring": scoring_name,
    })
    return [
        {
            "feature": str(name),
            "importance": round(float(mean), 6),
            "importance_std": round(float(std), 6),
        }
        for name, mean, std in pairs[:20]
    ], metadata


def get_classification_models() -> List[tuple]:
    """Get all classification models"""
    logistic = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=1500,
        random_state=RANDOM_SEED,
    )
    random_forest = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    extra_trees = ExtraTreesClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    gradient_boosting = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    models = [
        ("Logistic Regression", logistic),
        ("Decision Tree", DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
        )),
        ("Random Forest", random_forest),
        ("Extra Trees", extra_trees),
        ("Gradient Boosting", gradient_boosting),
        ("Soft Voting Ensemble", VotingClassifier(
            estimators=[
                ("logistic", logistic),
                ("forest", random_forest),
                ("extra", extra_trees),
                ("gradient", gradient_boosting),
            ],
            voting="soft",
            n_jobs=1,
        )),
        ("AdaBoost", AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Naive Bayes", GaussianNB()),
        (
            "Support Vector Machine",
            CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
        ),
    ]
    
    if HAS_XGBOOST:
        models.append(("XGBoost", XGBClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=1)))
    
    return models


def get_regression_models() -> List[tuple]:
    """Get all regression models"""
    ridge = Ridge(random_state=RANDOM_SEED)
    random_forest = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    extra_trees = ExtraTreesRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    gradient_boosting = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    models = [
        ("Ridge Regression", ridge),
        ("Lasso Regression", Lasso(random_state=RANDOM_SEED)),
        ("ElasticNet", ElasticNet(random_state=RANDOM_SEED)),
        ("Decision Tree", DecisionTreeRegressor(random_state=RANDOM_SEED, max_depth=10)),
        ("Random Forest", random_forest),
        ("Extra Trees", extra_trees),
        ("Gradient Boosting", gradient_boosting),
        ("Voting Ensemble", VotingRegressor(
            estimators=[
                ("ridge", ridge),
                ("forest", random_forest),
                ("extra", extra_trees),
                ("gradient", gradient_boosting),
            ],
            n_jobs=1,
        )),
        ("AdaBoost", AdaBoostRegressor(n_estimators=50, random_state=RANDOM_SEED)),
        ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5)),
        ("Support Vector Machine", SVR(kernel='rbf')),
    ]
    
    if HAS_XGBOOST:
        models.append(("XGBoost", XGBRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=1)))
    
    return models



def evaluate_classification(
    y_true,
    y_pred,
    y_pred_proba=None,
    positive_label: Any = None,
    class_labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Comprehensive classification metrics"""
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    }

    classes = list(class_labels) if class_labels is not None else list(np.unique(y_true))
    if len(classes) == 2:
        positive_label = positive_label if positive_label is not None else _choose_positive_label(y_true)
        positive_label = positive_label if positive_label is not None else classes[-1]
        binary_truth = (np.asarray(y_true) == positive_label).astype(int)
        binary_predictions = (np.asarray(y_pred) == positive_label).astype(int)
        metrics["positive_precision"] = round(
            float(precision_score(binary_truth, binary_predictions, zero_division=0)), 4
        )
        metrics["positive_recall"] = round(
            float(recall_score(binary_truth, binary_predictions, zero_division=0)), 4
        )
        metrics["positive_f1"] = round(
            float(f1_score(binary_truth, binary_predictions, zero_division=0)), 4
        )

    # Probability metrics for binary classification.
    if y_pred_proba is not None and len(classes) == 2:
        try:
            positive_index = _class_index(classes, positive_label)
            positive_probabilities = np.asarray(y_pred_proba)[:, positive_index]
            metrics["roc_auc"] = round(float(roc_auc_score(binary_truth, positive_probabilities)), 4)
            metrics["pr_auc"] = round(
                float(average_precision_score(binary_truth, positive_probabilities)),
                4,
            )
            metrics["log_loss"] = round(float(log_loss(y_true, y_pred_proba, labels=classes)), 4)
            metrics["brier_score"] = round(
                float(brier_score_loss(binary_truth, positive_probabilities)),
                4,
            )
            negative_label = next(label for label in classes if label != positive_label)
            matrix = confusion_matrix(y_true, y_pred, labels=[negative_label, positive_label])
            if matrix.shape == (2, 2):
                tn, fp, _, _ = matrix.ravel()
                metrics["specificity"] = round(
                    float(tn / (tn + fp)) if (tn + fp) else 0.0,
                    4,
                )
        except Exception:
            pass
    
    return metrics


def evaluate_regression(y_true, y_pred) -> Dict[str, Any]:
    """Comprehensive regression metrics"""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    non_zero = np.abs(truth) > np.finfo(float).eps
    mape = (
        float(np.mean(np.abs((truth[non_zero] - predictions[non_zero]) / truth[non_zero])) * 100)
        if np.any(non_zero)
        else None
    )
    target_iqr = float(np.quantile(truth, 0.75) - np.quantile(truth, 0.25))
    target_std = float(np.std(truth))
    mean_absolute_target = float(np.mean(np.abs(truth)))
    
    metrics = {
        "mae": round(mae, 4),
        "median_ae": round(float(median_absolute_error(y_true, y_pred)), 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "normalized_mae_iqr": (
            round(mae / target_iqr, 4) if target_iqr > 0 else None
        ),
        "normalized_rmse_std": (
            round(rmse / target_std, 4) if target_std > 0 else None
        ),
        "wape": (
            round(mae / mean_absolute_target * 100.0, 2)
            if mean_absolute_target > 0
            else None
        ),
    }
    
    if mape is not None:
        metrics["mape"] = round(mape, 2)
    
    return metrics


def run_enhanced_workflow(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "auto",
    max_models: int = 10,
    tune_best: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Enhanced ML workflow with multiple models
    """
    result = {
        "target_column": target_column,
        "task_type": task_type,
        "data_shape": {"rows": len(df), "columns": len(df.columns)},
        "models_trained": [],
        "best_model": None,
        "feature_importance": [],
        "warnings": []
    }
    
    # Detect task type
    if task_type == "auto":
        task_type = _detect_task_type(df, target_column)
        result["task_type"] = task_type
    
    # Clean data
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=[target_column])
    if task_type == "regression":
        if not pd.api.types.is_numeric_dtype(df_clean[target_column]):
            raise ValueError("Regression requires a numeric target column.")
        finite_target = np.isfinite(pd.to_numeric(df_clean[target_column], errors="coerce"))
        removed_non_finite = int((~finite_target).sum())
        if removed_non_finite:
            df_clean = df_clean.loc[finite_target].copy()
            result["warnings"].append(
                f"Removed {removed_non_finite} rows with a non-finite target."
            )
    if len(df_clean) < 10:
        raise ValueError("At least 10 usable rows are required for model training.")
    if df_clean[target_column].nunique(dropna=True) < 2:
        raise ValueError("The target column must contain at least two distinct values.")
    if task_type == "classification":
        target_counts = df_clean[target_column].value_counts()
        if len(target_counts) > 100 or len(target_counts) / len(df_clean) > 0.5:
            raise ValueError(
                "The selected target has too many classes and looks like an identifier."
            )
        if int(target_counts.min()) < 2:
            raise ValueError("Every target class needs at least two rows.")
    
    result["cleaning"] = {
        "original_rows": len(df),
        "cleaned_rows": len(df_clean),
        "rows_removed": len(df) - len(df_clean)
    }
    
    # Prepare features
    feature_cols = [c for c in df_clean.columns if c != target_column]
    dropped_constant = [
        col for col in feature_cols
        if df_clean[col].nunique(dropna=False) <= 1
    ]
    dropped_identifiers = [
        col for col in feature_cols
        if (
            (
                str(col).lower() in {"id", "uuid", "identifier", "index"}
                or str(col).lower().endswith("_id")
                or str(col).lower().startswith("id_")
            )
            and df_clean[col].nunique(dropna=True) / max(len(df_clean), 1) > 0.8
        )
    ]
    initially_dropped = set(dropped_constant + dropped_identifiers)
    feature_cols = [col for col in feature_cols if col not in initially_dropped]
    numeric_cols = df_clean[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean[feature_cols].select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    if numeric_cols:
        df_clean[numeric_cols] = df_clean[numeric_cols].replace(
            [np.inf, -np.inf],
            np.nan,
        )
    
    # Drop high-cardinality categoricals
    dropped_cols = list(initially_dropped)
    for col in categorical_cols[:]:
        if df_clean[col].nunique() > 50:
            categorical_cols.remove(col)
            feature_cols.remove(col)
            dropped_cols.append(col)

    # Bound dense one-hot expansion to protect the local workspace from
    # high-cardinality memory blowups.
    encoded_levels = sum(max(1, int(df_clean[col].nunique())) for col in categorical_cols)
    if encoded_levels > 1000:
        for col in sorted(
            categorical_cols,
            key=lambda name: int(df_clean[name].nunique()),
            reverse=True,
        ):
            if encoded_levels <= 1000:
                break
            encoded_levels -= max(1, int(df_clean[col].nunique()))
            categorical_cols.remove(col)
            feature_cols.remove(col)
            dropped_cols.append(col)
    
    if dropped_cols:
        result["warnings"].append(
            f"Dropped constant, identifier, or high-cardinality columns: "
            f"{', '.join(map(str, dropped_cols))}"
        )
    
    result["features"] = {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "dropped": list(map(str, dropped_cols)),
        "total": len(numeric_cols) + len(categorical_cols),
        "encoded_category_levels": encoded_levels,
    }
    
    # Build preprocessor
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_cols))
    
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols))
    
    if not transformers:
        result["error"] = "No usable features found"
        return result
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # Split data
    X = df_clean[feature_cols]
    y = df_clean[target_column]
    positive_label = (
        _choose_positive_label(y)
        if task_type == "classification"
        else None
    )
    if positive_label is not None:
        result["positive_label"] = _safe(positive_label)
    selection_scoring, selection_metric = _selection_scorer(
        task_type,
        y,
        positive_label,
    )
    
    stratify = y if task_type == "classification" else None
    stratified_used = stratify is not None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )
        stratified_used = False
        result["warnings"].append("Stratified split failed, used random split")
    
    result["split"] = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split_ratio": "80/20",
        "random_seed": RANDOM_SEED,
        "stratified": stratified_used,
        "selection_data": "cross-validation on the training partition",
        "final_evaluation_data": "untouched 20% holdout",
    }
    result["diagnostic_safeguards"] = {
        "holdout_rows": len(X_test),
        "calibration_max_bins": CALIBRATION_MAX_BINS,
        "permutation_importance_max_rows": PERMUTATION_MAX_ROWS,
        "permutation_importance_max_features": PERMUTATION_MAX_FEATURES,
        "threshold_optimization_max_training_rows": THRESHOLD_OPTIMIZATION_MAX_ROWS,
        "holdout_used_for_model_or_tuning_selection": False,
    }

    if task_type == "classification":
        cv_splits = min(3, int(y_train.value_counts().min()))
    else:
        cv_splits = min(3, max(2, len(X_train) // 3))
    cv_splits = cv_splits if cv_splits >= 2 else None
    
    # ═══════════════════════════════════════════════════════════════
    # HANDLE IMBALANCED DATA
    # ═══════════════════════════════════════════════════════════════
    # Random oversampling duplicates observed minority rows without creating
    # fractional one-hot categories. Keeping it inside the pipeline prevents
    # validation leakage.
    use_resampling = False
    imbalanced_pipeline_class = None
    sampler_class = None
    if task_type == "classification":
        class_counts = y_train.value_counts()
        if len(class_counts) >= 2 and class_counts.min() > 1:
            imbalance_ratio = float(class_counts.max() / class_counts.min())
            if imbalance_ratio > 3:
                try:
                    from imblearn.over_sampling import RandomOverSampler
                    from imblearn.pipeline import Pipeline as ImbalancedPipeline

                    use_resampling = True
                    sampler_class = RandomOverSampler
                    imbalanced_pipeline_class = ImbalancedPipeline
                    result["data_balancing"] = (
                        "Candidate-aware class weights with fold-local minority "
                        "oversampling for estimators without native weighting"
                    )
                    result["class_distribution"] = {
                        str(label): int(count)
                        for label, count in class_counts.items()
                    }
                except ImportError:
                    result["warnings"].append(
                        "Class imbalance detected; install imbalanced-learn to enable fold-safe oversampling."
                    )
    
    # Get models
    models = get_classification_models() if task_type == "classification" else get_regression_models()
    if len(X_train) > 20000:
        skipped_families = {"Support Vector Machine", "K-Nearest Neighbors"}
        models = [
            (name, model)
            for name, model in models
            if name not in skipped_families
        ]
        result["warnings"].append(
            "Skipped support-vector and nearest-neighbor models because the "
            "training set exceeds 20,000 rows."
        )
    models = models[:max_models]  # Limit non-trivial candidates.
    baseline = (
        ("Dummy Baseline", DummyClassifier(strategy="prior", random_state=RANDOM_SEED))
        if task_type == "classification"
        else ("Dummy Baseline", DummyRegressor(strategy="mean"))
    )
    models = [baseline, *models]
    
    # Select with cross-validation on training data. The holdout is reported,
    # but never used to choose a winner.
    best_selection_score = -np.inf
    best_name = ""
    best_pipeline = None
    best_predictions = None
    best_probabilities = None
    
    total_candidates = len(models)
    internally_balanced_models = {
        "Dummy Baseline",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "Soft Voting Ensemble",
        "Support Vector Machine",
    }
    for candidate_index, (name, model) in enumerate(models):
        if is_cancelled is not None and is_cancelled():
            raise WorkflowCancelled("Training cancelled between model candidates.")
        if progress_callback is not None:
            progress_callback(candidate_index, total_candidates, name)
        try:
            print(f"Training {name}...")
            
            # Create a leakage-safe preprocessing, balancing, and model pipeline.
            pipeline_steps = [("preprocessor", preprocessor)]
            candidate_uses_resampling = (
                use_resampling and name not in internally_balanced_models
            )
            if candidate_uses_resampling:
                pipeline_steps.append((
                    "balancer",
                    sampler_class(random_state=RANDOM_SEED),
                ))
            pipeline_steps.append(("model", model))
            pipeline_class = (
                imbalanced_pipeline_class if candidate_uses_resampling else Pipeline
            )
            pipeline = pipeline_class(pipeline_steps)

            validation_started = time.perf_counter()
            try:
                if not cv_splits:
                    raise ValueError("Not enough rows per fold")
                cv = (
                    StratifiedKFold(
                        n_splits=cv_splits,
                        shuffle=True,
                        random_state=RANDOM_SEED,
                    )
                    if task_type == "classification"
                    else KFold(
                        n_splits=cv_splits,
                        shuffle=True,
                        random_state=RANDOM_SEED,
                    )
                )
                validation_jobs = _parallel_jobs(len(X_train))
                validation = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=selection_scoring,
                    n_jobs=validation_jobs,
                    return_train_score=True,
                )
                cv_scores = np.asarray(validation["test_score"])
                train_cv_scores = np.asarray(validation["train_score"])
                cv_mean = float(np.mean(cv_scores))
                cv_std = float(np.std(cv_scores))
                training_score = float(np.mean(train_cv_scores))
                training_seconds = float(np.sum(validation["fit_time"]))
                inference_seconds = float(np.sum(validation["score_time"]))
            except Exception as exc:
                print(f"CV failed for {name}: {exc}")
                raise ValueError(f"cross-validation failed: {exc}") from exc
            validation_seconds = time.perf_counter() - validation_started

            if task_type == "classification":
                training_metric_name = (
                    "positive_f1" if positive_label is not None else "macro_f1"
                )
                training_metrics = {training_metric_name: round(training_score, 4)}
                metrics = {
                    selection_metric: round(cv_mean, 4),
                    # Kept for compatibility with older saved result readers.
                    "f1_score": round(cv_mean, 4),
                }
            else:
                training_metrics = {"r2": round(training_score, 4)}
                metrics = {"r2": round(cv_mean, 4)}

            overfitting_gap = (
                float(training_score - cv_mean)
                if cv_mean is not None
                else None
            )
            selection_score = (
                float(cv_mean - 0.10 * cv_std)
                if cv_mean is not None and cv_std is not None
                else -np.inf
            )
            fit_assessment = (
                "overfitting risk"
                if overfitting_gap is not None and overfitting_gap > 0.12
                else "underfitting risk"
                if training_score < 0.55
                else "balanced fit"
            )

            model_result = {
                "name": name,
                "metrics": _safe(metrics),
                "training_metrics": _safe(training_metrics),
                "training_score": round(float(training_score), 4),
                "holdout_score": None,
                "cv_score_mean": round(cv_mean, 4) if cv_mean is not None else None,
                "cv_score_std": round(cv_std, 4) if cv_std is not None else None,
                "cv_scores": (
                    [round(float(score), 6) for score in cv_scores]
                    if cv_mean is not None
                    else []
                ),
                "cv_score_min": (
                    round(float(np.min(cv_scores)), 4)
                    if cv_mean is not None
                    else None
                ),
                "cv_score_max": (
                    round(float(np.max(cv_scores)), 4)
                    if cv_mean is not None
                    else None
                ),
                "selection_score": round(selection_score, 4),
                "overfitting_gap": (
                    round(overfitting_gap, 4)
                    if overfitting_gap is not None
                    else None
                ),
                "fit_assessment": fit_assessment,
                "metric_source": "cross_validation",
                "selection_metric": selection_metric,
                "balancing": (
                    "fold_local_random_oversampling"
                    if candidate_uses_resampling
                    else "native_class_weight_or_unweighted"
                ),
                "training_time_seconds": round(training_seconds, 4),
                "validation_time_seconds": round(validation_seconds, 4),
                "inference_time_ms": round(
                    inference_seconds / max(len(X_train), 1) * 1000,
                    4,
                ),
                "is_best": False,
                "parameters": _model_parameters(model),
            }
            result["models_trained"].append(model_result)

            if progress_callback is not None:
                progress_callback(candidate_index + 1, total_candidates, name)

            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_name = name
                best_pipeline = pipeline
        
        except WorkflowCancelled:
            raise
        except Exception as e:
            print(f"Error training {name}: {e}")
            result["warnings"].append(f"Failed to train {name}: {str(e)}")
            continue
    
    if is_cancelled is not None and is_cancelled():
        raise WorkflowCancelled("Training cancelled after candidate comparison.")

    # Cross-validation already fits every losing candidate on every fold. Only
    # the selected champion needs one full training-partition fit for final
    # calibration/evaluation and export.
    if best_pipeline is not None:
        champion_fit_started = time.perf_counter()
        best_pipeline.fit(X_train, y_train)
        champion_fit_seconds = time.perf_counter() - champion_fit_started
        for model_result in result["models_trained"]:
            if model_result["name"] == best_name:
                model_result["final_fit_time_seconds"] = round(
                    champion_fit_seconds,
                    4,
                )
                break

    # Mark best model
    for model_result in result["models_trained"]:
        if model_result["name"] == best_name:
            model_result["is_best"] = True
            break
    
    result["best_model"] = best_name
    result["best_score"] = None
    result["model_selection"] = {
        "method": "cross-validation on training data",
        "folds": cv_splits,
        "metric": selection_metric,
        "criterion": (
            f"{selection_metric.replace('_', ' ')} CV mean minus "
            "0.10 x CV standard deviation"
        ),
        "selection_score": (
            round(best_selection_score, 4)
            if np.isfinite(best_selection_score)
            else None
        ),
        "holdout_used_for_selection": False,
    }
    
    # ═══════════════════════════════════════════════════════════════
    # HYPERPARAMETER TUNING (Boost performance by 2-5%)
    # ═══════════════════════════════════════════════════════════════
    if tune_best and best_pipeline is not None and len(X_train) > 100:
        try:
            if is_cancelled is not None and is_cancelled():
                raise WorkflowCancelled("Training cancelled before tuning.")
            if progress_callback is not None:
                progress_callback(total_candidates, total_candidates, f"Tuning {best_name}")
            print(f"\nTuning hyperparameters for {best_name}...")
            tuned_result = _tune_hyperparameters(
                best_pipeline,
                X_train,
                y_train,
                task_type,
                best_name,
                positive_label,
            )
            
            if tuned_result and tuned_result["improved"]:
                best_pipeline = tuned_result["pipeline"]
                result["best_model"] = best_name + " (Tuned)"
                result["tuning_improvement"] = round(
                    (
                        tuned_result["cv_score_after"]
                        - tuned_result["cv_score_before"]
                    ) * 100,
                    2,
                )
                result["tuned_params"] = tuned_result["best_params"]
                result["model_selection"]["selection_score"] = tuned_result[
                    "cv_score_after"
                ]
                print(
                    f"Tuning improved cross-validation score by "
                    f"+{result['tuning_improvement']} points"
                )
            else:
                print("Tuning did not improve performance, keeping original")
                
        except WorkflowCancelled:
            raise
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            result["warnings"].append(f"Hyperparameter tuning skipped: {str(e)}")

    # Calibrate binary probabilities entirely within the training partition.
    # Thresholding still happens afterward from outer out-of-fold predictions.
    if (
        task_type == "classification"
        and positive_label is not None
        and best_pipeline is not None
        and hasattr(best_pipeline, "predict_proba")
        and len(X_train) >= 100
    ):
        calibration_folds = min(3, int(pd.Series(y_train).value_counts().min()))
        if calibration_folds >= 2:
            try:
                calibrated_pipeline = CalibratedClassifierCV(
                    estimator=best_pipeline,
                    method="sigmoid",
                    cv=calibration_folds,
                    ensemble=False,
                )
                calibrated_pipeline.fit(X_train, y_train)
                best_pipeline = calibrated_pipeline
                result["probability_calibration_training"] = {
                    "method": "sigmoid",
                    "cv_folds": calibration_folds,
                    "data": "training_partition_only",
                    "holdout_used": False,
                }
            except Exception as exc:
                result["warnings"].append(
                    f"Training-only probability calibration was unavailable: {exc}"
                )

    decision_threshold = None
    threshold_selection = None
    if best_pipeline is not None:
        if is_cancelled is not None and is_cancelled():
            raise WorkflowCancelled("Training cancelled before final evaluation.")
        best_predictions = best_pipeline.predict(X_test)
        best_probabilities = None
        if task_type == "classification" and hasattr(best_pipeline, "predict_proba"):
            try:
                best_probabilities = best_pipeline.predict_proba(X_test)
                fitted_classes = list(getattr(best_pipeline, "classes_", []))
                if len(fitted_classes) == 2 and cv_splits:
                    threshold_positive_label = (
                        positive_label
                        if positive_label is not None
                        else fitted_classes[-1]
                    )
                    positive_index = _class_index(
                        fitted_classes,
                        threshold_positive_label,
                    )
                    threshold_X = X_train
                    threshold_y = y_train
                    if len(X_train) > THRESHOLD_OPTIMIZATION_MAX_ROWS:
                        sample_positions, _ = train_test_split(
                            np.arange(len(X_train)),
                            train_size=THRESHOLD_OPTIMIZATION_MAX_ROWS,
                            random_state=RANDOM_SEED,
                            stratify=np.asarray(y_train),
                        )
                        sample_positions = np.sort(sample_positions)
                        threshold_X = X_train.iloc[sample_positions]
                        threshold_y = y_train.iloc[sample_positions]
                    threshold_cv_splits = min(
                        cv_splits,
                        int(pd.Series(threshold_y).value_counts().min()),
                    )
                    if threshold_cv_splits < 2:
                        raise ValueError(
                            "Not enough rows per class for threshold optimization."
                        )
                    threshold_cv = StratifiedKFold(
                        n_splits=threshold_cv_splits,
                        shuffle=True,
                        random_state=RANDOM_SEED,
                    )
                    threshold_jobs = _parallel_jobs(len(threshold_X))
                    oof_probabilities = cross_val_predict(
                        best_pipeline,
                        threshold_X,
                        threshold_y,
                        cv=threshold_cv,
                        method="predict_proba",
                        n_jobs=threshold_jobs,
                    )[:, positive_index]
                    negative_label = next(
                        label
                        for label in fitted_classes
                        if label != threshold_positive_label
                    )
                    binary_train = (
                        np.asarray(threshold_y) == threshold_positive_label
                    ).astype(int)
                    analysis_rows = []
                    for threshold in np.linspace(0.05, 0.95, 91):
                        threshold_predictions = (
                            oof_probabilities >= threshold
                        ).astype(int)
                        matrix = confusion_matrix(
                            binary_train,
                            threshold_predictions,
                            labels=[0, 1],
                        )
                        tn, fp, _, _ = matrix.ravel()
                        analysis_rows.append({
                            "threshold": round(float(threshold), 2),
                            "precision": float(precision_score(
                                binary_train,
                                threshold_predictions,
                                zero_division=0,
                            )),
                            "recall": float(recall_score(
                                binary_train,
                                threshold_predictions,
                                zero_division=0,
                            )),
                            "f1": float(f1_score(
                                binary_train,
                                threshold_predictions,
                                zero_division=0,
                            )),
                            "specificity": (
                                float(tn / (tn + fp))
                                if (tn + fp)
                                else 0.0
                            ),
                        })
                    chosen = max(
                        analysis_rows,
                        key=lambda row: (
                            row["f1"],
                            row["recall"],
                            -abs(row["threshold"] - 0.5),
                        ),
                    )
                    decision_threshold = float(chosen["threshold"])
                    best_predictions = np.where(
                        best_probabilities[:, positive_index] >= decision_threshold,
                        threshold_positive_label,
                        negative_label,
                    )
                    threshold_selection = {
                        "method": "training-only out-of-fold F1 maximization",
                        "positive_label": _safe(threshold_positive_label),
                        "threshold": decision_threshold,
                        "cv_folds": threshold_cv_splits,
                        "training_rows_available": len(X_train),
                        "training_rows_used": len(threshold_X),
                        "sampled_for_resource_safety": len(threshold_X) < len(X_train),
                        "holdout_used_for_selection": False,
                        "selected_validation_metrics": {
                            key: round(float(value), 4)
                            for key, value in chosen.items()
                            if key != "threshold"
                        },
                        "curve": [
                            {
                                key: round(float(value), 4)
                                for key, value in row.items()
                            }
                            for row in analysis_rows
                            if (
                                int(round(row["threshold"] * 100)) % 5 == 0
                                or row["threshold"] == decision_threshold
                            )
                        ],
                    }
            except Exception as exc:
                result["warnings"].append(
                    f"Threshold optimization was unavailable: {exc}"
                )

    if best_pipeline is not None and best_predictions is not None:
        if task_type == "classification":
            labels = list(getattr(best_pipeline, "classes_", np.unique(y_test)))
            final_metrics = evaluate_classification(
                y_test,
                best_predictions,
                best_probabilities,
                positive_label=positive_label,
                class_labels=labels,
            )
            classification_diagnostics = _classification_diagnostics(
                y_test,
                best_predictions,
                labels,
                best_probabilities,
                positive_label,
            )
            probability_calibration = classification_diagnostics.get(
                "probability_calibration"
            )
            error_counts = classification_diagnostics["error_counts"]
            if len(labels) == 2:
                final_metrics["false_positives"] = error_counts[
                    "false_positives"
                ]
                final_metrics["false_negatives"] = error_counts[
                    "false_negatives"
                ]
            if probability_calibration:
                final_metrics["expected_calibration_error"] = (
                    probability_calibration["expected_calibration_error"]
                )
            result["best_score"] = final_metrics[
                "positive_f1" if positive_label is not None else "macro_f1"
            ]
            result["evaluation_details"] = {
                "metrics": final_metrics,
                **classification_diagnostics,
            }
            result["decision_threshold"] = decision_threshold
            result["threshold_selection"] = threshold_selection
        else:
            final_metrics = evaluate_regression(y_test, best_predictions)
            result["best_score"] = final_metrics["r2"]
            result["evaluation_details"] = {
                "metrics": final_metrics,
                **_regression_diagnostics(y_test, best_predictions),
            }
    
    # Feature importance for best model
    if best_pipeline is not None:
        try:
            model = (
                best_pipeline.named_steps.get("model")
                if hasattr(best_pipeline, "named_steps")
                else None
            )
            native_method = None
            if model is not None and hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_)
                native_method = "native_feature_importance"
            elif model is not None and hasattr(model, "coef_"):
                coefficients = np.asarray(model.coef_)
                importances = (
                    np.mean(np.abs(coefficients), axis=0)
                    if coefficients.ndim > 1
                    else np.abs(coefficients)
                )
                native_method = "absolute_native_coefficients"
            else:
                importances = None

            native_importance_computed = False
            if importances is not None:
                # Build feature names
                feature_names = []
                if numeric_cols:
                    feature_names.extend(numeric_cols)
                if categorical_cols:
                    try:
                        fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
                        cat_step = fitted_preprocessor.named_transformers_["cat"]
                        encoder = cat_step.named_steps["encoder"]
                        cat_names = encoder.get_feature_names_out(categorical_cols).tolist()
                        feature_names.extend(cat_names)
                    except Exception:
                        feature_names.extend([f"cat_{i}" for i in range(len(importances) - len(numeric_cols))])
                
                if len(feature_names) == len(importances):
                    importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
                    result["feature_importance"] = [
                        {"feature": name, "importance": round(float(imp), 6)}
                        for name, imp in importance_pairs[:20]
                    ]
                    result["feature_importance_diagnostics"] = {
                        "method": native_method,
                        "status": "computed",
                        "feature_space": "post-preprocessing",
                        "used_for_model_selection": False,
                    }
                    native_importance_computed = True

            if not native_importance_computed:
                fallback_importance, fallback_metadata = (
                    _bounded_permutation_importance(
                        best_pipeline,
                        X_test,
                        y_test,
                        feature_cols,
                        task_type,
                        positive_label,
                    )
                )
                result["feature_importance"] = fallback_importance
                result["feature_importance_diagnostics"] = fallback_metadata
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            result["feature_importance_diagnostics"] = {
                "status": "unavailable",
                "reason": str(e),
                "used_for_model_selection": False,
            }
            result["warnings"].append(
                f"Feature-importance diagnostics were unavailable: {e}"
            )
    
    # Store pipeline for download
    result["_pipeline"] = best_pipeline
    result["_preprocessor"] = preprocessor
    result["run_manifest"] = {
        "schema_version": 1,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "task_type": task_type,
        "target_column": target_column,
        "positive_label": _safe(positive_label),
        "feature_schema": _safe(result.get("features")),
        "cleaning": _safe(result.get("cleaning")),
        "split": _safe(result.get("split")),
        "cross_validation": {
            "folds": cv_splits,
            "scoring": selection_metric,
            "parallel_jobs": _parallel_jobs(len(X_train)),
        },
        "candidate_models": [
            {
                "name": item.get("name"),
                "parameters": item.get("parameters"),
                "cv_scores": item.get("cv_scores"),
            }
            for item in result.get("models_trained", [])
        ],
        "selection": _safe(result.get("model_selection")),
        "best_model": result.get("best_model"),
        "tuned_params": _safe(result.get("tuned_params")),
        "decision_threshold": decision_threshold,
        "threshold_selection": _safe(threshold_selection),
        "probability_calibration_training": _safe(
            result.get("probability_calibration_training")
        ),
    }
    
    return _safe(result)


def _detect_task_type(df: pd.DataFrame, target: str) -> str:
    """Detect classification or regression"""
    col = df[target]
    if pd.api.types.is_numeric_dtype(col):
        if col.nunique() <= 20:
            return "classification"
        return "regression"
    return "classification"



def _tune_hyperparameters(
    pipeline: Pipeline,
    X_train,
    y_train,
    task_type: str,
    model_name: str,
    positive_label: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Tune exclusively inside the training partition.

    The final holdout is intentionally not accepted here, so tuning cannot
    optimize against the reported test result.
    """
    from sklearn.model_selection import RandomizedSearchCV
    
    # Define search spaces for different models
    param_distributions = {}
    
    if "Random Forest" in model_name:
        param_distributions = {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [10, 20, 30, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        }
    elif "Gradient Boosting" in model_name:
        param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7, 9],
            "model__subsample": [0.8, 0.9, 1.0],
        }
    elif "XGBoost" in model_name:
        param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, 9],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
        }
    elif "Extra Trees" in model_name:
        param_distributions = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [10, 20, 30, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    else:
        # No tuning for other models
        return None
    
    if not param_distributions:
        return None
    
    scoring, scoring_name = _selection_scorer(
        task_type,
        y_train,
        positive_label,
    )
    if task_type == "classification":
        cv_folds = min(5, int(pd.Series(y_train).value_counts().min()))
        if cv_folds < 2:
            return None
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=RANDOM_SEED,
        )
    else:
        cv_folds = min(5, max(2, len(X_train) // 50))
        cv = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

    tuning_jobs = _parallel_jobs(len(X_train))
    baseline_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=tuning_jobs,
    )
    baseline_cv_mean = float(np.mean(baseline_scores))

    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=12,
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_SEED,
        n_jobs=tuning_jobs,
        verbose=0,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    tuned_cv_score = float(search.best_score_)
    improved = bool(tuned_cv_score > baseline_cv_mean + 0.001)

    best_params = {
        k.replace("model__", ""): v 
        for k, v in search.best_params_.items()
    }

    return {
        "cv_score_before": round(baseline_cv_mean, 4),
        "cv_score_after": round(tuned_cv_score, 4),
        "improved": improved,
        "best_params": _safe(best_params),
        "cv_folds": cv_folds,
        "scoring": scoring_name,
        "pipeline": search.best_estimator_ if improved else pipeline,
    }
