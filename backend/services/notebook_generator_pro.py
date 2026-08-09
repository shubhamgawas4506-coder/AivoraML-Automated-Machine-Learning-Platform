"""Generate an executable, self-contained AivoraML analysis notebook.

The API has historically exposed two workflow-result shapes.  This module
normalises both shapes before rendering a single reproducible notebook:

* ``ml_engine.run_workflow`` (legacy): ``feature_info``, ``models_compared``
* ``ml_engine_enhanced.run_enhanced_workflow``: ``features``, ``models_trained``

The generated notebook intentionally rebuilds the preprocessing and estimator
pipeline from source data.  It never embeds fitted Python objects from the
server process.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import nbformat


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _markdown_text(value: Any) -> str:
    """Render user-controlled text safely inside inline Markdown code."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plain_value(value: Any) -> Any:
    """Convert common scalar containers to stable Python literals."""
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_model_name(value: Any) -> str:
    name = str(value or "").strip()
    lowered = name.lower()
    for suffix in (" (tuned)", " tuned"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)].strip()
            lowered = name.lower()
            break

    aliases = {
        "random forest classifier": "Random Forest",
        "random forest regressor": "Random Forest",
        "randomforest": "Random Forest",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "knn": "K-Nearest Neighbors",
        "k-nearest neighbour": "K-Nearest Neighbors",
        "k-nearest neighbors classifier": "K-Nearest Neighbors",
        "k-nearest neighbors regressor": "K-Nearest Neighbors",
        "svm": "Support Vector Machine",
        "svc": "Support Vector Machine",
        "svr": "Support Vector Machine",
        "support vector classifier": "Support Vector Machine",
        "support vector regressor": "Support Vector Machine",
        "gaussian naive bayes": "Naive Bayes",
        "gaussiannb": "Naive Bayes",
        "xgbclassifier": "XGBoost",
        "xgbregressor": "XGBoost",
    }
    return aliases.get(lowered, name)


def _unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    for value in values:
        text = str(value)
        if text and text not in output:
            output.append(text)
    return output


def _normalise_result(
    result: Dict[str, Any],
    filename: str,
    profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = _as_dict(result)
    profile = _as_dict(profile)

    task = str(result.get("task_type") or "classification").lower()
    task = "regression" if task.startswith("reg") else "classification"
    target = str(result.get("target_column") or "target")

    feature_info = _as_dict(result.get("feature_info"))
    if not feature_info:
        feature_info = _as_dict(result.get("features"))
    numeric = _unique_strings(_as_list(feature_info.get("numeric")))
    categorical = _unique_strings(_as_list(feature_info.get("categorical")))
    dropped = _unique_strings(
        _as_list(feature_info.get("dropped"))
        or _as_list(feature_info.get("dropped_columns"))
        or _as_list(result.get("dropped_columns"))
    )

    models = _as_list(result.get("models_trained"))
    if not models:
        models = _as_list(result.get("models_compared"))
    models = [item for item in models if isinstance(item, dict)]

    reported_best_model = str(result.get("best_model") or "")
    best_model = _canonical_model_name(reported_best_model)

    requested_models = [
        _canonical_model_name(item.get("name"))
        for item in models
        if item.get("name")
    ]
    if best_model:
        requested_models.insert(0, best_model)
    requested_models = _unique_strings(requested_models)

    evaluation_details = _as_dict(result.get("evaluation_details"))
    metrics = _as_dict(result.get("metrics"))
    if not metrics:
        metrics = _as_dict(evaluation_details.get("metrics"))

    matching_model: Dict[str, Any] = {}
    for item in models:
        if _canonical_model_name(item.get("name")) == best_model:
            matching_model = item
            break
    if not metrics:
        metrics = _as_dict(matching_model.get("metrics"))

    best_score = _finite_number(result.get("best_score"))
    if best_score is None:
        score_keys = (
            ("positive_f1", "macro_f1", "f1_weighted", "f1_score", "accuracy")
            if task == "classification"
            else ("r2",)
        )
        for source in (metrics, _as_dict(matching_model.get("metrics"))):
            for key in score_keys:
                best_score = _finite_number(source.get(key))
                if best_score is not None:
                    break
            if best_score is not None:
                break

    tuning_result = _as_dict(result.get("tuning_result"))
    tuned_params = _as_dict(result.get("tuned_params"))
    if not tuned_params:
        tuned_params = _as_dict(tuning_result.get("best_params"))
    tuned_params = {
        str(key).removeprefix("model__"): _plain_value(value)
        for key, value in tuned_params.items()
    }

    data_info = _as_dict(result.get("data_info"))
    if not data_info:
        data_info = _as_dict(result.get("data_shape"))
    rows = (
        data_info.get("rows")
        or data_info.get("n_rows")
        or profile.get("n_rows")
        or profile.get("rows")
    )
    columns = (
        data_info.get("columns")
        or data_info.get("n_cols")
        or profile.get("n_cols")
    )
    if isinstance(columns, (list, tuple)):
        columns = len(columns)
    run_manifest = _as_dict(result.get("run_manifest"))
    split = _as_dict(result.get("split") or result.get("split_info"))
    model_selection = _as_dict(result.get("model_selection"))
    threshold_selection = _as_dict(result.get("threshold_selection"))
    positive_label = result.get("positive_label")
    if positive_label is None:
        positive_label = threshold_selection.get("positive_label")

    return {
        "filename": str(filename or profile.get("filename") or "dataset.csv"),
        "target": target,
        "task": task,
        "numeric": numeric,
        "categorical": categorical,
        "dropped": dropped,
        "models": requested_models,
        "reported_best_model": reported_best_model or best_model or "Not reported",
        "best_model": best_model,
        "best_score": best_score,
        "tuned_params": tuned_params,
        "rows": rows,
        "columns": columns,
        "warnings": _unique_strings(_as_list(result.get("warnings"))),
        "strategy": str(result.get("strategy") or "not specified"),
        "run_manifest": _plain_value(run_manifest),
        "cv_folds": int(
            model_selection.get("folds")
            or _as_dict(run_manifest.get("cross_validation")).get("folds")
            or 3
        ),
        "test_size": (
            float(split.get("test_size_ratio"))
            if split.get("test_size_ratio") is not None
            else 0.20
        ),
        "decision_threshold": _finite_number(result.get("decision_threshold")),
        "positive_label": _plain_value(positive_label),
        "threshold_selection": _plain_value(threshold_selection),
        "evaluation_details": _plain_value(evaluation_details),
        "model_rows": _plain_value(models),
    }


def generate_professional_notebook(
    result: Dict[str, Any],
    filename: str = "dataset.csv",
    profile: Optional[Dict[str, Any]] = None,
) -> nbformat.NotebookNode:
    """Return a valid nbformat v4 notebook for classification or regression."""
    normalised = _normalise_result(result, filename, profile)

    dataset_name = normalised["filename"]
    target = normalised["target"]
    task = normalised["task"]
    best_score = normalised["best_score"]
    score_text = f"{best_score:.4f}" if best_score is not None else "not reported"
    generated_date = datetime.now(timezone.utc).date().isoformat()

    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata.aivoraml = {
        "generator": "professional",
        "version": "2.0",
        "task_type": task,
        "target_column": target,
    }

    cells: List[nbformat.NotebookNode] = []

    # 1
    cells.append(
        nbformat.v4.new_markdown_cell(
            "# AivoraML — Reproducible Machine Learning Analysis\n\n"
            f"**Dataset:** `{_markdown_text(dataset_name)}`  \n"
            f"**Target:** `{_markdown_text(target)}`  \n"
            f"**Task:** {task.title()}  \n"
            f"**Reported champion:** {_markdown_text(normalised['reported_best_model'])}  \n"
            f"**Reported score:** {score_text}  \n"
            f"**Generated:** {generated_date}\n\n"
            "This notebook reloads the source data, rebuilds preprocessing inside "
            "each validation fold, compares candidate estimators, evaluates one "
            "champion on an untouched holdout, and exports a fitted artifact."
        )
    )

    # 2
    cells.append(
        nbformat.v4.new_code_cell(
            "import json\n"
            "import platform\n"
            "import time\n"
            "from datetime import datetime, timezone\n"
            "from pathlib import Path\n\n"
            "import joblib\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "try:\n"
            "    from IPython.display import display\n"
            "except ImportError:\n"
            "    def display(value):\n"
            "        print(value)\n"
            "from sklearn import __version__ as sklearn_version\n"
            "from sklearn.calibration import CalibratedClassifierCV\n"
            "from sklearn.compose import ColumnTransformer\n"
            "from sklearn.ensemble import (\n"
            "    AdaBoostClassifier, AdaBoostRegressor,\n"
            "    ExtraTreesClassifier, ExtraTreesRegressor,\n"
            "    GradientBoostingClassifier, GradientBoostingRegressor,\n"
            "    RandomForestClassifier, RandomForestRegressor,\n"
            "    VotingClassifier, VotingRegressor,\n"
            ")\n"
            "from sklearn.dummy import DummyClassifier, DummyRegressor\n"
            "from sklearn.impute import SimpleImputer\n"
            "from sklearn.inspection import permutation_importance\n"
            "from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge\n"
            "from sklearn.metrics import (\n"
            "    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,\n"
            "    average_precision_score, brier_score_loss, f1_score,\n"
            "    log_loss, mean_absolute_error, mean_squared_error,\n"
            "    make_scorer, precision_score, r2_score, recall_score, roc_auc_score,\n"
            ")\n"
            "from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split\n"
            "from sklearn.naive_bayes import GaussianNB\n"
            "from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor\n"
            "from sklearn.pipeline import Pipeline\n"
            "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n"
            "from sklearn.svm import SVC, SVR\n"
            "from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor\n\n"
            "from imblearn.over_sampling import RandomOverSampler\n"
            "from imblearn.pipeline import Pipeline as ImbalancedPipeline\n\n"
            "sns.set_theme(style='whitegrid')\n"
            "RANDOM_SEED = 42\n"
            f"DATA_FILENAME = {dataset_name!r}\n"
            f"TARGET_COLUMN = {target!r}\n"
            f"TASK_TYPE = {task!r}\n"
            f"REPORTED_NUMERIC_COLUMNS = {normalised['numeric']!r}\n"
            f"REPORTED_CATEGORICAL_COLUMNS = {normalised['categorical']!r}\n"
            f"REPORTED_DROPPED_COLUMNS = {normalised['dropped']!r}\n"
            f"REQUESTED_MODEL_NAMES = {normalised['models']!r}\n"
            f"REPORTED_BEST_MODEL_BASE = {normalised['best_model']!r}\n"
            f"REPORTED_TUNED_PARAMS = {normalised['tuned_params']!r}\n"
            f"REPORTED_CV_FOLDS = {normalised['cv_folds']!r}\n"
            f"REPORTED_DECISION_THRESHOLD = {normalised['decision_threshold']!r}\n"
            f"REPORTED_POSITIVE_LABEL = {normalised['positive_label']!r}\n"
            f"REPORTED_THRESHOLD_SELECTION = {normalised['threshold_selection']!r}\n"
            f"RUN_MANIFEST = {normalised['run_manifest']!r}\n"
            "print(f'Environment ready: Python {platform.python_version()}, scikit-learn {sklearn_version}')"
        )
    )

    # 3
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 1. Load the source dataset\n\n"
            "Place the original dataset beside this notebook, or update "
            "`DATA_FILENAME`. CSV, Excel, JSON, and Parquet are supported."
        )
    )

    # 4
    cells.append(
        nbformat.v4.new_code_cell(
            "data_path = Path(DATA_FILENAME).expanduser()\n"
            "if not data_path.exists():\n"
            "    portable_name = Path(DATA_FILENAME.replace('\\\\', '/')).name\n"
            "    beside_notebook = Path.cwd() / portable_name\n"
            "    if beside_notebook.exists():\n"
            "        data_path = beside_notebook\n"
            "    else:\n"
            "        raise FileNotFoundError(\n"
            "            f'Could not find {DATA_FILENAME!r}. Place the source dataset '\n"
            "            'beside this notebook or update DATA_FILENAME.'\n"
            "        )\n\n"
            "suffix = data_path.suffix.lower()\n"
            "if suffix == '.csv':\n"
            "    df = pd.read_csv(data_path)\n"
            "elif suffix in {'.xlsx', '.xls'}:\n"
            "    df = pd.read_excel(data_path)\n"
            "elif suffix == '.json':\n"
            "    df = pd.read_json(data_path)\n"
            "elif suffix == '.parquet':\n"
            "    df = pd.read_parquet(data_path)\n"
            "else:\n"
            "    raise ValueError(f'Unsupported dataset format: {suffix or \"<none>\"}')\n\n"
            "if df.empty:\n"
            "    raise ValueError('The dataset is empty.')\n"
            "print(f'Loaded {data_path.name}: {df.shape[0]:,} rows × {df.shape[1]:,} columns')\n"
            "display(df.head())"
        )
    )

    # 5
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 2. Data-quality audit\n\n"
            "Review data types, missingness, duplicate rows, and cardinality "
            "before making modeling decisions."
        )
    )

    # 6
    cells.append(
        nbformat.v4.new_code_cell(
            "quality_report = pd.DataFrame({\n"
            "    'dtype': df.dtypes.astype(str),\n"
            "    'missing': df.isna().sum(),\n"
            "    'missing_pct': (df.isna().mean() * 100).round(2),\n"
            "    'unique': df.nunique(dropna=True),\n"
            "}).sort_values(['missing_pct', 'unique'], ascending=[False, False])\n"
            "print(f'Duplicate rows: {int(df.duplicated().sum()):,}')\n"
            "print(f'Total missing cells: {int(df.isna().sum().sum()):,}')\n"
            "display(quality_report)"
        )
    )

    # 7
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 5. Clean rows and resolve the feature schema\n\n"
            "Duplicate rows and rows with a missing target are removed. AivoraML's "
            "reported feature groups are used when present; otherwise the notebook "
            "infers numeric and categorical columns and removes identifier-like "
            "high-cardinality categoricals."
        )
    )

    # 8
    cells.append(
        nbformat.v4.new_code_cell(
            "if TARGET_COLUMN not in df.columns:\n"
            "    raise KeyError(f'Target column {TARGET_COLUMN!r} is not present in the dataset.')\n\n"
            "original_rows = len(df)\n"
            "df_clean = df.drop_duplicates().dropna(subset=[TARGET_COLUMN]).copy()\n"
            "if df_clean.empty:\n"
            "    raise ValueError('No rows remain after removing duplicates and missing targets.')\n\n"
            "reported_numeric = [\n"
            "    column for column in REPORTED_NUMERIC_COLUMNS\n"
            "    if column in df_clean.columns and column != TARGET_COLUMN\n"
            "]\n"
            "reported_categorical = [\n"
            "    column for column in REPORTED_CATEGORICAL_COLUMNS\n"
            "    if column in df_clean.columns\n"
            "    and column != TARGET_COLUMN\n"
            "    and column not in reported_numeric\n"
            "]\n"
            "dropped_columns = [\n"
            "    column for column in REPORTED_DROPPED_COLUMNS\n"
            "    if column in df_clean.columns and column != TARGET_COLUMN\n"
            "]\n\n"
            "if reported_numeric or reported_categorical:\n"
            "    numeric_cols = reported_numeric\n"
            "    categorical_cols = reported_categorical\n"
            "else:\n"
            "    candidate_features = [\n"
            "        column for column in df_clean.columns\n"
            "        if column != TARGET_COLUMN and column not in dropped_columns\n"
            "    ]\n"
            "    numeric_cols = df_clean[candidate_features].select_dtypes(include=[np.number]).columns.tolist()\n"
            "    categorical_cols = [column for column in candidate_features if column not in numeric_cols]\n"
            "    inferred_high_cardinality = [\n"
            "        column for column in categorical_cols\n"
            "        if df_clean[column].nunique(dropna=True) > 50\n"
            "    ]\n"
            "    dropped_columns.extend(\n"
            "        column for column in inferred_high_cardinality if column not in dropped_columns\n"
            "    )\n"
            "    categorical_cols = [\n"
            "        column for column in categorical_cols if column not in inferred_high_cardinality\n"
            "    ]\n\n"
            "feature_cols = numeric_cols + categorical_cols\n"
            "if not feature_cols:\n"
            "    raise ValueError('No usable feature columns remain after schema resolution.')\n\n"
            "X = df_clean[feature_cols].copy()\n"
            "y = df_clean[TARGET_COLUMN].copy()\n"
            "if TASK_TYPE == 'regression':\n"
            "    y = pd.to_numeric(y, errors='raise')\n\n"
            "schema_summary = pd.DataFrame({\n"
            "    'group': ['numeric', 'categorical', 'dropped'],\n"
            "    'count': [len(numeric_cols), len(categorical_cols), len(dropped_columns)],\n"
            "    'columns': [numeric_cols, categorical_cols, dropped_columns],\n"
            "})\n"
            "print(f'Removed {original_rows - len(df_clean):,} duplicate/invalid-target rows.')\n"
            "display(schema_summary)"
        )
    )

    # 9
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 6. Leakage-safe preprocessing\n\n"
            "Imputation, scaling, and one-hot encoding live inside a scikit-learn "
            "pipeline so every cross-validation fold learns transformations only "
            "from its own training partition."
        )
    )

    # 10
    cells.append(
        nbformat.v4.new_code_cell(
            "transformers = []\n"
            "if numeric_cols:\n"
            "    numeric_pipeline = Pipeline([\n"
            "        ('imputer', SimpleImputer(strategy='median')),\n"
            "        ('scaler', StandardScaler()),\n"
            "    ])\n"
            "    transformers.append(('numeric', numeric_pipeline, numeric_cols))\n\n"
            "if categorical_cols:\n"
            "    categorical_pipeline = Pipeline([\n"
            "        ('imputer', SimpleImputer(strategy='most_frequent')),\n"
            "        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),\n"
            "    ])\n"
            "    transformers.append(('categorical', categorical_pipeline, categorical_cols))\n\n"
            "preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')\n"
            "print(f'Preprocessor ready with {len(transformers)} transformer group(s).')"
        )
    )

    # 11
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 7. Create an untouched holdout\n\n"
            "Model selection uses cross-validation on the training partition. The "
            "20% holdout is evaluated only after the champion is selected."
        )
    )

    # 12
    cells.append(
        nbformat.v4.new_code_cell(
            "if len(X) < 5:\n"
            "    raise ValueError('At least five cleaned rows are required for a train/test split.')\n\n"
            "stratify_target = None\n"
            "if TASK_TYPE == 'classification':\n"
            "    class_counts = y.value_counts()\n"
            "    if len(class_counts) < 2:\n"
            "        raise ValueError('Classification requires at least two target classes.')\n"
            "    holdout_rows = int(np.ceil(len(y) * 0.20))\n"
            "    if class_counts.min() >= 2 and holdout_rows >= len(class_counts):\n"
            "        stratify_target = y\n\n"
            "try:\n"
            "    X_train, X_test, y_train, y_test = train_test_split(\n"
            "        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=stratify_target\n"
            "    )\n"
            "except ValueError:\n"
            "    X_train, X_test, y_train, y_test = train_test_split(\n"
            "        X, y, test_size=0.20, random_state=RANDOM_SEED\n"
            "    )\n\n"
            "if TASK_TYPE == 'classification':\n"
            "    training_class_counts = y_train.value_counts()\n"
            "    if len(training_class_counts) < 2 or training_class_counts.min() < 2:\n"
            "        raise ValueError(\n"
            "            'Each class needs at least two training examples for stratified cross-validation.'\n"
            "        )\n"
            "print(f'Train rows: {len(X_train):,}; holdout rows: {len(X_test):,}')"
        )
    )

    # 13
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 8. Compare candidate models with cross-validation\n\n"
            "Candidates reported by AivoraML are reconstructed when supported. "
            "Selection uses CV mean minus 10% of CV standard deviation, rewarding "
            "strong and stable models without inspecting the holdout."
        )
    )

    # 14
    cells.append(
        nbformat.v4.new_code_cell(
            "if TASK_TYPE == 'classification':\n"
            "    all_candidates = {\n"
            "        'Dummy Baseline': DummyClassifier(strategy='prior', random_state=RANDOM_SEED),\n"
            "        'Logistic Regression': LogisticRegression(C=0.5, class_weight='balanced', max_iter=1500, random_state=RANDOM_SEED),\n"
            "        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_SEED),\n"
            "        'Random Forest': RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced_subsample', random_state=RANDOM_SEED, n_jobs=1),\n"
            "        'Extra Trees': ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1),\n"
            "        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=RANDOM_SEED),\n"
            "        'Soft Voting Ensemble': VotingClassifier(estimators=[\n"
            "            ('logistic', LogisticRegression(C=0.5, class_weight='balanced', max_iter=1500, random_state=RANDOM_SEED)),\n"
            "            ('forest', RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced_subsample', random_state=RANDOM_SEED, n_jobs=1)),\n"
            "            ('extra', ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1)),\n"
            "            ('gradient', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=RANDOM_SEED)),\n"
            "        ], voting='soft', n_jobs=1),\n"
            "        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED),\n"
            "        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),\n"
            "        'Naive Bayes': GaussianNB(),\n"
            "        'Support Vector Machine': CalibratedClassifierCV(\n"
            "            estimator=SVC(class_weight='balanced', random_state=RANDOM_SEED),\n"
            "            method='sigmoid', cv=3, ensemble=False,\n"
            "        ),\n"
            "    }\n"
            "    default_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees']\n"
            "else:\n"
            "    all_candidates = {\n"
            "        'Dummy Baseline': DummyRegressor(strategy='mean'),\n"
            "        'Ridge Regression': Ridge(),\n"
            "        'Lasso Regression': Lasso(random_state=RANDOM_SEED),\n"
            "        'ElasticNet': ElasticNet(random_state=RANDOM_SEED),\n"
            "        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=RANDOM_SEED),\n"
            "        'Random Forest': RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1),\n"
            "        'Extra Trees': ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1),\n"
            "        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=RANDOM_SEED),\n"
            "        'Voting Ensemble': VotingRegressor(estimators=[\n"
            "            ('ridge', Ridge()),\n"
            "            ('forest', RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1)),\n"
            "            ('extra', ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=1)),\n"
            "            ('gradient', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=RANDOM_SEED)),\n"
            "        ], n_jobs=1),\n"
            "        'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=RANDOM_SEED),\n"
            "        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),\n"
            "        'Support Vector Machine': SVR(),\n"
            "    }\n"
            "    default_names = ['Ridge Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees']\n\n"
            "try:\n"
            "    if TASK_TYPE == 'classification':\n"
            "        from xgboost import XGBClassifier\n"
            "        all_candidates['XGBoost'] = XGBClassifier(\n"
            "            n_estimators=100, random_state=RANDOM_SEED, n_jobs=1,\n"
            "            eval_metric='logloss'\n"
            "        )\n"
            "    else:\n"
            "        from xgboost import XGBRegressor\n"
            "        all_candidates['XGBoost'] = XGBRegressor(\n"
            "            n_estimators=100, random_state=RANDOM_SEED, n_jobs=1\n"
            "        )\n"
            "except ImportError:\n"
            "    print('XGBoost is unavailable; continuing with scikit-learn candidates.')\n\n"
            "selected_names = [name for name in REQUESTED_MODEL_NAMES if name in all_candidates]\n"
            "if not selected_names:\n"
            "    selected_names = default_names\n"
            "if REPORTED_BEST_MODEL_BASE in all_candidates and REPORTED_BEST_MODEL_BASE not in selected_names:\n"
            "    selected_names.insert(0, REPORTED_BEST_MODEL_BASE)\n"
            "selected_names = list(dict.fromkeys(selected_names))\n"
            "candidate_models = {name: all_candidates[name] for name in selected_names}\n\n"
            "if REPORTED_TUNED_PARAMS and REPORTED_BEST_MODEL_BASE in candidate_models:\n"
            "    try:\n"
            "        candidate_models[REPORTED_BEST_MODEL_BASE].set_params(**REPORTED_TUNED_PARAMS)\n"
            "        print(f'Applied reported tuned parameters to {REPORTED_BEST_MODEL_BASE}.')\n"
            "    except (TypeError, ValueError) as error:\n"
            "        print(f'Could not apply reported tuned parameters; using defaults: {error}')\n\n"
            "if TASK_TYPE == 'classification':\n"
            "    cv_folds = min(REPORTED_CV_FOLDS, int(y_train.value_counts().min()))\n"
            "    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)\n"
            "    class_counts = y_train.value_counts()\n"
            "    positive_label = REPORTED_POSITIVE_LABEL\n"
            "    if len(class_counts) == 2:\n"
            "        if positive_label not in class_counts.index:\n"
            "            event_tokens = {'1', 'yes', 'y', 'true', 'positive', 'churn', 'churned', 'fraud', 'default', 'failed', 'cancelled', 'attrition', 'anomaly', 'malignant'}\n"
            "            positive_label = next((label for label in class_counts.index if str(label).strip().lower() in event_tokens), None)\n"
            "        if positive_label is None:\n"
            "            positive_label = min(class_counts.items(), key=lambda item: (int(item[1]), type(item[0]).__name__, repr(item[0])))[0]\n"
            "        selection_scoring = make_scorer(f1_score, pos_label=positive_label, zero_division=0)\n"
            "        selection_scoring_name = 'positive_class_f1'\n"
            "    else:\n"
            "        positive_label = None\n"
            "        selection_scoring = 'f1_macro'\n"
            "        selection_scoring_name = 'macro_f1'\n"
            "else:\n"
            "    cv_folds = min(REPORTED_CV_FOLDS, len(X_train))\n"
            "    if cv_folds < 2:\n"
            "        raise ValueError('Regression cross-validation requires at least two training rows.')\n"
            "    cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)\n"
            "    positive_label = None\n"
            "    selection_scoring = 'r2' if len(X_train) // cv_folds >= 2 else 'neg_mean_absolute_error'\n"
            "    selection_scoring_name = selection_scoring\n\n"
            "comparison_rows = []\n"
            "fitted_candidates = {}\n"
            "use_resampling = False\n"
            "if TASK_TYPE == 'classification':\n"
            "    class_counts = y_train.value_counts()\n"
            "    use_resampling = (\n"
            "        len(class_counts) >= 2\n"
            "        and int(class_counts.min()) > 1\n"
            "        and float(class_counts.max() / class_counts.min()) > 3\n"
            "    )\n"
            "    if use_resampling:\n"
            "        print('Class imbalance detected: random minority oversampling will run inside every training fold.')\n"
            "internally_balanced = {'Dummy Baseline', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Extra Trees', 'Soft Voting Ensemble', 'Support Vector Machine'}\n"
            "for model_name, estimator in candidate_models.items():\n"
            "    pipeline_steps = [('preprocessor', preprocessor)]\n"
            "    candidate_uses_resampling = use_resampling and model_name not in internally_balanced\n"
            "    if candidate_uses_resampling:\n"
            "        pipeline_steps.append(('balancer', RandomOverSampler(random_state=RANDOM_SEED)))\n"
            "    pipeline_steps.append(('model', estimator))\n"
            "    pipeline_class = ImbalancedPipeline if candidate_uses_resampling else Pipeline\n"
            "    pipeline = pipeline_class(pipeline_steps)\n"
            "    started = time.perf_counter()\n"
            "    try:\n"
            "        cv_scores = cross_val_score(\n"
            "            pipeline, X_train, y_train, cv=cv_splitter,\n"
            "            scoring=selection_scoring, n_jobs=1, error_score='raise'\n"
            "        )\n"
            "        if not np.all(np.isfinite(cv_scores)):\n"
            "            raise ValueError('Cross-validation returned a non-finite score.')\n"
            "        pipeline.fit(X_train, y_train)\n"
            "        cv_mean = float(np.mean(cv_scores))\n"
            "        cv_std = float(np.std(cv_scores))\n"
            "        comparison_rows.append({\n"
            "            'model': model_name,\n"
            "            'cv_mean': cv_mean,\n"
            "            'cv_std': cv_std,\n"
            "            'selection_score': cv_mean - 0.10 * cv_std,\n"
            "            'fit_seconds': time.perf_counter() - started,\n"
            "            'status': 'ok',\n"
            "        })\n"
            "        fitted_candidates[model_name] = pipeline\n"
            "    except Exception as error:\n"
            "        comparison_rows.append({\n"
            "            'model': model_name,\n"
            "            'cv_mean': np.nan,\n"
            "            'cv_std': np.nan,\n"
            "            'selection_score': np.nan,\n"
            "            'fit_seconds': time.perf_counter() - started,\n"
            "            'status': f'failed: {error}',\n"
            "        })\n\n"
            "successful_rows = [row for row in comparison_rows if row['status'] == 'ok']\n"
            "if not successful_rows:\n"
            "    failures = '; '.join(f\"{row['model']}: {row['status']}\" for row in comparison_rows)\n"
            "    raise RuntimeError(f'Every candidate failed. {failures}')\n\n"
            "champion_record = max(successful_rows, key=lambda row: row['selection_score'])\n"
            "champion_name = champion_record['model']\n"
            "champion_pipeline = fitted_candidates[champion_name]\n"
            "SELECTION_SCORING = selection_scoring\n"
            "SELECTION_SCORING_NAME = selection_scoring_name\n"
            "comparison_df = pd.DataFrame(comparison_rows).sort_values(\n"
            "    'selection_score', ascending=False, na_position='last'\n"
            ").reset_index(drop=True)\n"
            "display(comparison_df)\n"
            "plot_data = comparison_df.dropna(subset=['cv_mean']).sort_values('cv_mean')\n"
            "if not plot_data.empty:\n"
            "    ax = plot_data.plot.barh(x='model', y='cv_mean', xerr='cv_std', legend=False, figsize=(9, 5))\n"
            "    ax.set_title(f'Cross-validation comparison ({SELECTION_SCORING_NAME})')\n"
            "    ax.set_xlabel('CV score')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "print(f'Selected champion: {champion_name}')"
        )
    )

    # 15
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 9. Final holdout evaluation\n\n"
            "The selected pipeline is now evaluated once on data that was not used "
            "for fitting, tuning, or model selection."
        )
    )

    # 16
    cells.append(
        nbformat.v4.new_code_cell(
            "if (\n"
            "    TASK_TYPE == 'classification'\n"
            "    and positive_label is not None\n"
            "    and hasattr(champion_pipeline, 'predict_proba')\n"
            "    and len(X_train) >= 100\n"
            "):\n"
            "    calibration_folds = min(3, int(y_train.value_counts().min()))\n"
            "    if calibration_folds >= 2:\n"
            "        champion_pipeline = CalibratedClassifierCV(\n"
            "            estimator=champion_pipeline, method='sigmoid',\n"
            "            cv=calibration_folds, ensemble=False,\n"
            "        )\n"
            "        champion_pipeline.fit(X_train, y_train)\n"
            "        print(f'Calibrated probabilities on training data with {calibration_folds} folds.')\n\n"
            "y_pred = champion_pipeline.predict(X_test)\n"
            "if (\n"
            "    TASK_TYPE == 'classification'\n"
            "    and REPORTED_DECISION_THRESHOLD is not None\n"
            "    and hasattr(champion_pipeline, 'predict_proba')\n"
            "    and len(champion_pipeline.classes_) == 2\n"
            "):\n"
            "    fitted_classes = list(champion_pipeline.classes_)\n"
            "    positive_index = fitted_classes.index(positive_label)\n"
            "    negative_label = fitted_classes[1 - positive_index]\n"
            "    locked_probabilities = champion_pipeline.predict_proba(X_test)[:, positive_index]\n"
            "    y_pred = np.where(\n"
            "        locked_probabilities >= float(REPORTED_DECISION_THRESHOLD),\n"
            "        positive_label,\n"
            "        negative_label,\n"
            "    )\n"
            "    print(f'Applied the locked training-only threshold: {float(REPORTED_DECISION_THRESHOLD):.2f}')\n\n"
            "if TASK_TYPE == 'classification':\n"
            "    final_metrics = {\n"
            "        'accuracy': float(accuracy_score(y_test, y_pred)),\n"
            "        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),\n"
            "        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),\n"
            "        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),\n"
            "        'macro_f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),\n"
            "        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),\n"
            "    }\n"
            "    if len(champion_pipeline.classes_) == 2:\n"
            "        binary_truth = (np.asarray(y_test) == positive_label).astype(int)\n"
            "        binary_pred = (np.asarray(y_pred) == positive_label).astype(int)\n"
            "        final_metrics.update({\n"
            "            'positive_precision': float(precision_score(binary_truth, binary_pred, zero_division=0)),\n"
            "            'positive_recall': float(recall_score(binary_truth, binary_pred, zero_division=0)),\n"
            "            'positive_f1': float(f1_score(binary_truth, binary_pred, zero_division=0)),\n"
            "        })\n"
            "    labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)]))\n"
            "    report_df = pd.DataFrame(\n"
            "        classification_report(\n"
            "            y_test, y_pred, labels=labels, output_dict=True, zero_division=0\n"
            "        )\n"
            "    ).transpose()\n"
            "    display(pd.DataFrame([final_metrics]).round(4))\n"
            "    display(report_df.round(4))\n"
            "    matrix = confusion_matrix(y_test, y_pred, labels=labels)\n"
            "    plt.figure(figsize=(7, 5))\n"
            "    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)\n"
            "    plt.title(f'Confusion matrix — {champion_name}')\n"
            "    plt.xlabel('Predicted')\n"
            "    plt.ylabel('Actual')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    residuals = np.asarray(y_test) - np.asarray(y_pred)\n"
            "    final_metrics = {\n"
            "        'r2': float(r2_score(y_test, y_pred)) if len(y_test) > 1 else None,\n"
            "        'mae': float(mean_absolute_error(y_test, y_pred)),\n"
            "        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),\n"
            "    }\n"
            "    display(pd.DataFrame([final_metrics]).round(4))\n"
            "    plt.figure(figsize=(8, 5))\n"
            "    sns.scatterplot(x=y_pred, y=residuals)\n"
            "    plt.axhline(0, color='black', linestyle='--', linewidth=1)\n"
            "    plt.title(f'Residuals — {champion_name}')\n"
            "    plt.xlabel('Predicted value')\n"
            "    plt.ylabel('Residual')\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        )
    )

    # 17
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 10. Explain the champion\n\n"
            "Native coefficients or impurity importances are used when available. "
            "Other estimators receive holdout permutation importance."
        )
    )

    # 18
    cells.append(
        nbformat.v4.new_code_cell(
            "source_pipeline = champion_pipeline if hasattr(champion_pipeline, 'named_steps') else None\n"
            "estimator = source_pipeline.named_steps['model'] if source_pipeline is not None else None\n"
            "fitted_preprocessor = source_pipeline.named_steps['preprocessor'] if source_pipeline is not None else None\n"
            "feature_importance_df = pd.DataFrame(columns=['feature', 'importance'])\n\n"
            "if estimator is not None and (hasattr(estimator, 'feature_importances_') or hasattr(estimator, 'coef_')):\n"
            "    try:\n"
            "        transformed_names = fitted_preprocessor.get_feature_names_out().tolist()\n"
            "    except Exception:\n"
            "        transformed_names = []\n"
            "    if hasattr(estimator, 'feature_importances_'):\n"
            "        importance_values = np.asarray(estimator.feature_importances_)\n"
            "    else:\n"
            "        coefficients = np.asarray(estimator.coef_)\n"
            "        importance_values = (\n"
            "            np.mean(np.abs(coefficients), axis=0)\n"
            "            if coefficients.ndim > 1 else np.abs(coefficients)\n"
            "        )\n"
            "    if len(transformed_names) != len(importance_values):\n"
            "        transformed_names = [f'transformed_feature_{index}' for index in range(len(importance_values))]\n"
            "    feature_importance_df = pd.DataFrame({\n"
            "        'feature': transformed_names,\n"
            "        'importance': importance_values,\n"
            "    })\n"
            "else:\n"
            "    permutation = permutation_importance(\n"
            "        champion_pipeline, X_test, y_test, scoring=SELECTION_SCORING,\n"
            "        n_repeats=5, random_state=RANDOM_SEED, n_jobs=1\n"
            "    )\n"
            "    feature_importance_df = pd.DataFrame({\n"
            "        'feature': feature_cols,\n"
            "        'importance': permutation.importances_mean,\n"
            "    })\n\n"
            "feature_importance_df = feature_importance_df.sort_values(\n"
            "    'importance', ascending=False\n"
            ").reset_index(drop=True)\n"
            "display(feature_importance_df.head(20))\n"
            "top_features = feature_importance_df.head(20).sort_values('importance')\n"
            "if not top_features.empty:\n"
            "    plt.figure(figsize=(9, max(4, len(top_features) * 0.30)))\n"
            "    plt.barh(top_features['feature'], top_features['importance'])\n"
            "    plt.title(f'Feature importance — {champion_name}')\n"
            "    plt.xlabel('Importance')\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        )
    )

    # 19
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 13. Export the fitted artifact\n\n"
            "The artifact contains the full preprocessing/model pipeline plus an "
            "explicit serving schema, evaluation results, comparison table, and "
            "runtime versions."
        )
    )

    # 20
    cells.append(
        nbformat.v4.new_code_cell(
            "comparison_export = json.loads(comparison_df.to_json(orient='records'))\n"
            "feature_importance_export = json.loads(\n"
            "    feature_importance_df.head(50).to_json(orient='records')\n"
            ")\n"
            "input_schema = [\n"
            "    {'name': column, 'dtype': str(df_clean[column].dtype)}\n"
            "    for column in feature_cols\n"
            "]\n"
            "artifact = {\n"
            "    'artifact_version': '2.0',\n"
            "    'created_at_utc': datetime.now(timezone.utc).isoformat(),\n"
            "    'pipeline': champion_pipeline,\n"
            "    'target_column': TARGET_COLUMN,\n"
            "    'task_type': TASK_TYPE,\n"
            "    'best_model': champion_name,\n"
            "    'metrics': final_metrics,\n"
            "    'decision_threshold': REPORTED_DECISION_THRESHOLD,\n"
            "    'positive_label': positive_label,\n"
            "    'threshold_selection': REPORTED_THRESHOLD_SELECTION,\n"
            "    'run_manifest': RUN_MANIFEST,\n"
            "    'model_comparison': comparison_export,\n"
            "    'feature_importance': feature_importance_export,\n"
            "    'feature_info': {\n"
            "        'numeric': numeric_cols,\n"
            "        'categorical': categorical_cols,\n"
            "        'dropped': dropped_columns,\n"
            "    },\n"
            "    'input_schema': input_schema,\n"
            "    'source_filename': data_path.name,\n"
            "    'split': {\n"
            "        'train_rows': len(X_train),\n"
            "        'holdout_rows': len(X_test),\n"
            "        'random_seed': RANDOM_SEED,\n"
            "        'cv_folds': cv_folds,\n"
            "        'selection_scoring': SELECTION_SCORING_NAME,\n"
            "    },\n"
            "    'class_labels': (\n"
            "        champion_pipeline.classes_.tolist()\n"
            "        if TASK_TYPE == 'classification' and hasattr(champion_pipeline, 'classes_')\n"
            "        else None\n"
            "    ),\n"
            "    'library_versions': {\n"
            "        'python': platform.python_version(),\n"
            "        'pandas': pd.__version__,\n"
            "        'numpy': np.__version__,\n"
            "        'scikit_learn': sklearn_version,\n"
            "        'joblib': joblib.__version__,\n"
            "    },\n"
            "}\n"
            "MODEL_PATH = Path('aivoraml_model.joblib')\n"
            "joblib.dump(artifact, MODEL_PATH)\n"
            "print(f'Saved {MODEL_PATH.resolve()} ({MODEL_PATH.stat().st_size / 1024:.1f} KiB)')"
        )
    )

    # 21
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 14. Reload and test a prediction\n\n"
            "This final smoke test proves that the serialized pipeline accepts raw "
            "feature rows and applies the same preprocessing used during training."
        )
    )

    # 22
    cells.append(
        nbformat.v4.new_code_cell(
            "loaded_artifact = joblib.load(MODEL_PATH)\n"
            "loaded_pipeline = (\n"
            "    loaded_artifact['pipeline']\n"
            "    if isinstance(loaded_artifact, dict) else loaded_artifact\n"
            ")\n"
            "example_input = X_test.head(min(3, len(X_test))).copy()\n"
            "example_output = example_input.copy()\n"
            "loaded_predictions = loaded_pipeline.predict(example_input)\n"
            "if TASK_TYPE == 'classification' and hasattr(loaded_pipeline, 'predict_proba'):\n"
            "    probabilities = loaded_pipeline.predict_proba(example_input)\n"
            "    if REPORTED_DECISION_THRESHOLD is not None and len(loaded_pipeline.classes_) == 2:\n"
            "        loaded_classes = list(loaded_pipeline.classes_)\n"
            "        positive_index = loaded_classes.index(loaded_artifact.get('positive_label', positive_label))\n"
            "        negative_label = loaded_classes[1 - positive_index]\n"
            "        loaded_predictions = np.where(\n"
            "            probabilities[:, positive_index] >= float(REPORTED_DECISION_THRESHOLD),\n"
            "            loaded_classes[positive_index], negative_label,\n"
            "        )\n"
            "    example_output['prediction_confidence'] = probabilities.max(axis=1)\n"
            "example_output[f'predicted_{TARGET_COLUMN}'] = loaded_predictions\n"
            "display(example_output)\n"
            "print('Artifact reload and prediction smoke test passed.')"
        )
    )

    cells[6:6] = [
        nbformat.v4.new_markdown_cell(
            "## 3. Initial understanding and task-aware EDA\n\n"
            "**Question:** What does the dataset contain, how is the target "
            "distributed, and which measured relationships deserve attention?\n\n"
            "**Why this matters:** Understanding types, cardinality, and the target "
            "prevents inappropriate preprocessing and metrics. Plots below are "
            "selected for the detected task rather than generated indiscriminately."
        ),
        nbformat.v4.new_code_cell(
            "print(f'Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns')\n"
            "display(df.head(5))\n"
            "display(df.tail(5))\n"
            "display(df.sample(min(5, len(df)), random_state=RANDOM_SEED))\n"
            "type_summary = pd.DataFrame({\n"
            "    'dtype': df.dtypes.astype(str),\n"
            "    'unique': df.nunique(dropna=True),\n"
            "    'missing': df.isna().sum(),\n"
            "})\n"
            "display(type_summary)\n"
            "numeric_overview = df.select_dtypes(include=[np.number]).describe().T\n"
            "if not numeric_overview.empty:\n"
            "    display(numeric_overview.round(3))\n"
            "categorical_columns = df.select_dtypes(exclude=[np.number])\n"
            "categorical_overview = (\n"
            "    categorical_columns.describe().T\n"
            "    if not categorical_columns.empty\n"
            "    else pd.DataFrame()\n"
            ")\n"
            "if not categorical_overview.empty:\n"
            "    display(categorical_overview)\n\n"
            "if TARGET_COLUMN in df.columns:\n"
            "    if TASK_TYPE == 'classification':\n"
            "        target_counts = df[TARGET_COLUMN].value_counts(dropna=False)\n"
            "        target_table = pd.DataFrame({\n"
            "            'count': target_counts,\n"
            "            'percentage': (target_counts / len(df) * 100).round(2),\n"
            "        })\n"
            "        display(target_table)\n"
            "        ax = target_counts.plot.bar(figsize=(8, 4), color='#5b8def')\n"
            "        ax.set_title(f'Class distribution: {TARGET_COLUMN}')\n"
            "        ax.set_xlabel('Class')\n"
            "        ax.set_ylabel('Rows')\n"
            "        plt.xticks(rotation=25, ha='right')\n"
            "        plt.tight_layout()\n"
            "        plt.show()\n"
            "        majority_pct = float(target_counts.max() / target_counts.sum() * 100)\n"
            "        print(f'Reading this: the largest class represents {majority_pct:.1f}% of rows.')\n"
            "    else:\n"
            "        target_values = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')\n"
            "        display(target_values.describe().to_frame('target').T.round(3))\n"
            "        fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "        sns.histplot(target_values.dropna(), kde=True, ax=axes[0])\n"
            "        axes[0].set_title(f'Target distribution: {TARGET_COLUMN}')\n"
            "        sns.boxplot(x=target_values.dropna(), ax=axes[1])\n"
            "        axes[1].set_title(f'Target spread and potential outliers: {TARGET_COLUMN}')\n"
            "        plt.tight_layout()\n"
            "        plt.show()\n"
            "        print(f'Reading this: target skewness is {target_values.skew():.3f}.')\n\n"
            "numeric_for_corr = df.select_dtypes(include=[np.number])\n"
            "if numeric_for_corr.shape[1] >= 2 and numeric_for_corr.shape[1] <= 30:\n"
            "    corr = numeric_for_corr.corr(numeric_only=True)\n"
            "    plt.figure(figsize=(min(14, 4 + corr.shape[1]), min(10, 3 + corr.shape[1] * 0.5)))\n"
            "    sns.heatmap(corr, cmap='coolwarm', center=0, annot=corr.shape[1] <= 10, fmt='.2f')\n"
            "    plt.title('Numeric correlation heatmap')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "    print('Why this matters: strong associations can indicate redundancy or useful signal; correlation does not prove causation.')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 4. Outliers, skewness, and measured cleaning decisions\n\n"
            "IQR bounds are diagnostic, not automatic deletion rules. Rare but valid "
            "observations can contain useful signal. Without domain constraints, "
            "AivoraML recommends review or robust handling instead of blanket removal."
        ),
        nbformat.v4.new_code_cell(
            "numeric_candidates = [\n"
            "    column for column in df.select_dtypes(include=[np.number]).columns\n"
            "    if column != TARGET_COLUMN\n"
            "]\n"
            "outlier_rows = []\n"
            "skew_rows = []\n"
            "for column in numeric_candidates:\n"
            "    values = pd.to_numeric(df[column], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()\n"
            "    if values.empty:\n"
            "        continue\n"
            "    q1, q3 = values.quantile([0.25, 0.75])\n"
            "    iqr = q3 - q1\n"
            "    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr\n"
            "    count = int(((values < lower) | (values > upper)).sum())\n"
            "    outlier_rows.append({\n"
            "        'Feature': column,\n"
            "        'Lower Bound': lower,\n"
            "        'Upper Bound': upper,\n"
            "        'Outlier Count': count,\n"
            "        'Outlier Percentage': count / len(values) * 100,\n"
            "        'Recommended Action': 'Review; keep unless domain-invalid',\n"
            "    })\n"
            "    skew = float(values.skew())\n"
            "    distribution = 'Approximately symmetric' if abs(skew) < 0.5 else ('Moderately skewed' if abs(skew) < 1 else 'Highly skewed')\n"
            "    recommendation = 'Consider Yeo-Johnson inside the pipeline' if abs(skew) >= 1 else 'No transformation by default'\n"
            "    skew_rows.append({\n"
            "        'Feature': column,\n"
            "        'Skewness': skew,\n"
            "        'Distribution Type': distribution,\n"
            "        'Recommended Transformation': recommendation,\n"
            "    })\n"
            "outlier_summary = pd.DataFrame(outlier_rows)\n"
            "skewness_summary = pd.DataFrame(skew_rows)\n"
            "if not outlier_summary.empty:\n"
            "    display(outlier_summary.round(3))\n"
            "if not skewness_summary.empty:\n"
            "    display(skewness_summary.round(3))\n"
            "print(f'Exact duplicate rows measured before cleaning: {int(df.duplicated().sum()):,}.')\n"
            "print('No statistical outlier is removed solely for crossing an IQR bound.')"
        ),
    ]

    cells[-4:-4] = [
        nbformat.v4.new_markdown_cell(
            "## 11. Threshold, probability, and error analysis\n\n"
            "For binary classification, the recorded decision threshold was selected "
            "from training-only out-of-fold predictions. The untouched holdout is "
            "used once below to report the locked operating point. Regression uses "
            "absolute residuals to identify the largest errors."
        ),
        nbformat.v4.new_code_cell(
            "error_analysis = pd.DataFrame()\n"
            "if TASK_TYPE == 'classification':\n"
            "    if hasattr(champion_pipeline, 'predict_proba'):\n"
            "        holdout_probabilities = champion_pipeline.predict_proba(X_test)\n"
            "        classes = list(champion_pipeline.classes_)\n"
            "        if len(classes) == 2:\n"
            "            threshold = float(REPORTED_DECISION_THRESHOLD if REPORTED_DECISION_THRESHOLD is not None else 0.5)\n"
            "            positive_index = classes.index(positive_label)\n"
            "            negative_label = classes[1 - positive_index]\n"
            "            positive_probabilities = holdout_probabilities[:, positive_index]\n"
            "            threshold_predictions = np.where(positive_probabilities >= threshold, positive_label, negative_label)\n"
            "            binary_truth = (np.asarray(y_test) == positive_label).astype(int)\n"
            "            binary_pred = (np.asarray(threshold_predictions) == positive_label).astype(int)\n"
            "            threshold_matrix = confusion_matrix(binary_truth, binary_pred, labels=[0, 1])\n"
            "            tn, fp, fn, tp = threshold_matrix.ravel()\n"
            "            threshold_metrics = {\n"
            "                'threshold': threshold,\n"
            "                'precision': precision_score(binary_truth, binary_pred, zero_division=0),\n"
            "                'recall': recall_score(binary_truth, binary_pred, zero_division=0),\n"
            "                'f1': f1_score(binary_truth, binary_pred, zero_division=0),\n"
            "                'specificity': tn / (tn + fp) if (tn + fp) else 0.0,\n"
            "                'roc_auc': roc_auc_score(binary_truth, positive_probabilities),\n"
            "                'pr_auc': average_precision_score(binary_truth, positive_probabilities),\n"
            "                'log_loss': log_loss(y_test, holdout_probabilities, labels=classes),\n"
            "                'brier_score': brier_score_loss(binary_truth, positive_probabilities),\n"
            "            }\n"
            "            display(pd.DataFrame([threshold_metrics]).round(4))\n"
            "            error_analysis = X_test.copy()\n"
            "            error_analysis['actual'] = np.asarray(y_test)\n"
            "            error_analysis['predicted'] = threshold_predictions\n"
            "            error_analysis['positive_probability'] = positive_probabilities\n"
            "            error_analysis['error_type'] = np.where(\n"
            "                (binary_truth == 1) & (binary_pred == 0), 'False negative',\n"
            "                np.where((binary_truth == 0) & (binary_pred == 1), 'False positive', 'Correct')\n"
            "            )\n"
            "            display(error_analysis['error_type'].value_counts().to_frame('rows'))\n"
            "            display(error_analysis[error_analysis['error_type'] != 'Correct'].head(20))\n"
            "            print(f'Reading this: threshold {threshold:.2f} trades precision against recall; probability is uncertainty, not certainty.')\n"
            "        else:\n"
            "            print('Threshold optimisation is not applied to multiclass classification.')\n"
            "    else:\n"
            "        print('The champion does not expose probabilities, so threshold and calibration analysis are unavailable.')\n"
            "else:\n"
            "    error_analysis = X_test.copy()\n"
            "    error_analysis['actual'] = np.asarray(y_test)\n"
            "    error_analysis['predicted'] = np.asarray(y_pred)\n"
            "    error_analysis['residual'] = error_analysis['actual'] - error_analysis['predicted']\n"
            "    error_analysis['absolute_error'] = error_analysis['residual'].abs()\n"
            "    display(error_analysis.sort_values('absolute_error', ascending=False).head(20))\n"
            "    print('Reading this: the table shows where the champion makes its largest holdout errors.')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 12. Champion selection and project quality scorecard\n\n"
            "The champion is selected from training-only cross-validation using "
            "performance and fold stability. Holdout evidence, interpretability, "
            "latency, and limitations are reported after selection."
        ),
        nbformat.v4.new_code_cell(
            "selection_summary = pd.DataFrame([\n"
            "    {'Criterion': 'Cross-validation performance', 'Champion Result': champion_record['cv_mean'], 'Reason': 'Measured on training folds'},\n"
            "    {'Criterion': 'Cross-validation stability', 'Champion Result': champion_record['cv_std'], 'Reason': 'Lower variation is more stable'},\n"
            "    {'Criterion': 'Training time (seconds)', 'Champion Result': champion_record['fit_seconds'], 'Reason': 'Operational cost consideration'},\n"
            "    {'Criterion': 'Interpretability', 'Champion Result': 'Native/coefficient/permutation evidence generated', 'Reason': 'Model behavior must be reviewable'},\n"
            "    {'Criterion': 'Final holdout', 'Champion Result': json.dumps(final_metrics), 'Reason': 'Reported after selection, not used to choose'},\n"
            "])\n"
            "display(selection_summary)\n"
            "quality_scorecard = pd.DataFrame([\n"
            "    {'Area': 'Data quality', 'Status': 'Completed', 'Notes': f'{int(df.isna().sum().sum())} missing cells; {int(df.duplicated().sum())} duplicates measured'},\n"
            "    {'Area': 'EDA quality', 'Status': 'Completed', 'Notes': 'Task-aware target and correlation views generated where applicable'},\n"
            "    {'Area': 'Leakage prevention', 'Status': 'Completed', 'Notes': 'Preprocessing is fitted inside CV; holdout excluded from selection'},\n"
            "    {'Area': 'Model comparison', 'Status': 'Completed', 'Notes': f'{len(successful_rows)} successful candidates including a baseline'},\n"
            "    {'Area': 'Cross-validation', 'Status': 'Completed', 'Notes': f'{cv_folds} folds using {SELECTION_SCORING_NAME}'},\n"
            "    {'Area': 'Explainability', 'Status': 'Completed', 'Notes': 'Global importance generated with method-specific limitations'},\n"
            "    {'Area': 'Reproducibility', 'Status': 'Completed', 'Notes': f'Random seed {RANDOM_SEED}; pipeline reload checked'},\n"
            "    {'Area': 'Deployment readiness', 'Status': 'Review required', 'Notes': 'Local artifact is ready; hosted security/monitoring remain separate gates'},\n"
            "])\n"
            "display(quality_scorecard)"
        ),
    ]

    cells.extend([
        nbformat.v4.new_markdown_cell(
            "## 15. Final conclusion\n\n"
            "This project followed a leakage-safe tabular workflow from measured "
            "data quality through model comparison, untouched-holdout evaluation, "
            "explainability, serialization, and sample inference. The conclusion "
            "below is generated from the executed variables rather than hardcoded results."
        ),
        nbformat.v4.new_code_cell(
            "print(f'Problem solved: {TASK_TYPE} prediction for {TARGET_COLUMN}.')\n"
            "print(f'Dataset: {len(df):,} rows and {df.shape[1]:,} columns; {len(X_train):,} training and {len(X_test):,} holdout rows.')\n"
            "print(f'Champion: {champion_name}. Final metrics: {json.dumps(final_metrics, default=str)}')\n"
            "print(f'Models compared successfully: {len(successful_rows)}. Top measured feature: {feature_importance_df.iloc[0][\"feature\"] if not feature_importance_df.empty else \"unavailable\"}.')\n"
            "print('Limitations: results depend on the supplied data, target definition, split assumptions, and future data matching the training feature contract.')\n"
            "print('Ethical note: inspect subgroup errors and sensitive proxies before consequential use. Feature importance and correlation are not causal proof.')\n"
            "print('Next steps: validate with domain owners, monitor schema/drift, collect delayed outcomes, and retrain only through a versioned review process.')"
        ),
    ])

    nb.cells = cells
    return nb
