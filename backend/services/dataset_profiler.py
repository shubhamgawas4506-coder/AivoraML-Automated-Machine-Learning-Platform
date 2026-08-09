"""Dataset profiling for AivoraML's AI-native ML workspace."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _serialise(obj: Any) -> Any:
    """Convert pandas and NumPy values into JSON-safe Python values."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_serialise(value) for value in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _serialise(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(value) for value in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def profile_dataset(df: pd.DataFrame, filename: str = "") -> Dict[str, Any]:
    """Return a rich, JSON-safe profile for an uploaded DataFrame."""
    missing = df.isnull().sum()
    n_rows = int(len(df))
    n_cols = int(len(df.columns))
    duplicate_count = int(df.duplicated().sum())
    total_missing = int(missing.sum())
    total_cells = max(n_rows * n_cols, 1)

    profile: Dict[str, Any] = {
        "filename": filename,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_missing": total_missing,
        "n_duplicates": duplicate_count,
        "memory_usage_bytes": int(df.memory_usage(index=True, deep=True).sum()),
        "columns": [],
        "preview": [],
        "numeric_summary": None,
        "target_suggestions": [],
        "warnings": [],
        "feature_types": {"numeric": 0, "categorical": 0, "datetime": 0},
        "correlations": [],
        "outlier_count": 0,
        "quality_score": 0,
    }

    for col in df.columns:
        series = df[col]
        info: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "n_missing": int(missing[col]),
            "missing_pct": round(float(missing[col] / max(n_rows, 1) * 100), 2),
            "n_unique": int(series.nunique()),
        }

        try:
            info["sample_values"] = _serialise(series.dropna().head(5).tolist())
        except Exception:
            info["sample_values"] = []

        if pd.api.types.is_datetime64_any_dtype(series):
            info["semantic_type"] = "datetime"
            profile["feature_types"]["datetime"] += 1
            try:
                info["date_range"] = {
                    "min": str(series.min()),
                    "max": str(series.max()),
                }
            except Exception:
                info["date_range"] = None
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            info["semantic_type"] = "numeric"
            profile["feature_types"]["numeric"] += 1
            if not series.isna().all():
                numeric_series = pd.to_numeric(series, errors="coerce")
                finite_series = numeric_series[np.isfinite(numeric_series)].dropna()
                if finite_series.empty:
                    info["stats"] = None
                    info["histogram"] = {"bin_edges": [], "counts": []}
                    profile["columns"].append(info)
                    continue
                q25 = float(finite_series.quantile(0.25))
                q75 = float(finite_series.quantile(0.75))
                iqr = (q75 - q25) if q25 is not None and q75 is not None else 0
                outlier_count = (
                    int(
                        (
                            (finite_series < q25 - 1.5 * iqr)
                            | (finite_series > q75 + 1.5 * iqr)
                        ).sum()
                    )
                    if iqr
                    else 0
                )
                info["stats"] = _serialise({
                    "mean": round(float(finite_series.mean()), 4),
                    "std": round(float(finite_series.std()), 4),
                    "min": round(float(finite_series.min()), 4),
                    "max": round(float(finite_series.max()), 4),
                    "median": round(float(finite_series.median()), 4),
                    "q25": round(q25, 4),
                    "q75": round(q75, 4),
                    "skewness": round(float(finite_series.skew()), 4),
                    "kurtosis": round(float(finite_series.kurtosis()), 4),
                    "outlier_count": outlier_count,
                })
                info["histogram"] = _histogram(finite_series)
            else:
                info["stats"] = None
                info["histogram"] = {"bin_edges": [], "counts": []}
        else:
            info["semantic_type"] = "categorical"
            profile["feature_types"]["categorical"] += 1
            try:
                top = series.value_counts().head(10)
                info["top_values"] = _serialise(dict(zip(top.index.astype(str), top.values)))
            except Exception:
                info["top_values"] = {}

        profile["columns"].append(info)

    try:
        preview_df = df.head(10).copy()
        for col in preview_df.columns:
            preview_df[col] = preview_df[col].apply(lambda value: "" if pd.isna(value) else value)
        profile["preview"] = _serialise(preview_df.to_dict(orient="records"))
    except Exception:
        profile["preview"] = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        numeric_df = df[numeric_cols]
        profile["numeric_summary"] = _serialise(numeric_df.describe().round(4).to_dict())
        profile["correlations"] = _top_correlations(numeric_df)
        profile["outlier_count"] = _count_outliers(numeric_df)

    profile["target_suggestions"] = _suggest_targets(df)
    profile["warnings"] = _generate_warnings(df)

    completeness = 1.0 - total_missing / total_cells
    uniqueness = 1.0 - duplicate_count / max(n_rows, 1)
    constant_count = sum(1 for col in df.columns if df[col].nunique(dropna=False) <= 1)
    validity = 1.0 - constant_count / max(n_cols, 1)
    combined_quality = completeness * 0.60 + uniqueness * 0.25 + validity * 0.15
    warning_penalty = min(12.0, len(profile["warnings"]) * 1.25)
    profile["quality_score"] = int(round(max(
        0.0,
        min(100.0, combined_quality * 100 - warning_penalty),
    )))

    return profile


def _histogram(series: pd.Series, max_bins: int = 20) -> Dict[str, List[Any]]:
    """Return true dataset-derived histogram bins for a numeric series."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"bin_edges": [], "counts": []}
    unique_count = len(np.unique(values))
    bin_count = min(max_bins, max(1, unique_count))
    try:
        counts, edges = np.histogram(values, bins=bin_count)
        return _serialise({
            "bin_edges": [round(float(edge), 6) for edge in edges],
            "counts": [int(count) for count in counts],
        })
    except Exception:
        return {"bin_edges": [], "counts": []}


def _top_correlations(numeric_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return the strongest non-duplicate Pearson correlation pairs."""
    if numeric_df.shape[1] < 2:
        return []
    try:
        matrix = numeric_df.corr(method="pearson")
        pairs: List[Dict[str, Any]] = []
        columns = matrix.columns.tolist()
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1:]:
                value = matrix.loc[left, right]
                if pd.isna(value):
                    continue
                pairs.append({
                    "feature_a": str(left),
                    "feature_b": str(right),
                    "value": round(float(value), 4),
                })
        pairs.sort(key=lambda item: abs(item["value"]), reverse=True)
        return pairs[:10]
    except Exception:
        return []


def _count_outliers(numeric_df: pd.DataFrame) -> int:
    """Count numeric cells beyond 1.5 IQR, ignoring binary columns."""
    count = 0
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty or series.nunique() <= 2:
            continue
        try:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            count += int(mask.sum())
        except Exception:
            continue
    return count


def _suggest_targets(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Rank likely prediction targets with transparent heuristics."""
    suggestions: List[Dict[str, Any]] = []
    n_rows = len(df)
    target_words = {
        "target", "label", "outcome", "churn", "fraud", "default", "price",
        "revenue", "sales", "score", "class", "status", "response",
    }

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique()
        missing_pct = series.isnull().sum() / max(n_rows, 1) * 100
        if missing_pct > 30 or n_unique < 2:
            continue

        normalised_name = str(col).lower().replace("-", "_").replace(" ", "_")
        name_tokens = set(normalised_name.split("_"))
        name_bonus = 0.12 if target_words.intersection(name_tokens) else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

        if is_numeric:
            if n_unique == 2:
                suggestions.append({
                    "column": str(col),
                    "task_type": "classification",
                    "reason": "Binary numeric outcome with two classes",
                    "score": min(0.99, 0.87 + name_bonus),
                })
            elif 2 < n_unique <= 20 and n_unique / max(n_rows, 1) < 0.05:
                suggestions.append({
                    "column": str(col),
                    "task_type": "classification",
                    "reason": f"Low-cardinality numeric outcome ({n_unique} classes)",
                    "score": min(0.95, 0.70 + name_bonus),
                })
            elif n_unique > 20:
                suggestions.append({
                    "column": str(col),
                    "task_type": "regression",
                    "reason": "Continuous numeric outcome",
                    "score": min(0.92, 0.58 + name_bonus),
                })
        elif 2 <= n_unique <= 30:
            base_score = 0.82 if n_unique == 2 else 0.72
            suggestions.append({
                "column": str(col),
                "task_type": "classification",
                "reason": f"Categorical outcome with {n_unique} classes",
                "score": min(0.98, base_score + name_bonus),
            })

    suggestions.sort(key=lambda suggestion: suggestion["score"], reverse=True)
    return _serialise(suggestions[:6])


def _generate_warnings(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generate actionable data-quality and leakage warnings."""
    warnings: List[Dict[str, str]] = []
    n_rows = len(df)

    for col in df.columns:
        missing_pct = df[col].isnull().sum() / max(n_rows, 1) * 100
        if missing_pct > 50:
            warnings.append({
                "type": "high_missing",
                "severity": "critical",
                "message": f'Column "{col}" has {missing_pct:.1f}% missing values; consider dropping it.',
            })
        elif missing_pct > 2:
            warnings.append({
                "type": "missing",
                "severity": "warning",
                "message": f'Column "{col}" has {missing_pct:.1f}% missing values.',
            })

    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        unique_count = df[col].nunique()
        if unique_count > 50:
            warnings.append({
                "type": "high_cardinality",
                "severity": "warning",
                "message": f'Column "{col}" has {unique_count} unique values; high cardinality may hurt model quality.',
            })

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warnings.append({
            "type": "duplicates",
            "severity": "info",
            "message": f"Found {duplicate_count} duplicate rows ({duplicate_count / max(n_rows, 1) * 100:.1f}% of data).",
        })

    for col in df.columns:
        if df[col].nunique() <= 1:
            warnings.append({
                "type": "constant",
                "severity": "warning",
                "message": f'Column "{col}" is constant and provides no predictive value.',
            })

        normalised_name = str(col).lower().replace("-", "_").replace(" ", "_")
        looks_like_identifier = (
            normalised_name in {"id", "uuid", "identifier", "index"}
            or normalised_name.endswith("_id")
            or normalised_name.startswith("id_")
        )
        unique_ratio = df[col].nunique(dropna=True) / max(n_rows, 1)
        if looks_like_identifier and unique_ratio > 0.8:
            warnings.append({
                "type": "identifier",
                "severity": "warning",
                "message": f'Column "{col}" looks like an identifier and may not generalise.',
            })

    if n_rows < 50:
        warnings.append({
            "type": "small_dataset",
            "severity": "warning",
            "message": f"Dataset has only {n_rows} rows. ML results may be unreliable.",
        })

    return warnings


def check_class_imbalance(df: pd.DataFrame, target_col: str) -> Dict[str, Any] | None:
    """Return class distribution details and an imbalance assessment."""
    if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20:
        return None
    distribution = df[target_col].value_counts(normalize=True)
    if distribution.empty:
        return None
    result = {
        "imbalanced": bool(distribution.min() < 0.10),
        "distribution": dict(zip(distribution.index.astype(str), distribution.values.round(4))),
    }
    if result["imbalanced"]:
        result["message"] = f"Class imbalance detected; the minority class has {distribution.min() * 100:.1f}% of samples."
    else:
        result["message"] = "No severe class imbalance detected."
    return _serialise(result)
