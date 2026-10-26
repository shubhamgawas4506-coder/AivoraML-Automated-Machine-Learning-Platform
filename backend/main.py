"""AivoraML — AI-native data science workspace.

Start with:
    python -m uvicorn backend.main:app --port 8000
"""

import io
import hashlib
import json
import logging
import os
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

try:
    from .schemas import AssistantChatRequest, PredictionRequest, TrainingJobRequest
    from .services.dataset_profiler import profile_dataset, check_class_imbalance
    from .services.ml_engine_enhanced import WorkflowCancelled, run_enhanced_workflow
    from .services.eda_generator import generate_comprehensive_eda
    from .services.ai_advisor import get_ai_advisor
    from .services.ai_assistant import (
        AssistantUnavailable,
        assistant_status,
        get_project_assistant,
    )
    from .services.report_generator import (
        ReportGenerationError,
        ReportInputError,
        generate_report_bundle,
    )
    from .services.training_jobs import (
        ActiveTrainingJobError,
        TrainingJobCancelled,
        TrainingJobError,
        TrainingJobManager,
        TrainingJobNotFoundError,
        TrainingJobQueueFullError,
    )
    from .services.model_registry import (
        DuplicateModelVersionError,
        InvalidModelIdentifierError,
        ModelArtifactError,
        ModelRegistry,
        ModelVersionNotFoundError,
    )
    from .services.project_store import (
        InvalidProjectIdError,
        ProjectAlreadyExistsError,
        ProjectCapacityError,
        ProjectNotFoundError,
        ProjectRepository,
        ProjectStoreError,
    )
except (ImportError, ValueError):
    from schemas import AssistantChatRequest, PredictionRequest, TrainingJobRequest
    from services.dataset_profiler import profile_dataset, check_class_imbalance
    from services.ml_engine_enhanced import WorkflowCancelled, run_enhanced_workflow
    from services.eda_generator import generate_comprehensive_eda
    from services.ai_advisor import get_ai_advisor
    from services.ai_assistant import (
        AssistantUnavailable,
        assistant_status,
        get_project_assistant,
    )
    from services.report_generator import (
        ReportGenerationError,
        ReportInputError,
        generate_report_bundle,
    )
    from services.training_jobs import (
        ActiveTrainingJobError,
        TrainingJobCancelled,
        TrainingJobError,
        TrainingJobManager,
        TrainingJobNotFoundError,
        TrainingJobQueueFullError,
    )
    from services.model_registry import (
        DuplicateModelVersionError,
        InvalidModelIdentifierError,
        ModelArtifactError,
        ModelRegistry,
        ModelVersionNotFoundError,
    )
    from services.project_store import (
        InvalidProjectIdError,
        ProjectAlreadyExistsError,
        ProjectCapacityError,
        ProjectNotFoundError,
        ProjectRepository,
        ProjectStoreError,
    )

# Load environment variables
CURRENT_FILE = Path(__file__).resolve()
if (CURRENT_FILE.parent / "frontend").is_dir():
    PROJECT_ROOT = CURRENT_FILE.parent
elif (CURRENT_FILE.parent.parent / "frontend").is_dir():
    PROJECT_ROOT = CURRENT_FILE.parent.parent
else:
    PROJECT_ROOT = Path(os.getcwd())

load_dotenv(PROJECT_ROOT / ".env")
artifact_logger = logging.getLogger("aivoraml.artifacts")


# Helper function to make data JSON-safe
def make_json_safe(obj, _depth=0):
    """Convert any object to JSON-safe format recursively."""
    import json
    
    # Prevent infinite recursion
    if _depth > 50:
        return "<max_depth_reached>"
    
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None

    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Functions, methods, and built-ins
    if callable(obj):
        return None
    
    # NumPy types
    if hasattr(obj, 'dtype'):  # NumPy array or scalar
        try:
            return make_json_safe(obj.tolist(), _depth + 1)
        except:
            return str(obj)
    
    if type(obj).__module__ == 'numpy':
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
    
    # Dictionaries
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # Skip private keys and callable values
            if not str(k).startswith('_') and not callable(v):
                result[str(k)] = make_json_safe(v, _depth + 1)
        return result
    
    # Lists and tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item, _depth + 1) for item in obj if not callable(item)]
    
    # Try direct JSON encoding
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    
    # Objects with __dict__
    if hasattr(obj, '__dict__'):
        return make_json_safe(obj.__dict__, _depth + 1)
    
    # Last resort
    return str(obj)

# ── In-memory session store (single-user local MVP) ─────────────────────────
APP_DIR = Path(__file__).resolve().parent


def _aivoraml_setting(name: str, default: str = "") -> str:
    """Read the current setting while accepting the legacy prefix during migration."""
    return os.getenv(
        f"AIVORAML_{name}",
        os.getenv(f"REDIMENT_{name}", default),
    )


is_vercel = bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))
default_data_dir = "/tmp/storage" if is_vercel else str(PROJECT_ROOT / "storage")
DATA_DIR = Path(
    _aivoraml_setting("DATA_DIR", default_data_dir)
).resolve()
WORKSPACE_FILE = DATA_DIR / "workspace.joblib"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"
MAX_UPLOAD_BYTES = int(_aivoraml_setting("MAX_UPLOAD_MB", "100")) * 1024 * 1024
MAX_DATASET_ROWS = int(_aivoraml_setting("MAX_ROWS", "1000000"))
MAX_DATASET_COLUMNS = int(_aivoraml_setting("MAX_COLUMNS", "500"))
MAX_DATASET_CELLS = int(_aivoraml_setting("MAX_CELLS", "25000000"))
MAX_BATCH_BYTES = int(_aivoraml_setting("MAX_BATCH_MB", "25")) * 1024 * 1024
MAX_BATCH_ROWS = int(_aivoraml_setting("MAX_BATCH_ROWS", "10000"))
MAX_BATCH_CELLS = int(_aivoraml_setting("MAX_BATCH_CELLS", "2000000"))


class _Store:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.filename: str = ""
        self.profile: Optional[dict] = None
        self.workflow_result: Optional[dict] = None
        self.project_id: str = ""
        self.created_at: str = ""
        self.updated_at: str = ""
        self.predictions: list[dict] = []
        self.report_bundle: Optional[dict] = None
        self.persistence_error: Optional[str] = None

    def begin_project(self, filename: str):
        now = datetime.now(timezone.utc).isoformat()
        self.df = None
        self.filename = filename
        self.project_id = uuid.uuid4().hex[:12]
        self.created_at = now
        self.updated_at = now
        self.profile = None
        self.workflow_result = None
        self.predictions = []
        self.report_bundle = None
        self.persistence_error = None

store = _Store()
STORE_LOCK = threading.RLock()
PERSIST_LOCK = threading.Lock()
REPORT_GENERATION_LOCK = threading.Lock()
training_jobs = TrainingJobManager(
    max_workers=max(1, min(2, int(_aivoraml_setting("TRAINING_WORKERS", "1")))),
    max_history=100,
    max_outstanding_jobs=16,
)
model_registry = ModelRegistry(
    DATA_DIR / "model_versions",
    retention_limit=int(_aivoraml_setting("MODEL_VERSION_RETENTION", "20")),
)
project_repository = ProjectRepository(
    DATA_DIR / "projects",
    max_projects=int(_aivoraml_setting("MAX_PROJECTS", "100")),
    max_snapshot_bytes=int(
        _aivoraml_setting("MAX_PROJECT_SNAPSHOT_MB", "512")
    ) * 1024 * 1024,
)


def _persistence_enabled() -> bool:
    return _aivoraml_setting("DISABLE_PERSISTENCE").lower() not in {
        "1",
        "true",
        "yes",
    }


def _persist_workspace():
    """Atomically persist the active workspace in its isolated project slot."""
    if not _persistence_enabled():
        return
    project_id = ""
    with PERSIST_LOCK:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with STORE_LOCK:
                if store.df is None:
                    return
                store.updated_at = datetime.now(timezone.utc).isoformat()
                project_id = store.project_id
                filename = store.filename
                snapshot = {
                    "version": 1,
                    "project_id": project_id,
                    "filename": filename,
                    "created_at": store.created_at,
                    "updated_at": store.updated_at,
                    "df": store.df,
                    "profile": store.profile,
                    "workflow_result": store.workflow_result,
                    "predictions": list(store.predictions),
                    "report_bundle": store.report_bundle,
                }
            try:
                metadata = project_repository.save_project(
                    project_id,
                    snapshot,
                    filename=filename,
                    activate=False,
                )
            except ProjectNotFoundError:
                metadata = project_repository.create_project(
                    snapshot,
                    project_id=project_id,
                    filename=filename,
                    activate=False,
                )
            with STORE_LOCK:
                if store.project_id == project_id:
                    project_repository.activate_project(project_id)
                    store.created_at = str(
                        metadata.get("created_at") or store.created_at
                    )
                    store.updated_at = str(
                        metadata.get("updated_at") or store.updated_at
                    )
                    store.persistence_error = None
        except Exception as exc:
            with STORE_LOCK:
                if not project_id or store.project_id == project_id:
                    store.persistence_error = type(exc).__name__
            artifact_logger.warning("Workspace persistence failed: %s", exc)


def _persist_prediction_history():
    """Persist small prediction history without rewriting the dataset and model."""
    if not _persistence_enabled() or store.df is None:
        return
    with PERSIST_LOCK:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with STORE_LOCK:
                payload = {
                    "project_id": store.project_id,
                    "predictions": make_json_safe(store.predictions),
                }
            project_id = project_repository.validate_project_id(
                str(payload["project_id"])
            )
            project_dir = (project_repository.root / project_id).resolve()
            if project_dir.parent != project_repository.root or not project_dir.is_dir():
                raise RuntimeError("Active project storage is unavailable.")
            predictions_file = project_dir / "predictions.json"
            temp_file = project_dir / ".predictions.json.tmp"
            temp_file.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_file.replace(predictions_file)
            with STORE_LOCK:
                if store.project_id == project_id:
                    store.updated_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            artifact_logger.warning("Prediction persistence failed: %s", exc)


def _hydrate_store_from_snapshot(snapshot: dict) -> None:
    """Replace the in-memory active store from one verified repository snapshot."""
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
        raise ValueError("Unsupported project snapshot.")
    restored_df = snapshot.get("df")
    if not isinstance(restored_df, pd.DataFrame):
        raise ValueError("Project snapshot has no valid dataset.")
    with STORE_LOCK:
        store.df = restored_df
        store.project_id = str(snapshot.get("project_id") or "")
        store.filename = str(snapshot.get("filename") or "dataset.csv")
        store.created_at = str(snapshot.get("created_at") or "")
        store.updated_at = str(snapshot.get("updated_at") or "")
        store.profile = snapshot.get("profile")
        store.workflow_result = snapshot.get("workflow_result")
        store.predictions = list(snapshot.get("predictions") or [])[-5000:]
        store.report_bundle = snapshot.get("report_bundle")
        store.persistence_error = None

    snapshot_project_id = str(snapshot.get("project_id") or "")
    predictions_file = (
        project_repository.root / snapshot_project_id / "predictions.json"
    )
    if (
        predictions_file.is_file()
        and predictions_file.parent.parent == project_repository.root
    ):
        try:
            prediction_data = json.loads(
                predictions_file.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            artifact_logger.warning(
                "Ignoring invalid prediction sidecar project_id=%s error=%s",
                snapshot_project_id,
                type(exc).__name__,
            )
        else:
            if (
                isinstance(prediction_data, dict)
                and prediction_data.get("project_id") == snapshot_project_id
                and isinstance(prediction_data.get("predictions"), list)
            ):
                with STORE_LOCK:
                    if store.project_id == snapshot_project_id:
                        store.predictions = prediction_data["predictions"][-5000:]


def _clear_active_store() -> None:
    with STORE_LOCK:
        store.df = None
        store.filename = ""
        store.profile = None
        store.workflow_result = None
        store.project_id = ""
        store.created_at = ""
        store.updated_at = ""
        store.predictions = []
        store.report_bundle = None
        store.persistence_error = None


def _restore_workspace():
    """Restore the checksum-verified active project, importing legacy state once."""
    if not _persistence_enabled():
        return
    try:
        if project_repository.active_project_id:
            snapshot = project_repository.load_project()
            _hydrate_store_from_snapshot(snapshot)
            return
        if not project_repository.list_projects() and WORKSPACE_FILE.is_file():
            metadata = project_repository.import_legacy_snapshot(WORKSPACE_FILE)
            snapshot = project_repository.load_project(metadata["id"])
            _hydrate_store_from_snapshot(snapshot)
            if PREDICTIONS_FILE.is_file():
                legacy_predictions = json.loads(
                    PREDICTIONS_FILE.read_text(encoding="utf-8")
                )
                if (
                    isinstance(legacy_predictions, dict)
                    and legacy_predictions.get("project_id") == store.project_id
                    and isinstance(legacy_predictions.get("predictions"), list)
                ):
                    store.predictions = legacy_predictions["predictions"][-5000:]
                    _persist_prediction_history()
    except Exception as exc:
        _clear_active_store()
        artifact_logger.warning("Workspace restore failed: %s", type(exc).__name__)


_restore_workspace()


def _project_metadata() -> dict:
    """Return real workspace metadata without exposing the dataset itself."""
    if store.df is None:
        raise HTTPException(404, "No project is open.")
    return {
        "id": store.project_id,
        "filename": store.filename,
        "created_at": store.created_at,
        "updated_at": store.updated_at,
        "rows": int(len(store.df)),
        "columns": int(len(store.df.columns)),
        "trained": bool(
            store.workflow_result
            and store.workflow_result.get("_pipeline") is not None
        ),
        "prediction_count": len(store.predictions),
        "report_ready": bool(
            store.report_bundle
            and store.report_bundle.get("status") == "completed"
        ),
        "persistence": {
            "enabled": _persistence_enabled(),
            "status": (
                "disabled"
                if not _persistence_enabled()
                else "error"
                if store.persistence_error
                else "saved"
            ),
            "error_type": store.persistence_error,
        },
    }


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable content fingerprint without retaining source bytes."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    schema = "|".join(f"{column}:{df[column].dtype}" for column in df.columns)
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8", errors="replace"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _feature_spec() -> tuple[list[str], list[str]]:
    """Get the exact feature contract used by the fitted pipeline."""
    if not store.workflow_result:
        return [], []
    info = (
        store.workflow_result.get("features")
        or store.workflow_result.get("feature_info")
        or {}
    )
    return list(info.get("numeric") or []), list(info.get("categorical") or [])


def _require_pipeline():
    if not store.workflow_result:
        raise HTTPException(404, "No trained model available. Train a model first.")
    pipeline = store.workflow_result.get("_pipeline")
    if pipeline is None:
        raise HTTPException(404, "No fitted pipeline available. Train a model first.")
    return pipeline


def _positive_class_contract(workflow_result: dict, pipeline) -> tuple[Any, int, Any]:
    """Resolve the saved event label to its fitted probability column."""
    classes = list(getattr(pipeline, "classes_", []))
    if len(classes) != 2:
        raise ValueError("A binary fitted classifier is required.")
    threshold_selection = workflow_result.get("threshold_selection") or {}
    positive_label = workflow_result.get("positive_label")
    if positive_label is None:
        positive_label = threshold_selection.get("positive_label")
    if positive_label is None:
        # Backward compatibility for models saved before positive labels were
        # explicit. New training results always persist the chosen event label.
        positive_label = classes[-1]
    positive_index = next(
        (
            index
            for index, label in enumerate(classes)
            if bool(label == positive_label)
        ),
        None,
    )
    if positive_index is None:
        raise ValueError("The saved positive label does not match the fitted classifier.")
    negative_label = classes[1 - positive_index]
    return positive_label, positive_index, negative_label


def _extract_fitted_preprocessor(pipeline, workflow_result: Optional[dict] = None):
    """Find preprocessing inside ordinary or probability-calibrated pipelines."""
    workflow_result = workflow_result or {}
    stored = workflow_result.get("_preprocessor")
    if stored is not None:
        return stored

    candidates = [pipeline, getattr(pipeline, "estimator", None)]
    candidates.extend(
        getattr(calibrated, "estimator", None)
        for calibrated in getattr(pipeline, "calibrated_classifiers_", [])
    )
    for candidate in candidates:
        if candidate is None:
            continue
        fitted = getattr(candidate, "named_steps", {}).get("preprocessor")
        if fitted is not None:
            return fitted
    return None


def _compact_workflow_for_registry(workflow_result: dict) -> dict:
    """Keep activation-ready evidence while removing redundant bulky audit data."""
    compact = {
        key: value
        for key, value in workflow_result.items()
        if not key.startswith("_")
    }
    model_fields = {
        "name", "metrics", "training_score", "holdout_score",
        "cv_score_mean", "cv_score_std", "cv_score_min", "cv_score_max",
        "selection_score", "overfitting_gap", "fit_assessment",
        "metric_source", "selection_metric", "balancing",
        "training_time_seconds", "validation_time_seconds",
        "inference_time_ms", "is_best",
    }
    compact["models_trained"] = [
        {key: value for key, value in model.items() if key in model_fields}
        for model in workflow_result.get("models_trained", [])
        if isinstance(model, dict)
    ]

    threshold = dict(workflow_result.get("threshold_selection") or {})
    threshold.pop("curve", None)
    compact["threshold_selection"] = threshold or None

    evaluation = dict(workflow_result.get("evaluation_details") or {})
    # These matrices are reproducible from the artifact and report bundle; the
    # UI and activation flow use the compact metrics/diagnostics below.
    evaluation.pop("classification_report", None)
    evaluation.pop("confusion_matrix", None)
    compact["evaluation_details"] = evaluation

    run_manifest = dict(workflow_result.get("run_manifest") or {})
    run_manifest.pop("candidate_models", None)
    manifest_threshold = dict(run_manifest.get("threshold_selection") or {})
    manifest_threshold.pop("curve", None)
    if manifest_threshold:
        run_manifest["threshold_selection"] = manifest_threshold
    compact["run_manifest"] = run_manifest
    return compact


def _register_model_version(
    *,
    project_id: str,
    pipeline,
    workflow_result: dict,
    approach: str,
) -> dict:
    """Register an immutable champion and attach its project-scoped version."""
    existing = model_registry.list_versions(project_id)
    model_number = max(
        [int((item.get("metadata") or {}).get("model_number") or 0) for item in existing]
        or [0]
    ) + 1
    run_manifest = workflow_result.get("run_manifest") or {}
    run_id = str(run_manifest.get("run_id") or uuid.uuid4().hex)
    version_id = f"v{model_number:04d}-{run_id[:8].lower()}"
    public_workflow = _compact_workflow_for_registry(workflow_result)
    entry = model_registry.register(
        project_id,
        version_id,
        pipeline,
        {
            "model_number": model_number,
            "run_id": run_id,
            "best_model": workflow_result.get("best_model"),
            "best_score": workflow_result.get("best_score"),
            "target_column": workflow_result.get("target_column"),
            "task_type": workflow_result.get("task_type"),
            "strategy": approach,
            "dataset_fingerprint": run_manifest.get("dataset_fingerprint"),
            "trained_at_utc": run_manifest.get("trained_at_utc"),
            "workflow": make_json_safe(public_workflow),
        },
    )
    workflow_result["model_version"] = version_id
    run_manifest["model_version"] = version_id
    workflow_result["run_manifest"] = run_manifest
    return entry


def _csv_response(frame: pd.DataFrame, filename: str) -> Response:
    safe_frame = frame.copy()
    text_columns = safe_frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in text_columns:
        safe_frame[column] = safe_frame[column].map(
            lambda value: (
                "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                else value
            )
        )
    return Response(
        content=safe_frame.to_csv(index=False),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _read_tabular_bytes(raw: bytes, filename: str) -> pd.DataFrame:
    """Parse a supported tabular payload from a bounded in-memory buffer."""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in {"csv", "xlsx", "xls", "json", "parquet"}:
        raise HTTPException(
            400,
            f"Unsupported format: .{extension or '(none)'}. Use CSV, Excel, JSON, or Parquet.",
        )
    buffer = io.BytesIO(raw)
    try:
        if extension == "csv":
            return pd.read_csv(buffer)
        if extension in {"xlsx", "xls"}:
            return pd.read_excel(buffer)
        if extension == "json":
            return pd.read_json(buffer)
        return pd.read_parquet(buffer)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Failed to parse {extension.upper()} file: {exc}")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="AivoraML",
    description=(
        "Local-first machine-learning workspace for dataset profiling, "
        "workflow orchestration, prediction, and verified artifact delivery."
    ),
    version="0.2.0",
    docs_url=None,
    redoc_url=None,
)

trusted_hosts = ["*"] if is_vercel or _aivoraml_setting("ALLOW_NONLOCAL_HOSTS").lower() in {"1", "true", "yes"} else ["127.0.0.1", "localhost", "testserver", "[::1]"]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)


@app.middleware("http")
async def add_request_context_and_security_headers(request: Request, call_next):
    """Attach a safe request identifier, timing, and browser hardening headers."""
    request_id = request.headers.get("x-request-id", "").strip()
    if not request_id or len(request_id) > 100:
        request_id = uuid.uuid4().hex
    origin = request.headers.get("origin", "").strip().lower()
    origin_host = urlsplit(origin).hostname if origin else None
    if (
        origin
        and not is_vercel
        and origin_host not in {"127.0.0.1", "localhost", "::1", "testserver"}
        and _aivoraml_setting("ALLOW_NONLOCAL_ORIGINS").lower()
        not in {"1", "true", "yes"}
    ):
        return Response(
            content=json.dumps({"detail": "Cross-origin access is disabled in local mode."}),
            status_code=403,
            media_type="application/json",
            headers={"X-Request-ID": request_id},
        )
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = (
        f"{(time.perf_counter() - started) * 1000:.2f}"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; font-src 'self' data:; "
        "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; object-src 'none'; base-uri 'self'; "
        "frame-ancestors 'none'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

STATIC_DIR = PROJECT_ROOT / "frontend"
if not STATIC_DIR.is_dir():
    for candidate in [CURRENT_FILE.parent.parent / "frontend", CURRENT_FILE.parent / "frontend", Path(os.getcwd()) / "frontend"]:
        if candidate.is_dir():
            STATIC_DIR = candidate
            break


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["system"])
@app.get("/api/v1/health", tags=["system"])
async def health_check():
    """Lightweight local readiness signal without exposing dataset contents."""
    active_job = (
        training_jobs.get_active_job(store.project_id)
        if store.project_id
        else None
    )
    return {
        "status": "ok",
        "service": "AivoraML",
        "version": app.version,
        "mode": "local",
        "persistence": _persistence_enabled(),
        "project_open": store.df is not None,
        "training_job": active_job,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/capabilities", tags=["system"])
async def capabilities():
    """Return backend-owned product limits so the UI never has to guess."""
    remote_assistant = assistant_status()
    return {
        "upload": {
            "formats": ["csv", "xlsx", "xls", "json", "parquet"],
            "max_bytes": MAX_UPLOAD_BYTES,
            "max_rows": MAX_DATASET_ROWS,
            "max_columns": MAX_DATASET_COLUMNS,
            "max_cells": MAX_DATASET_CELLS,
        },
        "batch_prediction": {
            "enabled": True,
            "max_bytes": MAX_BATCH_BYTES,
            "max_rows": MAX_BATCH_ROWS,
            "max_cells": MAX_BATCH_CELLS,
        },
        "training": {
            "background_jobs": True,
            "cancellation": "cooperative",
            "strategies": ["fast", "balanced", "high_performance"],
        },
        "remote_ai": {
            "enabled": _aivoraml_setting("ENABLE_REMOTE_AI").lower()
            in {"1", "true", "yes"},
            "data_scope": "project metadata, schema summaries, and model metrics only",
            "assistant": remote_assistant,
        },
        "prediction_history": {
            "retains_inputs": _aivoraml_setting(
                "RETAIN_PREDICTION_INPUTS",
                "false",
            ).lower() in {"1", "true", "yes"},
            "max_records": 5000,
            "clearable": True,
        },
        "mode": "local",
    }


@app.get("/", response_class=HTMLResponse)
async def landing():
    """Serve the Premium Landing Page."""
    landing_page = STATIC_DIR / "landing.html"
    if landing_page.exists():
        return HTMLResponse(content=landing_page.read_text(encoding="utf-8"))
    # Fallback to ML app if landing page doesn't exist
    index = STATIC_DIR / "index_premium.html"
    if not index.exists():
        index = STATIC_DIR / "index_ultimate.html"
    if not index.exists():
        index = STATIC_DIR / "index_enhanced.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))

@app.get("/app", response_class=HTMLResponse)
async def app_ui():
    """Serve the ML Application UI (Premium Edition)."""
    index = STATIC_DIR / "index_premium.html"
    if not index.exists():
        index = STATIC_DIR / "index_ultimate.html"
    if not index.exists():
        index = STATIC_DIR / "index_enhanced.html"
    return HTMLResponse(content=index.read_text(encoding="utf-8"))


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
async def api_documentation():
    """Serve the first-party AivoraML API reference and analytics interface."""
    documentation = STATIC_DIR / "api_docs.html"
    if not documentation.is_file():
        raise HTTPException(
            status_code=404,
            detail="The API documentation interface is unavailable.",
        )
    return HTMLResponse(content=documentation.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file (CSV, Excel, JSON, Parquet)."""
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    safe_filename = Path(file.filename.replace("\\", "/")).name.strip()
    if not safe_filename or "\x00" in safe_filename:
        raise HTTPException(400, "Invalid filename.")

    ext = safe_filename.rsplit(".", 1)[-1].lower() if "." in safe_filename else ""
    if ext not in {"csv", "xlsx", "xls", "json", "parquet"}:
        raise HTTPException(400, f"Unsupported format: .{ext or '(none)'}")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(413, f"File exceeds the {max_mb} MB upload limit.")
    buf = io.BytesIO(raw)

    try:
        if ext == "csv":
            df = pd.read_csv(buf)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(buf)
        elif ext == "json":
            df = pd.read_json(buf)
        elif ext == "parquet":
            df = pd.read_parquet(buf)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Failed to parse file: {exc}")

    if df.empty:
        raise HTTPException(400, "Uploaded file produced an empty DataFrame.")
    if len(df) > MAX_DATASET_ROWS:
        raise HTTPException(
            413,
            f"Dataset has {len(df):,} rows; the limit is {MAX_DATASET_ROWS:,}.",
        )
    if len(df.columns) > MAX_DATASET_COLUMNS:
        raise HTTPException(
            413,
            f"Dataset has {len(df.columns):,} columns; the limit is {MAX_DATASET_COLUMNS:,}.",
        )
    dataset_cells = int(len(df) * len(df.columns))
    if dataset_cells > MAX_DATASET_CELLS:
        raise HTTPException(
            413,
            f"Dataset has {dataset_cells:,} cells; the limit is {MAX_DATASET_CELLS:,}.",
        )
    if len(df) < 2:
        raise HTTPException(400, "Dataset must contain at least two rows.")
    if any(str(column).strip() == "" for column in df.columns):
        raise HTTPException(400, "Every column must have a non-empty name.")
    if df.columns.duplicated().any():
        duplicates = [str(value) for value in df.columns[df.columns.duplicated()].tolist()]
        raise HTTPException(400, f"Duplicate column names are not supported: {duplicates}")

    profile = await run_in_threadpool(profile_dataset, df, safe_filename)
    profile["dataset_fingerprint"] = await run_in_threadpool(
        _dataset_fingerprint,
        df,
    )
    
    # Chat enablement must not add a billable network call to every upload.
    # Automatic profile advice therefore has its own explicit opt-in.
    advisor = (
        get_ai_advisor()
        if _aivoraml_setting("ENABLE_REMOTE_AI").lower()
        in {"1", "true", "yes"}
        and _aivoraml_setting("AUTO_PROFILE_AI", "false").lower()
        in {"1", "true", "yes"}
        else None
    )
    if advisor is not None:
        try:
            ai_insights = await run_in_threadpool(
                advisor.get_dataset_insights,
                profile,
            )
            profile["ai_insights"] = ai_insights
        except Exception as exc:
            artifact_logger.warning("Remote dataset advice failed: %s", type(exc).__name__)

    with STORE_LOCK:
        store.begin_project(safe_filename)
        store.df = df
        store.profile = profile
    
    await run_in_threadpool(_persist_workspace)
    await run_in_threadpool(_persist_prediction_history)
    return profile


@app.post("/api/demo")
async def create_demo_dataset():
    """Create a deterministic customer-churn dataset for the guided product demo."""
    rng = np.random.default_rng(42)
    base_rows = 2488

    tenure = rng.integers(0, 73, base_rows)
    monthly_charges = np.round(rng.uniform(18.0, 120.0, base_rows), 2)
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        base_rows,
        p=[0.56, 0.25, 0.19],
    )
    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No"],
        base_rows,
        p=[0.45, 0.36, 0.19],
    )
    tech_support = rng.choice(["No", "Yes"], base_rows, p=[0.58, 0.42]).astype(object)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        base_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    senior_citizen = rng.choice([0, 1], base_rows, p=[0.84, 0.16])
    partner = rng.choice(["No", "Yes"], base_rows, p=[0.52, 0.48])
    dependents = rng.choice(["No", "Yes"], base_rows, p=[0.70, 0.30])
    support_tickets = np.clip(rng.poisson(1.8, base_rows), 0, 9)

    satisfaction = np.clip(
        4.25
        - support_tickets * 0.32
        - (contract_type == "Month-to-month") * 0.25
        + rng.normal(0, 0.62, base_rows),
        1,
        5,
    )
    total_charges = np.round(
        np.maximum(18.0, monthly_charges * np.maximum(tenure, 1) + rng.normal(0, 85, base_rows)),
        2,
    )

    churn_logit = (
        -1.45
        + (contract_type == "Month-to-month") * 1.18
        + (internet_service == "Fiber optic") * 0.42
        + (tech_support == "No") * 0.46
        + (payment_method == "Electronic check") * 0.33
        + senior_citizen * 0.25
        + support_tickets * 0.20
        - tenure * 0.026
        - satisfaction * 0.31
    )
    churn_probability = 1.0 / (1.0 + np.exp(-churn_logit))
    churn = np.where(rng.random(base_rows) < churn_probability, "Yes", "No")

    demo_df = pd.DataFrame({
        "customer_id": [f"RD-{index:05d}" for index in range(10001, 10001 + base_rows)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "satisfaction_score": np.round(satisfaction, 2),
        "support_tickets": support_tickets,
        "churn": churn,
    })

    # Introduce realistic, deterministic quality issues for the cleaning workflow.
    total_missing_rows = rng.choice(base_rows, size=52, replace=False)
    satisfaction_missing_rows = rng.choice(base_rows, size=13, replace=False)
    support_missing_rows = rng.choice(base_rows, size=8, replace=False)
    demo_df.loc[total_missing_rows, "total_charges"] = np.nan
    demo_df.loc[satisfaction_missing_rows, "satisfaction_score"] = np.nan
    demo_df.loc[support_missing_rows, "tech_support"] = None

    duplicate_rows = demo_df.sample(n=12, random_state=42)
    demo_df = pd.concat([demo_df, duplicate_rows], ignore_index=True)

    profile = await run_in_threadpool(
        profile_dataset,
        demo_df,
        "customer_churn.csv",
    )
    profile["dataset_fingerprint"] = await run_in_threadpool(
        _dataset_fingerprint,
        demo_df,
    )
    profile["is_demo"] = True
    with STORE_LOCK:
        store.begin_project("customer_churn.csv")
        store.df = demo_df
        store.profile = profile
    await run_in_threadpool(_persist_workspace)
    await run_in_threadpool(_persist_prediction_history)
    return profile


@app.get("/api/dataset/profile")
async def get_profile():
    """Return the last-computed dataset profile."""
    if store.profile is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    return store.profile


@app.get("/api/projects")
async def list_projects():
    """List every isolated durable local project, newest first."""
    active_id = store.project_id or project_repository.active_project_id
    projects = project_repository.list_projects() if _persistence_enabled() else (
        [_project_metadata()] if store.df is not None else []
    )
    for project in projects:
        project["active"] = project.get("id") == active_id
    return {"active_project_id": active_id or None, "projects": projects}


def _activate_project_snapshot(project_id: str) -> None:
    """Serialize repository activation with workspace persistence."""
    with PERSIST_LOCK:
        snapshot = project_repository.load_project(project_id)
        _hydrate_store_from_snapshot(snapshot)


@app.post("/api/projects/{project_id}/activate")
async def activate_project(project_id: str):
    """Switch the active workspace to one checksum-verified project snapshot."""
    if not _persistence_enabled():
        raise HTTPException(409, "Project switching requires local persistence.")
    active_job = training_jobs.get_active_job(store.project_id) if store.project_id else None
    target_job = training_jobs.get_active_job(project_id)
    active_job = active_job or target_job
    if active_job:
        raise HTTPException(
            409,
            {
                "message": "Cancel or finish the active training job before switching projects.",
                "active_job_id": active_job["job_id"],
            },
        )
    try:
        await run_in_threadpool(_activate_project_snapshot, project_id)
    except (InvalidProjectIdError, ProjectNotFoundError):
        raise HTTPException(404, "Project not found.")
    except ProjectStoreError:
        raise HTTPException(409, "Project storage failed its integrity check.")
    return {
        "project": _project_metadata(),
        "profile": make_json_safe(store.profile),
        "trained": bool(store.workflow_result),
    }


def _delete_flat_project_directory(root: Path, project_id: str) -> dict:
    """Delete one exact, non-recursive project artifact directory."""
    result = {"deleted_files": [], "failures": []}
    target = root / project_id
    if not target.exists():
        return result
    if target.is_symlink():
        result["failures"].append({"path": str(target), "error": "symbolic link refused"})
        return result
    resolved = target.resolve()
    if resolved.parent != root.resolve() or not resolved.is_dir():
        result["failures"].append({"path": str(target), "error": "unsafe directory"})
        return result
    children = list(resolved.iterdir())
    if len(children) > 256 or any(child.is_symlink() or not child.is_file() for child in children):
        result["failures"].append({"path": str(target), "error": "non-flat or oversized directory"})
        return result
    for child in children:
        try:
            child.unlink()
            try:
                display_path = str(child.relative_to(PROJECT_ROOT))
            except ValueError:
                display_path = str(child)
            result["deleted_files"].append(display_path)
        except OSError as exc:
            result["failures"].append({"path": str(child), "error": type(exc).__name__})
    if not result["failures"]:
        try:
            resolved.rmdir()
        except OSError as exc:
            result["failures"].append({"path": str(resolved), "error": type(exc).__name__})
    return result


def _delete_project_repository_state(project_id: str) -> tuple[dict, dict]:
    """Load and delete one repository project without racing persistence writes."""
    with PERSIST_LOCK:
        snapshot = project_repository.load_project(project_id, activate=False)
        result = project_repository.delete_project(project_id)
        return snapshot, result


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete one exact local project plus its project-scoped model/report files."""
    if not _persistence_enabled():
        raise HTTPException(409, "Project deletion requires local persistence.")
    active_job = training_jobs.get_active_job(project_id)
    if active_job:
        raise HTTPException(409, "Cancel or finish this project's training job first.")
    try:
        snapshot, repository_result = await run_in_threadpool(
            _delete_project_repository_state,
            project_id,
        )
    except (InvalidProjectIdError, ProjectNotFoundError):
        raise HTTPException(404, "Project not found.")
    except ProjectStoreError:
        raise HTTPException(409, "Project storage failed its integrity check.")

    if not repository_result.get("deleted"):
        raise HTTPException(409, {"message": "Project deletion was not completed.", **repository_result})

    cleanup = _delete_flat_project_directory(model_registry.root, project_id)
    report_cleanup = {"deleted_files": [], "failures": []}
    allowed_roots = [
        (PROJECT_ROOT / name).resolve()
        for name in ("reports", "models", "notebooks", "outputs")
    ]
    artifact_paths = (snapshot.get("report_bundle") or {}).get("artifacts") or {}
    for value in artifact_paths.values():
        candidate = (PROJECT_ROOT / str(value)).resolve()
        safe_root = next(
            (root for root in allowed_roots if candidate.is_relative_to(root)),
            None,
        )
        if (
            safe_root is None
            or not candidate.is_file()
            or project_id not in candidate.name
        ):
            if candidate.exists():
                report_cleanup["failures"].append({
                    "path": str(candidate),
                    "error": "retained unscoped or unsafe legacy artifact",
                })
            continue
        try:
            candidate.unlink()
            report_cleanup["deleted_files"].append(str(candidate.relative_to(PROJECT_ROOT)))
        except OSError as exc:
            report_cleanup["failures"].append({
                "path": str(candidate),
                "error": type(exc).__name__,
            })

    # Also remove project-prefixed partial outputs from interrupted report jobs.
    for directory in (
        PROJECT_ROOT / "reports",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "notebooks" / "generated",
    ):
        resolved_directory = directory.resolve()
        if not resolved_directory.is_dir():
            continue
        candidates = [
            path
            for path in resolved_directory.iterdir()
            if path.is_file() and path.name.startswith(f"{project_id}_")
        ][:100]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                candidate.unlink()
                relative = str(candidate.relative_to(PROJECT_ROOT))
                if relative not in report_cleanup["deleted_files"]:
                    report_cleanup["deleted_files"].append(relative)
            except OSError as exc:
                report_cleanup["failures"].append({
                    "path": str(candidate),
                    "error": type(exc).__name__,
                })

    if store.project_id == project_id:
        _clear_active_store()
    remaining = project_repository.list_projects()
    if remaining and store.df is None:
        next_id = remaining[0]["id"]
        next_snapshot = await run_in_threadpool(
            project_repository.load_project,
            next_id,
        )
        _hydrate_store_from_snapshot(next_snapshot)
    return {
        **repository_result,
        "model_artifacts": cleanup,
        "report_artifacts": report_cleanup,
        "active_project_id": store.project_id or None,
    }


@app.get("/api/project/current")
async def get_current_project():
    return _project_metadata()


@app.get("/api/workflow/result")
async def get_workflow_result():
    """Return persisted run metadata without serialized estimators."""
    if not store.workflow_result:
        raise HTTPException(404, "No workflow result available.")
    safe_result = {
        key: value
        for key, value in store.workflow_result.items()
        if not key.startswith("_")
    }
    return make_json_safe(safe_result)


@app.get("/api/predict/schema")
async def get_prediction_schema():
    """Return the fitted model's real input contract and useful sample values."""
    _require_pipeline()
    numeric_cols, categorical_cols = _feature_spec()
    if not numeric_cols and not categorical_cols:
        raise HTTPException(409, "The trained model has no recorded feature schema.")

    profile_columns = {
        column.get("name"): column
        for column in (store.profile or {}).get("columns", [])
    }
    fields = []
    for name in numeric_cols:
        column = profile_columns.get(name, {})
        stats = column.get("stats") or {}
        recommended_value = stats.get("median")
        fields.append({
            "name": name,
            "kind": "number",
            "nullable": True,
            "sample_value": recommended_value,
            "recommended_value": recommended_value,
            "auto_strategy": "median",
            "minimum": stats.get("min"),
            "maximum": stats.get("max"),
        })
    for name in categorical_cols:
        column = profile_columns.get(name, {})
        top_values = column.get("top_values") or {}
        recommended_value = next(iter(top_values), None)
        fields.append({
            "name": name,
            "kind": "category",
            "nullable": True,
            "sample_value": recommended_value,
            "recommended_value": recommended_value,
            "auto_strategy": "most_frequent",
            "choices": list(top_values)[:50],
        })
    return {
        "target_column": store.workflow_result.get("target_column"),
        "task_type": store.workflow_result.get("task_type"),
        "model": store.workflow_result.get("best_model"),
        "positive_label": store.workflow_result.get("positive_label"),
        "fields": make_json_safe(fields),
    }


@app.post("/api/predict")
async def predict(payload: PredictionRequest):
    """Run one real inference through the fitted preprocessing/model pipeline."""
    inference_started = time.perf_counter()
    with STORE_LOCK:
        pipeline = _require_pipeline()
        prediction_project_id = store.project_id
        workflow_result = store.workflow_result or {}
        numeric_cols, categorical_cols = _feature_spec()
    expected = numeric_cols + categorical_cols
    if not expected:
        raise HTTPException(409, "The trained model has no recorded feature schema.")
    if not payload.values:
        raise HTTPException(422, "Provide at least one feature value.")

    unknown = sorted(set(payload.values) - set(expected))
    if unknown:
        raise HTTPException(422, f"Unknown feature fields: {', '.join(unknown)}")

    row: dict = {}
    validation_errors = []
    for name in numeric_cols:
        value = payload.values.get(name)
        if value is None or value == "":
            row[name] = np.nan
            continue
        try:
            parsed = float(value)
            if not np.isfinite(parsed):
                raise ValueError
            row[name] = parsed
        except (TypeError, ValueError):
            validation_errors.append(f"{name} must be a finite number")
    for name in categorical_cols:
        value = payload.values.get(name)
        if isinstance(value, (dict, list, tuple, set)):
            validation_errors.append(f"{name} must be a single value")
            continue
        if isinstance(value, str) and len(value) > 1000:
            validation_errors.append(f"{name} must be at most 1,000 characters")
            continue
        row[name] = np.nan if value is None or value == "" else value

    if validation_errors:
        raise HTTPException(422, {"field_errors": validation_errors})

    frame = pd.DataFrame([row], columns=expected)
    probabilities_for_row = None
    decision_threshold = workflow_result.get("decision_threshold")
    try:
        if (
            decision_threshold is not None
            and workflow_result.get("task_type") == "classification"
            and hasattr(pipeline, "predict_proba")
        ):
            probabilities_for_row = pipeline.predict_proba(frame)[0]
            classes = list(getattr(pipeline, "classes_", []))
            if len(classes) == 2:
                _, positive_index, negative_label = _positive_class_contract(
                    workflow_result,
                    pipeline,
                )
                predicted = (
                    classes[positive_index]
                    if float(probabilities_for_row[positive_index]) >= float(decision_threshold)
                    else negative_label
                )
            else:
                predicted = pipeline.predict(frame)[0]
        else:
            predicted = pipeline.predict(frame)[0]
    except Exception as exc:
        artifact_logger.warning("Prediction failed: %s", type(exc).__name__)
        raise HTTPException(422, "Prediction failed. Check the supplied feature values.")

    response = {
        "prediction": make_json_safe(predicted),
        "model": workflow_result.get("best_model"),
        "model_version": workflow_result.get("model_version") or (
            (workflow_result.get("run_manifest") or {}).get("run_id")
        ),
        "task_type": workflow_result.get("task_type"),
        "target_column": workflow_result.get("target_column"),
        "positive_label": workflow_result.get("positive_label"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision_threshold": decision_threshold,
        "latency_ms": round((time.perf_counter() - inference_started) * 1000, 3),
    }

    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = (
                probabilities_for_row
                if probabilities_for_row is not None
                else pipeline.predict_proba(frame)[0]
            )
            classes = getattr(pipeline, "classes_", range(len(probabilities)))
            response["probabilities"] = {
                str(label): round(float(probability), 6)
                for label, probability in zip(classes, probabilities)
            }
            response["confidence"] = round(float(np.max(probabilities)), 6)
        except Exception:
            pass

    importance = list(workflow_result.get("feature_importance") or [])[:5]
    if importance:
        response["global_drivers"] = importance

    history_item = {
        "id": uuid.uuid4().hex[:12],
        **response,
    }
    if _aivoraml_setting("RETAIN_PREDICTION_INPUTS", "false").lower() in {
        "1",
        "true",
        "yes",
    }:
        history_item["inputs"] = make_json_safe(row)
    with STORE_LOCK:
        if store.project_id != prediction_project_id:
            raise HTTPException(409, "The active project changed during prediction.")
        store.predictions.append(history_item)
        store.predictions = store.predictions[-5000:]
    await run_in_threadpool(_persist_prediction_history)
    return response


@app.get("/api/predictions")
async def get_predictions():
    if store.df is None:
        raise HTTPException(404, "No project is open.")
    return {"predictions": list(reversed(store.predictions[-100:]))}


@app.delete("/api/predictions")
async def clear_predictions():
    """Permanently clear prediction history for the active local project."""
    if store.df is None:
        raise HTTPException(404, "No project is open.")
    deleted = len(store.predictions)
    with STORE_LOCK:
        store.predictions = []
    await run_in_threadpool(_persist_prediction_history)
    await run_in_threadpool(_persist_workspace)
    return {"deleted": deleted, "project_id": store.project_id}


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Run a bounded dataset through the exact fitted pipeline and return CSV."""
    with STORE_LOCK:
        pipeline = _require_pipeline()
        batch_workflow = store.workflow_result or {}
        batch_project_id = store.project_id
        numeric_cols, categorical_cols = _feature_spec()
    expected = numeric_cols + categorical_cols
    if not expected:
        raise HTTPException(409, "The trained model has no recorded feature schema.")
    if not file.filename:
        raise HTTPException(400, "No batch file provided.")

    safe_filename = Path(file.filename.replace("\\", "/")).name.strip()
    if not safe_filename or "\x00" in safe_filename:
        raise HTTPException(400, "Invalid filename.")
    raw = await file.read(MAX_BATCH_BYTES + 1)
    if not raw:
        raise HTTPException(400, "The batch file is empty.")
    if len(raw) > MAX_BATCH_BYTES:
        raise HTTPException(
            413,
            f"Batch file exceeds the {MAX_BATCH_BYTES // (1024 * 1024)} MB limit.",
        )
    frame = await run_in_threadpool(_read_tabular_bytes, raw, safe_filename)
    if frame.empty:
        raise HTTPException(400, "The batch file contains no prediction rows.")
    if len(frame) > MAX_BATCH_ROWS:
        raise HTTPException(
            413,
            f"Batch has {len(frame):,} rows; the limit is {MAX_BATCH_ROWS:,}.",
        )
    if int(len(frame) * len(frame.columns)) > MAX_BATCH_CELLS:
        raise HTTPException(413, "Batch exceeds the configured cell limit.")
    if frame.columns.duplicated().any():
        raise HTTPException(422, "Batch columns must be unique.")

    target_column = batch_workflow.get("target_column")
    permitted_columns = set(expected)
    if target_column:
        permitted_columns.add(target_column)
    missing = [name for name in expected if name not in frame.columns]
    unknown = [str(name) for name in frame.columns if name not in permitted_columns]
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing required columns: {', '.join(missing[:20])}")
        if unknown:
            details.append(f"unknown columns: {', '.join(unknown[:20])}")
        raise HTTPException(422, "; ".join(details))

    inference_frame = frame.loc[:, expected].copy()
    field_errors: list[str] = []
    for name in numeric_cols:
        original = inference_frame[name]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        non_finite = converted.notna() & ~np.isfinite(converted)
        if invalid.any() or non_finite.any():
            rows = (invalid | non_finite).to_numpy().nonzero()[0][:5]
            field_errors.append(
                f"{name} has invalid numeric values at rows "
                + ", ".join(str(int(index) + 2) for index in rows)
            )
        inference_frame[name] = converted
    for name in categorical_cols:
        invalid_rows = [
            index
            for index, value in enumerate(inference_frame[name])
            if isinstance(value, (dict, list, tuple, set))
            or (isinstance(value, str) and len(value) > 1000)
        ][:5]
        if invalid_rows:
            field_errors.append(
                f"{name} has non-scalar or overlong values at rows "
                + ", ".join(str(index + 2) for index in invalid_rows)
            )
    if field_errors:
        raise HTTPException(422, {"field_errors": field_errors})

    decision_threshold = batch_workflow.get("decision_threshold")
    task_type = batch_workflow.get("task_type")

    def run_batch_inference():
        probabilities = None
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(inference_frame)
        if (
            probabilities is not None
            and decision_threshold is not None
            and task_type == "classification"
            and probabilities.shape[1] == 2
        ):
            classes = list(getattr(pipeline, "classes_", []))
            if len(classes) == 2:
                positive_label, positive_index, negative_label = _positive_class_contract(
                    batch_workflow,
                    pipeline,
                )
                predictions = np.where(
                    probabilities[:, positive_index] >= float(decision_threshold),
                    positive_label,
                    negative_label,
                )
            else:
                predictions = pipeline.predict(inference_frame)
        else:
            predictions = pipeline.predict(inference_frame)
        return predictions, probabilities

    inference_started = time.perf_counter()
    try:
        predictions, probabilities = await run_in_threadpool(run_batch_inference)
    except Exception as exc:
        artifact_logger.warning("Batch inference failed: %s", type(exc).__name__)
        raise HTTPException(422, "Batch prediction failed. Verify the feature values.")

    output = frame.copy()
    output["aivoraml_prediction"] = predictions
    if probabilities is not None:
        classes = list(getattr(pipeline, "classes_", range(probabilities.shape[1])))
        for index, label in enumerate(classes):
            safe_label = "".join(
                character if character.isalnum() else "_"
                for character in str(label)
            ).strip("_")[:60] or str(index)
            output[f"aivoraml_probability_{safe_label}"] = probabilities[:, index]
        output["aivoraml_max_probability"] = np.max(probabilities, axis=1)
    if store.project_id != batch_project_id:
        raise HTTPException(409, "The active project changed during batch prediction.")
    run_id = batch_workflow.get("model_version") or (
        (batch_workflow.get("run_manifest") or {}).get("run_id")
    )
    output["aivoraml_model"] = batch_workflow.get("best_model")
    output["aivoraml_model_version"] = run_id

    response = _csv_response(output, "aivoraml_batch_predictions.csv")
    response.headers["X-AivoraML-Rows"] = str(len(output))
    response.headers["X-AivoraML-Model-Version"] = str(run_id or "unversioned")
    response.headers["X-AivoraML-Latency-Ms"] = (
        f"{(time.perf_counter() - inference_started) * 1000:.3f}"
    )
    return response


@app.get("/api/dataset/download")
async def download_dataset():
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    return _csv_response(store.df, "aivoraml_dataset.csv")


@app.get("/api/predictions/download")
async def download_predictions():
    if not store.predictions:
        raise HTTPException(404, "No predictions have been made yet.")
    rows = []
    for item in store.predictions:
        row = {
            "id": item.get("id"),
            "timestamp": item.get("timestamp"),
            "prediction": item.get("prediction"),
            "model_probability": item.get("confidence"),
            "model": item.get("model"),
        }
        row.update({
            f"input_{key}": value
            for key, value in (item.get("inputs") or {}).items()
        })
        rows.append(row)
    return _csv_response(pd.DataFrame(rows), "aivoraml_predictions.csv")


@app.get("/api/metrics/download")
async def download_metrics():
    if not store.workflow_result:
        raise HTTPException(404, "No workflow result available.")
    candidates = (
        store.workflow_result.get("models_trained")
        or store.workflow_result.get("models_compared")
        or []
    )
    rows = []
    for candidate in candidates:
        row = {
            "model": candidate.get("name"),
            "is_best": candidate.get("is_best", False),
            "cv_score_mean": candidate.get("cv_score_mean"),
            "cv_score_std": candidate.get("cv_score_std"),
            "training_time_seconds": candidate.get("training_time_seconds"),
            "inference_time_ms": candidate.get("inference_time_ms"),
        }
        row.update(candidate.get("metrics") or {})
        rows.append(row)
    if not rows:
        rows = [{
            "model": store.workflow_result.get("best_model"),
            **(store.workflow_result.get("metrics") or {}),
        }]
    return _csv_response(pd.DataFrame(rows), "aivoraml_model_metrics.csv")


@app.get("/api/project/download")
async def download_project_summary():
    metadata = _project_metadata()
    workflow = {
        key: value
        for key, value in (store.workflow_result or {}).items()
        if not key.startswith("_")
    }
    artifact = {
        "schema_version": 1,
        "project": metadata,
        "profile": store.profile,
        "workflow": make_json_safe(workflow),
        "predictions": store.predictions,
    }
    return Response(
        content=json.dumps(make_json_safe(artifact), indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="aivoraml_project.json"'
        },
    )


@app.post("/api/dataset/imbalance")
async def check_imbalance(payload: dict):
    """Check class imbalance for a given target column."""
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    col = payload.get("target_column")
    if not col or col not in store.df.columns:
        raise HTTPException(400, "Invalid target column.")
    result = check_class_imbalance(store.df, col)
    return result or {"imbalanced": False}


@app.post("/api/workflow/run", include_in_schema=False)
@app.post("/api/workflow/run-simple", include_in_schema=False)
async def retired_legacy_workflow():
    """Reject workflow routes that bypass the project-bound training job contract."""
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail={
            "code": "LEGACY_WORKFLOW_RETIRED",
            "message": (
                "This training route was retired. Use POST /api/training/jobs "
                "and monitor the returned job resource."
            ),
        },
    )


@app.get("/api/model/download")
async def download_model():
    """Download the trained model as a .joblib file."""
    if store.workflow_result is None:
        raise HTTPException(404, "No trained model available. Run a workflow first.")

    pipeline = store.workflow_result.get("_pipeline")
    if pipeline is None:
        raise HTTPException(404, "No fitted pipeline available.")

    numeric_cols, categorical_cols = _feature_spec()
    best_candidate = next(
        (
            candidate
            for candidate in (
                store.workflow_result.get("models_trained")
                or store.workflow_result.get("models_compared")
                or []
            )
            if candidate.get("is_best")
        ),
        None,
    )
    profile_columns = {
        column.get("name"): column
        for column in (store.profile or {}).get("columns", [])
    }
    example_payload = {}
    for name in numeric_cols:
        example_payload[name] = (profile_columns.get(name, {}).get("stats") or {}).get(
            "median"
        )
    for name in categorical_cols:
        values = profile_columns.get(name, {}).get("top_values") or {}
        example_payload[name] = next(iter(values), None)
    classes = getattr(pipeline, "classes_", None)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aivoraml_version": app.version,
        "project_id": store.project_id,
        "dataset_fingerprint": (store.profile or {}).get("dataset_fingerprint"),
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "pipeline": pipeline,
        "target_column": store.workflow_result.get("target_column"),
        "task_type": store.workflow_result.get("task_type"),
        "best_model": store.workflow_result.get("best_model"),
        "metrics": (
            (store.workflow_result.get("evaluation_details") or {}).get("metrics")
            or store.workflow_result.get("metrics")
            or (best_candidate or {}).get("metrics")
        ),
        "best_score": store.workflow_result.get("best_score"),
        "feature_schema": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "ordered": numeric_cols + categorical_cols,
            "example_payload": make_json_safe(example_payload),
        },
        "class_labels": make_json_safe(classes) if classes is not None else None,
        "positive_label": store.workflow_result.get("positive_label"),
        "probability_semantics": (
            "predict_proba columns follow class_labels order"
            if classes is not None
            else None
        ),
        "decision_threshold": store.workflow_result.get("decision_threshold"),
        "threshold_selection": store.workflow_result.get("threshold_selection"),
        "probability_calibration_training": store.workflow_result.get(
            "probability_calibration_training"
        ),
        "model_version": store.workflow_result.get("model_version"),
        "training_timestamp": (
            (store.workflow_result.get("run_manifest") or {}).get("trained_at_utc")
        ),
        "random_seed": (
            (store.workflow_result.get("run_manifest") or {}).get("random_seed")
        ),
        "run_manifest": store.workflow_result.get("run_manifest"),
        "strategy": store.workflow_result.get("strategy"),
        "split": (
            store.workflow_result.get("split")
            or store.workflow_result.get("split_info")
        ),
        "model_selection": store.workflow_result.get("model_selection"),
        "data_balancing": store.workflow_result.get("data_balancing"),
        "cleaning": store.workflow_result.get("cleaning"),
        "warnings": store.workflow_result.get("warnings"),
    }

    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=aivoraml_model.joblib"},
    )


@app.get("/api/preprocessor/download")
async def download_preprocessor():
    """Download the fitted preprocessing transformer on its own."""
    pipeline = _require_pipeline()
    fitted = _extract_fitted_preprocessor(pipeline, store.workflow_result)
    if fitted is None:
        raise HTTPException(404, "No fitted preprocessor is available.")
    numeric_cols, categorical_cols = _feature_spec()
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preprocessor": fitted,
        "feature_schema": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "ordered": numeric_cols + categorical_cols,
        },
    }
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=aivoraml_preprocessor.joblib"
        },
    )


def _ensure_current_model_version() -> Optional[dict]:
    """Lazily register a legacy active model created before versioning existed."""
    if store.df is None or not store.workflow_result:
        return None
    pipeline = store.workflow_result.get("_pipeline")
    if pipeline is None:
        return None
    active_version = store.workflow_result.get("model_version")
    if active_version:
        try:
            return model_registry.get_metadata(store.project_id, active_version)
        except ModelVersionNotFoundError:
            pass
    run_manifest = store.workflow_result.setdefault("run_manifest", {})
    run_id = str(run_manifest.get("run_id") or uuid.uuid4().hex)
    run_manifest["run_id"] = run_id
    for entry in model_registry.list_versions(store.project_id):
        if (entry.get("metadata") or {}).get("run_id") == run_id:
            store.workflow_result["model_version"] = entry["version_id"]
            run_manifest["model_version"] = entry["version_id"]
            return entry
    return _register_model_version(
        project_id=store.project_id,
        pipeline=pipeline,
        workflow_result=store.workflow_result,
        approach=str(store.workflow_result.get("strategy") or "balanced"),
    )


@app.get("/api/models/versions", tags=["model registry"])
async def list_model_versions():
    if store.df is None:
        raise HTTPException(404, "No project is open.")
    with STORE_LOCK:
        migrated = _ensure_current_model_version()
        active_version = store.workflow_result.get("model_version") if store.workflow_result else None
        versions = model_registry.list_versions(store.project_id)
    if migrated:
        await run_in_threadpool(_persist_workspace)
    for entry in versions:
        entry["active"] = entry["version_id"] == active_version
    return {
        "project_id": store.project_id,
        "active_version": active_version,
        "versions": versions,
    }


def _model_version_metadata_or_404(version_id: str) -> dict:
    if store.df is None:
        raise HTTPException(404, "No project is open.")
    try:
        return model_registry.get_metadata(store.project_id, version_id)
    except (ModelVersionNotFoundError, InvalidModelIdentifierError):
        raise HTTPException(404, "Model version not found.")


@app.get("/api/models/versions/{version_id}", tags=["model registry"])
async def get_model_version(version_id: str):
    metadata = _model_version_metadata_or_404(version_id)
    try:
        model_registry.verify_artifact(store.project_id, version_id)
    except ModelArtifactError:
        raise HTTPException(409, "The model artifact failed its integrity check.")
    metadata["active"] = bool(
        store.workflow_result
        and store.workflow_result.get("model_version") == version_id
    )
    return metadata


@app.get(
    "/api/models/versions/{version_id}/download",
    tags=["model registry"],
)
async def download_model_version(version_id: str):
    _model_version_metadata_or_404(version_id)
    try:
        artifact_bytes = await run_in_threadpool(
            model_registry.read_artifact_bytes,
            store.project_id,
            version_id,
        )
    except ModelArtifactError:
        raise HTTPException(409, "The model artifact failed its integrity check.")
    return Response(
        content=artifact_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": (
                f'attachment; filename="aivoraml_{version_id}.joblib"'
            )
        },
    )


@app.post(
    "/api/models/versions/{version_id}/activate",
    tags=["model registry"],
)
async def activate_model_version(version_id: str):
    metadata = _model_version_metadata_or_404(version_id)
    try:
        artifact = await run_in_threadpool(
            model_registry.load_artifact,
            store.project_id,
            version_id,
        )
    except ModelArtifactError:
        raise HTTPException(409, "The model artifact failed its integrity check.")
    public_metadata = artifact.get("public_metadata") or {}
    expected_fingerprint = public_metadata.get("dataset_fingerprint")
    active_fingerprint = (store.profile or {}).get("dataset_fingerprint")
    if expected_fingerprint and active_fingerprint != expected_fingerprint:
        raise HTTPException(
            409,
            "This model version belongs to a different dataset version and cannot be activated here.",
        )
    workflow = dict(public_metadata.get("workflow") or {})
    if not workflow:
        raise HTTPException(409, "This model version lacks restorable workflow metadata.")
    workflow["model_version"] = version_id
    run_manifest = workflow.setdefault("run_manifest", {})
    run_manifest["model_version"] = version_id
    pipeline = artifact["pipeline"]
    with STORE_LOCK:
        workflow["_pipeline"] = pipeline
        workflow["_preprocessor"] = _extract_fitted_preprocessor(pipeline)
        store.workflow_result = workflow
        store.report_bundle = None
    await run_in_threadpool(_persist_workspace)
    return {
        "activated": version_id,
        "best_model": workflow.get("best_model"),
        "target_column": workflow.get("target_column"),
        "metadata": metadata,
    }


REPORT_MEDIA_TYPES = {
    "notebook": "application/x-ipynb+json",
    "html_report": "text/html; charset=utf-8",
    "pdf_report": "application/pdf",
    "model_summary": "application/json",
    "metrics": "text/csv; charset=utf-8",
    "feature_schema": "application/json",
    "pipeline": "application/octet-stream",
}
ProjectIdQuery = Annotated[
    str,
    Query(
        min_length=12,
        max_length=12,
        pattern=r"^[0-9a-f]{12}$",
        description="Active AivoraML project ID",
    ),
]


def _require_active_project(project_id: str) -> None:
    if store.df is None or not store.project_id:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "PROJECT_NOT_FOUND",
                "message": "No project is open.",
            },
        )
    if project_id != store.project_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "PROJECT_ACCESS_DENIED",
                "message": "The requested project is not the active project.",
            },
        )


def _resolve_report_artifact(
    bundle: dict,
    artifact_key: str,
    project_id: str,
) -> Path:
    _require_active_project(project_id)
    artifact_path = (bundle.get("artifacts") or {}).get(artifact_key)
    if not artifact_path:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ARTIFACT_NOT_FOUND",
                "message": f"Report artifact is unavailable: {artifact_key}.",
            },
        )
    resolved = (PROJECT_ROOT / artifact_path).resolve()
    artifact_logger.info(
        "Resolved artifact project_id=%s key=%s path=%s exists=%s",
        project_id,
        artifact_key,
        resolved,
        resolved.exists(),
    )
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ARTIFACT_PATH_INVALID",
                "message": "The artifact path is outside the project root.",
            },
        )
    if not resolved.is_file():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "ARTIFACT_FILE_MISSING",
                "message": f"Report artifact file is unavailable: {artifact_key}.",
            },
        )
    try:
        size = resolved.stat().st_size
        with resolved.open("rb") as artifact_stream:
            first_byte = artifact_stream.read(1)
    except OSError as exc:
        artifact_logger.exception(
            "Artifact is unreadable project_id=%s key=%s path=%s",
            project_id,
            artifact_key,
            resolved,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ARTIFACT_UNREADABLE",
                "message": "The generated artifact is not readable.",
            },
        ) from exc
    if size <= 0 or not first_byte:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ARTIFACT_EMPTY",
                "message": "The generated artifact is empty.",
            },
        )
    return resolved


def _report_file_response(
    bundle: dict,
    artifact_key: str,
    project_id: str,
) -> FileResponse:
    resolved = _resolve_report_artifact(bundle, artifact_key, project_id)
    return FileResponse(
        path=resolved,
        media_type=REPORT_MEDIA_TYPES[artifact_key],
        filename=resolved.name,
    )


def _artifact_file_is_ready(artifact_path: str) -> bool:
    resolved = (PROJECT_ROOT / artifact_path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT) or not resolved.is_file():
        return False
    try:
        if resolved.stat().st_size <= 0:
            return False
        with resolved.open("rb") as artifact_stream:
            return bool(artifact_stream.read(1))
    except OSError:
        return False


async def _ensure_report_bundle(
    project_id: str,
    force: bool = False,
) -> dict:
    """Return or synchronously build a bundle through the single-writer gate."""
    _require_active_project(project_id)
    active_job = training_jobs.get_active_job(project_id)
    if active_job:
        raise HTTPException(
            409,
            {
                "message": "A project operation is already running.",
                "active_job_id": active_job["job_id"],
                "job_type": (active_job.get("metadata") or {}).get("job_type"),
            },
        )
    if not REPORT_GENERATION_LOCK.acquire(blocking=False):
        raise HTTPException(409, "A report bundle is already being generated.")
    try:
        return await _ensure_report_bundle_unlocked(project_id, force=force)
    finally:
        REPORT_GENERATION_LOCK.release()


async def _ensure_report_bundle_unlocked(
    project_id: str,
    force: bool = False,
) -> dict:
    with STORE_LOCK:
        _require_active_project(project_id)
        if store.df is None or not store.workflow_result:
            raise HTTPException(404, "Train a model before generating project reports.")
        existing = store.report_bundle or {}
    if existing.get("status") == "generating":
        raise HTTPException(409, "A report bundle is already being generated.")
    if not force and existing.get("status") == "completed":
        artifact_paths = list((existing.get("artifacts") or {}).values())
        if artifact_paths and all(
            _artifact_file_is_ready(path)
            for path in artifact_paths
        ):
            return existing

    with STORE_LOCK:
        _require_active_project(project_id)
        report_project_id = store.project_id
        report_result = store.workflow_result
        report_profile = store.profile or {}
        report_dataset = store.df.copy()
        report_filename = f"{report_project_id}_{store.filename}"
        store.report_bundle = {
            "status": "generating",
            "current_section": "Executing complete notebook",
            "completed_sections": [],
            "warnings": [],
        }

    def update_report_progress(
        current_section: str,
        completed_sections: list[str],
    ) -> None:
        with STORE_LOCK:
            if (
                store.project_id == report_project_id
                and store.report_bundle
                and store.report_bundle.get("status") == "generating"
            ):
                store.report_bundle["current_section"] = current_section
                store.report_bundle["completed_sections"] = completed_sections

    try:
        bundle = await run_in_threadpool(
            lambda: generate_report_bundle(
                project_root=PROJECT_ROOT,
                result=report_result,
                profile=report_profile,
                dataset=report_dataset,
                filename=report_filename,
                progress_callback=update_report_progress,
            )
        )
        bundle["current_section"] = "Complete"
        bundle["completed_sections"] = [
            "Notebook generated",
            "All cells executed",
            "Outputs embedded",
            "Pipeline reloaded",
            "HTML rendered",
            "PDF rendered and verified",
            "Metadata and metrics saved",
        ]
        bundle["warnings"] = list(
            (report_result or {}).get("warnings") or []
        )
        with STORE_LOCK:
            if store.project_id != report_project_id:
                raise HTTPException(
                    409,
                    "The active project changed while reports were being generated.",
                )
            store.report_bundle = bundle
        await run_in_threadpool(_persist_workspace)
        return bundle
    except ReportInputError as exc:
        artifact_logger.exception(
            "Report input rejected project_id=%s",
            project_id,
        )
        with STORE_LOCK:
            if store.project_id == report_project_id:
                store.report_bundle = {
                    "status": "failed",
                    "current_section": "Report generation failed",
                    "completed_sections": [],
                    "warnings": ["Report input is invalid."],
                }
        raise HTTPException(
            status_code=422,
            detail={
                "code": exc.code,
                "message": str(exc),
            },
        ) from exc
    except ReportGenerationError as exc:
        artifact_logger.exception(
            "Report generation failed project_id=%s code=%s",
            project_id,
            exc.code,
        )
        with STORE_LOCK:
            if store.project_id == report_project_id:
                store.report_bundle = {
                    "status": "failed",
                    "current_section": "Report generation failed",
                    "completed_sections": [],
                    "warnings": ["Report generation failed. Check server logs."],
                }
        raise HTTPException(
            status_code=500,
            detail={
                "code": exc.code,
                "message": "Report generation failed. Check server logs.",
            },
        ) from exc


def _execute_report_job(
    *,
    report_project_id: str,
    report_result: dict,
    report_profile: dict,
    report_dataset: pd.DataFrame,
    report_filename: str,
    force: bool,
    report_progress,
    is_cancelled,
) -> dict:
    """Run one background report through the process-wide single-writer gate."""
    if not REPORT_GENERATION_LOCK.acquire(blocking=False):
        raise RuntimeError("Another report bundle is already being generated.")
    try:
        return _execute_report_job_unlocked(
            report_project_id=report_project_id,
            report_result=report_result,
            report_profile=report_profile,
            report_dataset=report_dataset,
            report_filename=report_filename,
            force=force,
            report_progress=report_progress,
            is_cancelled=is_cancelled,
        )
    finally:
        REPORT_GENERATION_LOCK.release()


def _execute_report_job_unlocked(
    *,
    report_project_id: str,
    report_result: dict,
    report_profile: dict,
    report_dataset: pd.DataFrame,
    report_filename: str,
    force: bool,
    report_progress,
    is_cancelled,
) -> dict:
    """Generate a reconnectable report bundle with section-level progress."""
    with STORE_LOCK:
        if store.project_id != report_project_id:
            raise RuntimeError("The active project changed before report generation.")
        existing = store.report_bundle or {}
        if not force and existing.get("status") == "completed":
            paths = list((existing.get("artifacts") or {}).values())
            if paths and all(_artifact_file_is_ready(path) for path in paths):
                return existing
        store.report_bundle = {
            "status": "generating",
            "current_section": "Preparing report workspace",
            "completed_sections": [],
            "warnings": [],
        }

    def update_report_progress(current_section: str, completed: list[str]) -> None:
        if is_cancelled():
            raise TrainingJobCancelled("Report generation cancelled between sections.")
        percent = min(94.0, 8.0 + len(completed) / 7.0 * 84.0)
        report_progress(percent, "reporting", current_section)
        with STORE_LOCK:
            if store.project_id == report_project_id and store.report_bundle:
                store.report_bundle["current_section"] = current_section
                store.report_bundle["completed_sections"] = list(completed)

    report_progress(5, "preparing", "Preparing the isolated report workspace.")
    try:
        bundle = generate_report_bundle(
            project_root=PROJECT_ROOT,
            result=report_result,
            profile=report_profile,
            dataset=report_dataset,
            filename=report_filename,
            progress_callback=update_report_progress,
        )
        if is_cancelled():
            raise TrainingJobCancelled("Report generation cancelled before commit.")
        bundle["current_section"] = "Complete"
        bundle["completed_sections"] = [
            "Notebook generated",
            "All cells executed",
            "Outputs embedded",
            "Pipeline reloaded",
            "HTML rendered",
            "PDF rendered and verified",
            "Metadata and metrics saved",
        ]
        bundle["warnings"] = list(report_result.get("warnings") or [])
        with STORE_LOCK:
            if store.project_id != report_project_id:
                raise RuntimeError("The active project changed during report generation.")
            store.report_bundle = bundle
        _persist_workspace()
        report_progress(99, "verifying", "Verified every generated artifact.")
        return make_json_safe(bundle)
    except TrainingJobCancelled:
        with STORE_LOCK:
            if store.project_id == report_project_id:
                store.report_bundle = {
                    "status": "cancelled",
                    "current_section": "Cancelled",
                    "completed_sections": [],
                    "warnings": ["Report generation was cancelled."],
                }
        raise
    except (ReportInputError, ReportGenerationError) as exc:
        artifact_logger.exception("Background report generation failed")
        with STORE_LOCK:
            if store.project_id == report_project_id:
                store.report_bundle = {
                    "status": "failed",
                    "current_section": "Report generation failed",
                    "completed_sections": [],
                    "warnings": ["Report generation failed. Check server logs."],
                }
        raise RuntimeError("Report generation failed.") from exc


@app.post(
    "/api/reports/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["report jobs"],
)
async def create_report_job(project_id: ProjectIdQuery, force: bool = False):
    if REPORT_GENERATION_LOCK.locked():
        raise HTTPException(409, "A report bundle is already being generated.")
    with STORE_LOCK:
        _require_active_project(project_id)
        if store.df is None or not store.workflow_result:
            raise HTTPException(404, "Train a model before generating reports.")
        report_result = store.workflow_result
        report_profile = store.profile or {}
        report_dataset = store.df.copy()
        report_filename = f"{project_id}_{store.filename}"

    def runner(report_progress, is_cancelled):
        return _execute_report_job(
            report_project_id=project_id,
            report_result=report_result,
            report_profile=report_profile,
            report_dataset=report_dataset,
            report_filename=report_filename,
            force=force,
            report_progress=report_progress,
            is_cancelled=is_cancelled,
        )

    try:
        return training_jobs.submit(
            project_id,
            runner,
            metadata={
                "job_type": "report",
                "run_id": (report_result.get("run_manifest") or {}).get("run_id"),
            },
            message="Waiting for an available report worker.",
        )
    except ActiveTrainingJobError as exc:
        raise HTTPException(
            409,
            {
                "message": "This project already has an active operation.",
                "active_job_id": exc.active_job_id,
            },
        )
    except TrainingJobQueueFullError:
        raise HTTPException(429, "The local job queue is full. Try again shortly.")


@app.post("/api/reports/generate")
async def generate_reports(project_id: ProjectIdQuery, force: bool = False):
    """Generate and verify the complete executed notebook/report bundle."""
    return await _ensure_report_bundle(project_id=project_id, force=force)


@app.get("/api/reports/current")
async def get_current_report(project_id: ProjectIdQuery):
    _require_active_project(project_id)
    if not store.report_bundle:
        raise HTTPException(404, "No report bundle has been generated.")
    return store.report_bundle


@app.get("/api/reports/html")
async def download_html_report(project_id: ProjectIdQuery):
    bundle = await _ensure_report_bundle(project_id)
    return _report_file_response(bundle, "html_report", project_id)


@app.get("/api/reports/pdf")
async def download_pdf_report(project_id: ProjectIdQuery):
    bundle = await _ensure_report_bundle(project_id)
    return _report_file_response(bundle, "pdf_report", project_id)


@app.get("/api/reports/summary")
async def download_report_summary(project_id: ProjectIdQuery):
    bundle = await _ensure_report_bundle(project_id)
    return _report_file_response(bundle, "model_summary", project_id)


@app.get("/api/reports/metrics")
async def download_report_metrics(project_id: ProjectIdQuery):
    bundle = await _ensure_report_bundle(project_id)
    return _report_file_response(bundle, "metrics", project_id)


@app.get("/api/reports/feature-schema")
async def download_report_feature_schema(project_id: ProjectIdQuery):
    bundle = await _ensure_report_bundle(project_id)
    return _report_file_response(bundle, "feature_schema", project_id)


@app.get("/api/reports/pipeline")
async def download_report_pipeline(project_id: ProjectIdQuery):
    bundle = await _ensure_report_bundle(project_id)
    return _report_file_response(bundle, "pipeline", project_id)


@app.get("/api/notebook/download")
async def download_notebook(project_id: ProjectIdQuery):
    """Generate when needed and download the executed project notebook."""
    bundle = await _ensure_report_bundle(project_id)
    return _report_file_response(bundle, "notebook", project_id)


def _assistant_workspace_context(
    *,
    view: str,
    strategy: str,
    target_column: Optional[str],
) -> dict:
    """Build compact project evidence without including raw dataset rows."""
    with STORE_LOCK:
        profile = dict(store.profile or {})
        workflow = dict(store.workflow_result or {})
        project = {
            "id": store.project_id or None,
            "filename": store.filename or None,
        }

    columns = []
    for column in list(profile.get("columns") or [])[:40]:
        columns.append({
            "name": column.get("name"),
            "semantic_type": column.get("semantic_type"),
            "missing_pct": column.get("missing_pct"),
            "n_unique": column.get("n_unique"),
        })
    warnings = []
    for warning in list(profile.get("warnings") or [])[:8]:
        warnings.append(
            warning.get("message") if isinstance(warning, dict) else str(warning)
        )
    evaluation = workflow.get("evaluation_details") or {}
    model_rows = []
    for model in list(workflow.get("models_trained") or [])[:12]:
        model_rows.append({
            "name": model.get("name"),
            "cv_score_mean": model.get("cv_score_mean"),
            "cv_score_std": model.get("cv_score_std"),
            "selection_score": model.get("selection_score"),
            "is_best": model.get("is_best", False),
        })
    return make_json_safe({
        "workspace_view": view,
        "selected_strategy": strategy,
        "selected_target": target_column or workflow.get("target_column"),
        "project": project,
        "dataset": {
            "rows": profile.get("n_rows"),
            "columns": profile.get("n_cols"),
            "missing_cells": profile.get("total_missing"),
            "duplicates": profile.get("n_duplicates"),
            "quality_score": profile.get("quality_score"),
            "schema": columns,
            "warnings": warnings,
        } if profile else None,
        "trained_model": {
            "task_type": workflow.get("task_type"),
            "target": workflow.get("target_column"),
            "champion": workflow.get("best_model"),
            "selection": workflow.get("model_selection"),
            "holdout_metrics": evaluation.get("metrics"),
            "threshold": workflow.get("decision_threshold"),
            "top_features": list(workflow.get("feature_importance") or [])[:10],
            "candidates": model_rows,
            "warnings": list(workflow.get("warnings") or [])[:8],
            "model_version": workflow.get("model_version"),
        } if workflow else None,
    })


@app.post("/api/ai/chat", tags=["AI assistant"])
async def assistant_chat(payload: AssistantChatRequest):
    """Answer with Gemini first and Mistral fallback using safe project context."""
    assistant = get_project_assistant()
    if assistant is None:
        raise HTTPException(
            503,
            "Remote AI is disabled or no supported provider key is configured.",
        )
    context = _assistant_workspace_context(
        view=payload.view,
        strategy=payload.strategy.value,
        target_column=payload.target_column,
    )
    history = [turn.model_dump() for turn in payload.history]
    try:
        return await run_in_threadpool(
            assistant.answer,
            payload.message,
            context,
            history,
        )
    except AssistantUnavailable:
        raise HTTPException(
            503,
            "The configured AI providers are temporarily unavailable.",
        )
    except Exception:
        artifact_logger.exception("Project assistant request failed")
        raise HTTPException(500, "The project assistant could not answer this request.")


@app.post("/api/ai/recommend-models")
async def get_model_recommendations(payload: dict):
    """Get AI-powered model recommendations based on dataset and target."""
    if store.profile is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    
    target_column = payload.get("target_column")
    task_type = payload.get("task_type", "auto")
    
    if not target_column:
        raise HTTPException(400, "target_column is required.")
    
    advisor = get_ai_advisor()
    if not advisor:
        raise HTTPException(
            503,
            "Remote AI guidance is disabled or unavailable in this local workspace.",
        )
    
    try:
        recommendations = advisor.recommend_models(store.profile, target_column, task_type)
        return recommendations
    except Exception as exc:
        artifact_logger.exception("Model recommendation failed")
        raise HTTPException(500, "Model recommendations are temporarily unavailable.")


@app.post("/api/ai/feature-engineering")
async def get_feature_suggestions(payload: dict):
    """Get AI-powered feature engineering suggestions."""
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    
    target_column = payload.get("target_column")
    task_type = payload.get("task_type", "auto")
    
    if not target_column:
        raise HTTPException(400, "target_column is required.")
    
    advisor = get_ai_advisor()
    if not advisor:
        raise HTTPException(
            503,
            "Remote AI guidance is disabled or unavailable in this local workspace.",
        )
    
    try:
        suggestions = advisor.suggest_feature_engineering(store.df, target_column, task_type)
        return suggestions
    except Exception as exc:
        artifact_logger.exception("Feature recommendation failed")
        raise HTTPException(500, "Feature recommendations are temporarily unavailable.")


@app.get("/api/ai/explain-results")
async def get_results_explanation():
    """Get AI-powered explanation of workflow results."""
    if store.workflow_result is None:
        raise HTTPException(404, "No workflow result available.")
    
    advisor = get_ai_advisor()
    if not advisor:
        raise HTTPException(
            503,
            "Remote AI guidance is disabled or unavailable in this local workspace.",
        )
    
    try:
        # Create a safe copy for explanation
        safe_result = {k: v for k, v in store.workflow_result.items() if not k.startswith("_")}
        explanation = advisor.explain_results(safe_result)
        return explanation
    except Exception as exc:
        artifact_logger.exception("Result explanation failed")
        raise HTTPException(500, "The result explanation is temporarily unavailable.")


@app.get("/api/ai/status")
async def get_ai_status():
    """Report assistant configuration without making a billable provider call."""
    status_payload = assistant_status()
    models = {}
    if "gemini" in status_payload["configured_providers"]:
        models["gemini"] = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    if "mistral" in status_payload["configured_providers"]:
        models["mistral"] = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    return {
        **status_payload,
        "models": models,
        "status": "ready" if status_payload["available"] else "local_fallback",
    }


@app.get("/api/debug/dataset-info")
async def debug_dataset_info():
    """Debug endpoint to check dataset state."""
    if _aivoraml_setting("ENABLE_DEBUG").lower() not in {"1", "true", "yes"}:
        raise HTTPException(404, "Not found.")
    if store.df is None:
        return {"error": "No dataset uploaded"}
    
    return {
        "shape": store.df.shape,
        "columns": list(store.df.columns),
        "dtypes": {col: str(dtype) for col, dtype in store.df.dtypes.items()},
        "sample": store.df.head(3).to_dict()
    }


@app.post("/api/ai/generate-approaches")
async def generate_ml_approaches(payload: dict):
    """Generate multiple ML approaches for user to choose from."""
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    
    target_column = payload.get("target_column")
    if not target_column or target_column not in store.df.columns:
        raise HTTPException(400, "Valid target_column is required.")
    
    advisor = get_ai_advisor()
    if not advisor:
        # No AI advisor? Return fallback approaches directly
        from .services.ai_advisor import AIAdvisor
        temp_advisor = AIAdvisor.__new__(AIAdvisor)
        approaches = temp_advisor._generate_fallback_approaches(store.df, target_column)
        return approaches
    
    try:
        approaches = advisor.generate_multiple_approaches(store.df, target_column)
        return approaches
    except Exception as exc:
        # Log the full error
        artifact_logger.exception("Approach generation failed")
        
        # Return fallback approaches instead of failing
        try:
            approaches = advisor._generate_fallback_approaches(store.df, target_column)
            return approaches
        except Exception as fallback_exc:
            artifact_logger.exception("Approach generation fallback failed")
            raise HTTPException(500, "Could not generate training approaches.")


@app.post("/api/ai/generate-pipeline")
async def generate_ml_pipeline(payload: dict):
    """Generate a complete ML pipeline based on selected approach."""
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    
    target_column = payload.get("target_column")
    approach = payload.get("approach", "balanced")
    if approach not in {"fast", "balanced", "high_performance"}:
        raise HTTPException(
            422,
            "approach must be one of: fast, balanced, high_performance",
        )
    
    if not target_column or target_column not in store.df.columns:
        raise HTTPException(400, "Valid target_column is required.")
    
    try:
        # Always generate a working pipeline (AI or fallback)
        pipeline = _generate_simple_pipeline(store.df, target_column, approach)
        
        return pipeline
        
    except Exception as exc:
        artifact_logger.exception("Pipeline planning failed")
        raise HTTPException(500, "Could not generate the pipeline plan.")


def _generate_simple_pipeline(df: pd.DataFrame, target_column: str, approach: str) -> dict:
    """Generate a simple fallback pipeline without AI."""
    
    try:
        is_classification = df[target_column].nunique() <= 20
        task_type = "classification" if is_classification else "regression"
        
        # Count feature types
        feature_cols = [c for c in df.columns if c != target_column]
        numeric_cols = len(df[feature_cols].select_dtypes(include=['int64', 'float64']).columns)
        categorical_cols = len(df[feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns)
        
        # Set hyperparameters based on approach
        if approach == "fast":
            n_estimators = 50
            max_depth = 10
        elif approach == "high_performance":
            n_estimators = 300
            max_depth = 20
        else:  # balanced
            n_estimators = 200
            max_depth = 15
        
        return {
            "ai_powered": False,
            "provider": "Deterministic AivoraML planner",
            "plan_kind": "training_configuration",
            "approach": approach,
            "analysis": {
                "data_quality": "Automated assessment completed",
                "complexity": "moderate",
                "recommended_approach": f"Standard sklearn pipeline for {task_type}",
                "expected_performance": "Measured only after cross-validation and holdout evaluation",
            },
            "preprocessing": {
                "steps": [
                    "Remove duplicate rows",
                    "Drop rows with missing target values",
                    f"Impute missing values in {numeric_cols} numeric features (median strategy)",
                    f"Impute missing values in {categorical_cols} categorical features (mode strategy)",
                    "Scale numeric features using StandardScaler",
                    "Encode categorical features using OneHotEncoder"
                ],
                "rationale": "Standard preprocessing pipeline ensures clean, normalized data for machine learning"
            },
            "models": [
                {
                    "name": "Random Forest",
                    "algorithm": "RandomForestClassifier" if is_classification else "RandomForestRegressor",
                    "hyperparameters": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "priority": "high",
                    "rationale": "Random Forest is robust, handles mixed feature types well, and provides feature importance"
                }
            ],
            "feature_engineering": {
                "transformations": [
                    "Standard scaling for numeric features",
                    "One-hot encoding for categorical features",
                    "Automatic handling of missing values"
                ],
                "new_features": []
            },
            "optimization": {
                "tuning_strategy": "RandomizedSearchCV with 5-fold cross-validation" if approach == "high_performance" else "Default hyperparameters optimized for your data size",
                "cv_folds": 5 if approach == "high_performance" else 0,
                "search_space": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [10, 15, 20, 25],
                    "min_samples_split": [2, 5, 10]
                } if approach == "high_performance" else {}
            }
        }
    except Exception as e:
        # Ultimate fallback - return minimal working pipeline
        artifact_logger.warning("Deterministic pipeline planning failed: %s", type(e).__name__)
        return {
            "ai_powered": False,
            "provider": "Minimal Fallback",
            "analysis": {
                "data_quality": "Processing",
                "complexity": "standard",
                "recommended_approach": "Standard ML pipeline",
                "expected_performance": "Not estimated before training"
            },
            "preprocessing": {
                "steps": ["Clean data", "Scale features", "Encode categories"],
                "rationale": "Standard preprocessing"
            },
            "models": [{
                "name": "Random Forest",
                "algorithm": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 100, "random_state": 42},
                "priority": "high",
                "rationale": "Reliable baseline model"
            }],
            "feature_engineering": {
                "transformations": ["Standard preprocessing"],
                "new_features": []
            },
            "optimization": {
                "tuning_strategy": "Default parameters",
                "cv_folds": 0,
                "search_space": {}
            }
        }


@app.post("/api/ai/execute-pipeline")
async def execute_ai_pipeline(payload: dict):
    """Execute the selected deterministic plan through the real enhanced workflow."""
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    
    target_column = payload.get("target_column")
    if not target_column:
        raise HTTPException(400, "target_column is required.")
    
    if target_column not in store.df.columns:
        raise HTTPException(400, f"Column '{target_column}' not found. Available: {list(store.df.columns)}")
    
    approach = payload.get("approach", "balanced")
    if approach not in {"fast", "balanced", "high_performance"}:
        raise HTTPException(
            422,
            "approach must be one of: fast, balanced, high_performance",
        )

    try:
        response = await professional_ml_workflow({
            "target_column": target_column,
            "approach": approach,
        })
        return response
    except HTTPException:
        raise
    except Exception as exc:
        artifact_logger.exception("Pipeline execution failed")
        raise HTTPException(500, "Pipeline execution failed.")


def _training_strategy(approach: str) -> dict:
    strategies = {
        "fast": {"max_models": 5, "tune_best": False},
        "balanced": {"max_models": 8, "tune_best": False},
        "high_performance": {"max_models": 9, "tune_best": True},
    }
    try:
        return strategies[approach]
    except KeyError:
        raise HTTPException(
            422,
            "approach must be one of: fast, balanced, high_performance",
        )


def _execute_training_job(
    *,
    training_df: pd.DataFrame,
    target_column: str,
    approach: str,
    include_eda: bool,
    training_project_id: str,
    source_filename: str,
    dataset_fingerprint: Optional[str],
    report_progress,
    is_cancelled,
) -> dict:
    """Execute and commit one background run with truthful coarse progress."""
    strategy = _training_strategy(approach)

    def checkpoint(percent: float, stage: str, message: str) -> None:
        if is_cancelled():
            raise TrainingJobCancelled("Training cancelled by the user.")
        report_progress(percent, stage, message)

    checkpoint(5, "validating", "Validating the target and experiment budget.")
    target_series = training_df[target_column]
    task_type = (
        "classification"
        if (
            not pd.api.types.is_numeric_dtype(target_series)
            or target_series.nunique(dropna=True) <= 20
        )
        else "regression"
    )
    eda_results = []
    if include_eda:
        checkpoint(12, "eda", "Generating dataset-specific exploratory analysis.")
        eda_results = generate_comprehensive_eda(
            training_df,
            target_column,
            task_type,
        )

    checkpoint(
        22,
        "training",
        f"Comparing up to {strategy['max_models'] + 1} candidates with cross-validation.",
    )
    def candidate_progress(completed: int, total: int, model_name: str) -> None:
        if is_cancelled():
            raise WorkflowCancelled("Training cancelled between candidates.")
        fraction = completed / max(total, 1)
        report_progress(
            22 + 60 * fraction,
            "training",
            (
                f"Completed {completed} of {total} candidates."
                if completed
                else f"Training {model_name}."
            ),
        )

    try:
        ml_results = run_enhanced_workflow(
            training_df,
            target_column=target_column,
            task_type="auto",
            max_models=strategy["max_models"],
            tune_best=strategy["tune_best"],
            progress_callback=candidate_progress,
            is_cancelled=is_cancelled,
        )
    except WorkflowCancelled as exc:
        raise TrainingJobCancelled(str(exc))
    checkpoint(86, "validating", "Checking the champion and untouched holdout evidence.")

    run_manifest = ml_results.setdefault("run_manifest", {})
    run_manifest["run_id"] = uuid.uuid4().hex
    run_manifest["project_id"] = training_project_id
    run_manifest["dataset_fingerprint"] = dataset_fingerprint
    run_manifest["source_filename"] = source_filename

    pipeline = ml_results.pop("_pipeline", None)
    preprocessor = ml_results.pop("_preprocessor", None)
    models_trained = ml_results.get("models_trained") or []
    best_score = ml_results.get("best_score")
    if (
        pipeline is None
        or not models_trained
        or not ml_results.get("best_model")
        or best_score is None
        or not np.isfinite(float(best_score))
    ):
        details = "; ".join(ml_results.get("warnings") or [])
        raise ValueError(
            "No model could be trained for this dataset."
            + (f" {details}" if details else "")
        )
    checkpoint(92, "saving", "Saving the fitted pipeline and run lineage.")
    with STORE_LOCK:
        if store.project_id != training_project_id:
            raise RuntimeError(
                "The active project changed while training. Start a new run for the active project."
            )
        registry_entry = _register_model_version(
            project_id=training_project_id,
            pipeline=pipeline,
            workflow_result=ml_results,
            approach=approach,
        )
        ml_results["model_artifact"] = {
            "version_id": registry_entry["version_id"],
            "sha256": registry_entry["sha256"],
            "size_bytes": registry_entry["size_bytes"],
            "registered_at": registry_entry["registered_at"],
        }
        store.workflow_result = ml_results.copy()
        store.workflow_result["_pipeline"] = pipeline
        store.workflow_result["_preprocessor"] = preprocessor
        store.workflow_result["strategy"] = approach
        store.report_bundle = None
    _persist_workspace()

    checkpoint(98, "finalizing", "Preparing the experiment results.")
    return make_json_safe({
        "success": True,
        "task_type": ml_results.get("task_type"),
        "data_info": ml_results.get("data_shape"),
        "cleaning_summary": ml_results.get("cleaning"),
        "feature_summary": ml_results.get("features"),
        "split_info": ml_results.get("split"),
        "data_balancing": ml_results.get("data_balancing"),
        "class_distribution": ml_results.get("class_distribution"),
        "models_trained": models_trained,
        "best_model": ml_results.get("best_model"),
        "best_score": best_score,
        "target_column": target_column,
        "positive_label": ml_results.get("positive_label"),
        "model_selection": ml_results.get("model_selection"),
        "evaluation_details": ml_results.get("evaluation_details"),
        "decision_threshold": ml_results.get("decision_threshold"),
        "threshold_selection": ml_results.get("threshold_selection"),
        "probability_calibration_training": ml_results.get(
            "probability_calibration_training"
        ),
        "feature_importance": ml_results.get("feature_importance", []),
        "feature_importance_diagnostics": ml_results.get(
            "feature_importance_diagnostics"
        ),
        "eda_visualizations": eda_results,
        "warnings": ml_results.get("warnings", []),
        "professional_grade": True,
        "strategy": approach,
        "run_manifest": run_manifest,
        "model_version": ml_results.get("model_version"),
        "model_artifact": ml_results.get("model_artifact"),
        "total_models_compared": len(models_trained),
    })


@app.post(
    "/api/training/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["training jobs"],
)
@app.post(
    "/api/jobs/training",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["training jobs"],
)
async def create_training_job(request: TrainingJobRequest):
    """Queue a reconnectable local training run and return its job resource."""
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    if request.target_column not in store.df.columns:
        raise HTTPException(400, f"Invalid target column: {request.target_column}")

    training_df = store.df.copy()
    project_id = store.project_id
    filename = store.filename
    fingerprint = (store.profile or {}).get("dataset_fingerprint")
    starting_model_version = (store.workflow_result or {}).get("model_version")
    approach = request.approach.value

    def runner(report_progress, is_cancelled):
        return _execute_training_job(
            training_df=training_df,
            target_column=request.target_column,
            approach=approach,
            include_eda=request.include_eda,
            training_project_id=project_id,
            source_filename=filename,
            dataset_fingerprint=fingerprint,
            report_progress=report_progress,
            is_cancelled=is_cancelled,
        )

    try:
        return training_jobs.submit(
            project_id,
            runner,
            metadata={
                "job_type": "training",
                "target_column": request.target_column,
                "approach": approach,
                "include_eda": request.include_eda,
                "dataset_fingerprint": fingerprint,
                "starting_model_version": starting_model_version,
            },
        )
    except ActiveTrainingJobError as exc:
        raise HTTPException(
            409,
            {
                "message": "This project already has an active training job.",
                "active_job_id": exc.active_job_id,
            },
        )
    except TrainingJobQueueFullError:
        raise HTTPException(429, "The local training queue is full. Try again shortly.")


def _training_job_or_404(job_id: str) -> dict:
    try:
        return training_jobs.get_job(job_id)
    except TrainingJobNotFoundError:
        raise HTTPException(404, "Training job not found.")


@app.get("/api/training/jobs", tags=["training jobs"])
async def list_training_jobs(limit: int = Query(default=20, ge=1, le=100)):
    project_id = store.project_id or None
    jobs = training_jobs.list_jobs(project_id=project_id, limit=100)
    jobs = [job for job in jobs if (job.get("metadata") or {}).get("job_type") == "training"]
    return {"jobs": jobs[:limit]}


@app.get("/api/jobs", tags=["jobs"])
async def list_all_jobs(limit: int = Query(default=20, ge=1, le=100)):
    project_id = store.project_id or None
    return {"jobs": training_jobs.list_jobs(project_id=project_id, limit=limit)}


@app.get("/api/training/jobs/{job_id}", tags=["training jobs"])
@app.get("/api/jobs/{job_id}", tags=["training jobs"])
async def get_training_job(job_id: str):
    return _training_job_or_404(job_id)


@app.post("/api/training/jobs/{job_id}/cancel", tags=["training jobs"])
@app.delete("/api/jobs/{job_id}", tags=["training jobs"])
async def cancel_training_job(job_id: str):
    _training_job_or_404(job_id)
    try:
        return training_jobs.request_cancel(job_id)
    except TrainingJobError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/ml/professional-workflow")
async def professional_ml_workflow(payload: dict):
    """
    🚀 PROFESSIONAL ML WORKFLOW - Train 10+ models with comprehensive analysis
    This is the ULTIMATE endpoint for serious ML model building!
    """
    if store.df is None:
        raise HTTPException(404, "No dataset uploaded yet.")
    
    target_column = payload.get("target_column")
    if not target_column or target_column not in store.df.columns:
        raise HTTPException(400, f"Invalid target column: {target_column}")

    training_project_id = store.project_id
    training_df = store.df.copy()
    approach = payload.get("approach", "balanced")
    if approach not in {"fast", "balanced", "high_performance"}:
        raise HTTPException(
            422,
            "approach must be one of: fast, balanced, high_performance",
        )
    strategy_config = {
        "fast": {
            "max_models": 5,
            "tune_best": False,
            "generate_eda": False,
        },
        "balanced": {
            "max_models": 8,
            "tune_best": False,
            "generate_eda": False,
        },
        "high_performance": {
            "max_models": 9,
            "tune_best": True,
            "generate_eda": False,
        },
    }[approach]
    
    try:
        # Step 1: Generate comprehensive EDA when the selected runtime budget allows it.
        target_series = training_df[target_column]
        task_type = (
            "classification"
            if (
                not pd.api.types.is_numeric_dtype(target_series)
                or target_series.nunique(dropna=True) <= 20
            )
            else "regression"
        )
        eda_results = (
            await run_in_threadpool(
                generate_comprehensive_eda,
                training_df,
                target_column,
                task_type,
            )
            if strategy_config["generate_eda"] or bool(payload.get("include_eda"))
            else []
        )
        
        # Step 2: Run enhanced ML workflow with multiple models
        ml_results = await run_in_threadpool(
            lambda: run_enhanced_workflow(
                training_df,
                target_column=target_column,
                task_type="auto",
                max_models=strategy_config["max_models"],
                tune_best=strategy_config["tune_best"],
            )
        )
        run_manifest = ml_results.setdefault("run_manifest", {})
        run_manifest["run_id"] = uuid.uuid4().hex
        run_manifest["project_id"] = training_project_id
        run_manifest["dataset_fingerprint"] = (
            (store.profile or {}).get("dataset_fingerprint")
        )
        run_manifest["source_filename"] = store.filename
        
        # Store pipelines
        pipeline = ml_results.pop("_pipeline", None)
        preprocessor = ml_results.pop("_preprocessor", None)
        models_trained = ml_results.get("models_trained") or []
        best_score = ml_results.get("best_score")
        if (
            pipeline is None
            or not models_trained
            or not ml_results.get("best_model")
            or best_score is None
            or not np.isfinite(float(best_score))
        ):
            details = "; ".join(ml_results.get("warnings") or [])
            raise HTTPException(
                422,
                "No model could be trained for this dataset."
                + (f" {details}" if details else ""),
            )
        with STORE_LOCK:
            if store.project_id != training_project_id:
                raise HTTPException(
                    409,
                    "The active project changed while training. Run the experiment again.",
                )
            registry_entry = _register_model_version(
                project_id=training_project_id,
                pipeline=pipeline,
                workflow_result=ml_results,
                approach=approach,
            )
            ml_results["model_artifact"] = {
                "version_id": registry_entry["version_id"],
                "sha256": registry_entry["sha256"],
                "size_bytes": registry_entry["size_bytes"],
                "registered_at": registry_entry["registered_at"],
            }
            store.workflow_result = ml_results.copy()
            store.workflow_result["_pipeline"] = pipeline
            store.workflow_result["_preprocessor"] = preprocessor
            store.workflow_result["strategy"] = approach
            store.report_bundle = None
        await run_in_threadpool(_persist_workspace)
        
        # Combine results
        professional_result = {
            "success": True,
            "task_type": ml_results.get("task_type"),
            "data_info": ml_results.get("data_shape"),
            "cleaning_summary": ml_results.get("cleaning"),
            "feature_summary": ml_results.get("features"),
            "split_info": ml_results.get("split"),
            "data_balancing": ml_results.get("data_balancing"),
            "class_distribution": ml_results.get("class_distribution"),
            "models_trained": ml_results.get("models_trained", []),
            "best_model": ml_results.get("best_model"),
            "best_score": ml_results.get("best_score"),
            "target_column": target_column,
            "positive_label": ml_results.get("positive_label"),
            "model_selection": ml_results.get("model_selection"),
            "evaluation_details": ml_results.get("evaluation_details"),
            "decision_threshold": ml_results.get("decision_threshold"),
            "threshold_selection": ml_results.get("threshold_selection"),
            "probability_calibration_training": ml_results.get(
                "probability_calibration_training"
            ),
            "feature_importance": ml_results.get("feature_importance", []),
            "feature_importance_diagnostics": ml_results.get(
                "feature_importance_diagnostics"
            ),
            "eda_visualizations": eda_results,
            "warnings": ml_results.get("warnings", []),
            "professional_grade": True,
            "strategy": approach,
            "run_manifest": ml_results.get("run_manifest"),
            "model_version": ml_results.get("model_version"),
            "model_artifact": ml_results.get("model_artifact"),
            "total_models_compared": len(ml_results.get("models_trained", [])),
        }
        
        # Make JSON-safe and return
        safe_result = make_json_safe(professional_result)
        json.dumps(safe_result)  # Validate
        return safe_result
        
    except HTTPException:
        raise
    except json.JSONDecodeError:
        raise HTTPException(500, "Result contains non-JSON-serializable data")
    except Exception as exc:
        artifact_logger.exception("Professional workflow failed")
        raise HTTPException(500, "Professional workflow failed. Review the run warnings and dataset.")


# ── Static files (must be last) ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
