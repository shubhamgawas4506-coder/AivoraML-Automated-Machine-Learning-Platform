"""Thread-safe background training job orchestration.

The manager in this module deliberately has no FastAPI or machine-learning
dependencies.  API routes can therefore submit any callable that accepts a
progress reporter and a cancellation predicate, while keeping job lifecycle
rules in one small, testable component.
"""

from __future__ import annotations

import math
import os
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TypeAlias


class TrainingJobState(str, Enum):
    """States in the lifecycle of a background training job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {
        TrainingJobState.SUCCEEDED,
        TrainingJobState.FAILED,
        TrainingJobState.CANCELLED,
    }
)
ACTIVE_STATES = frozenset(
    {
        TrainingJobState.QUEUED,
        TrainingJobState.RUNNING,
        TrainingJobState.CANCEL_REQUESTED,
    }
)

ProgressCallback: TypeAlias = Callable[[float, Optional[str], Optional[str]], None]
CancellationPredicate: TypeAlias = Callable[[], bool]
TrainingRunner: TypeAlias = Callable[[ProgressCallback, CancellationPredicate], Any]


class TrainingJobError(RuntimeError):
    """Base exception for manager-level job errors."""


class TrainingJobNotFoundError(TrainingJobError):
    """Raised when a requested job ID is unknown or has expired from history."""


class ActiveTrainingJobError(TrainingJobError):
    """Raised when a project already has a queued or running training job."""

    def __init__(self, project_id: str, active_job_id: str) -> None:
        self.project_id = project_id
        self.active_job_id = active_job_id
        super().__init__(
            f"Project '{project_id}' already has active training job "
            f"'{active_job_id}'."
        )


class TrainingJobQueueFullError(TrainingJobError):
    """Raised when the manager's bounded outstanding-job capacity is full."""


class TrainingJobManagerShutdownError(TrainingJobError):
    """Raised when submission is attempted after manager shutdown."""


class TrainingJobCancelled(TrainingJobError):
    """Optional cooperative exception a runner may raise after cancellation."""


@dataclass(slots=True)
class _TrainingJob:
    """Mutable internal record; every mutation except Event.set holds the lock."""

    job_id: str
    project_id: str
    state: TrainingJobState
    progress: float
    stage: str
    message: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Any = None
    error: Optional[dict[str, str]] = None
    cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )
    future: Optional[Future[Any]] = field(default=None, repr=False)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(value: Optional[str], *, limit: int) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip()[:limit]


def _json_safe(
    value: Any,
    *,
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
) -> Any:
    """Return a bounded, JSON-serializable representation of ``value``.

    Private mapping keys and callables are intentionally omitted so public job
    snapshots do not accidentally expose fitted pipelines or helper functions.
    """

    if _depth > 30:
        return "<maximum depth reached>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _json_safe(value.value, _seen=_seen, _depth=_depth + 1)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if callable(value):
        return None

    seen = _seen if _seen is not None else set()
    track_identity = isinstance(value, (Mapping, list, tuple, set, frozenset)) or (
        is_dataclass(value) and not isinstance(value, type)
    )
    if track_identity:
        identity = id(value)
        if identity in seen:
            return "<circular reference>"
        seen.add(identity)

    try:
        # NumPy scalars and arrays expose item()/tolist() without requiring NumPy.
        if hasattr(value, "item") and callable(value.item):
            try:
                return _json_safe(value.item(), _seen=seen, _depth=_depth + 1)
            except (TypeError, ValueError):
                pass
        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return _json_safe(value.tolist(), _seen=seen, _depth=_depth + 1)
            except (TypeError, ValueError):
                pass
        if isinstance(value, Mapping):
            output: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 10_000:
                    output["_truncated"] = True
                    break
                key_text = str(key)
                if key_text.startswith("_") or callable(item):
                    continue
                output[key_text] = _json_safe(
                    item,
                    _seen=seen,
                    _depth=_depth + 1,
                )
            return output
        if isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
            output = [
                _json_safe(item, _seen=seen, _depth=_depth + 1)
                for item in items[:10_000]
                if not callable(item)
            ]
            if len(items) > 10_000:
                output.append("<truncated>")
            return output
        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: _json_safe(
                    getattr(value, item.name),
                    _seen=seen,
                    _depth=_depth + 1,
                )
                for item in fields(value)
                if not item.name.startswith("_")
            }
    finally:
        if track_identity:
            seen.discard(id(value))

    # Unknown library objects are represented by their type, not repr(), which
    # can contain secrets or memory addresses and is often not deterministic.
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


class TrainingJobManager:
    """Manage bounded, cooperative background training jobs.

    A runner is called as ``runner(report_progress, is_cancelled)``. Progress is
    a percentage from 0 through 100; the manager keeps it monotonic and reserves
    100 for successful completion. Cancellation is cooperative once a job has
    started, so runners should check ``is_cancelled()`` between expensive model
    fits and either return promptly or raise :class:`TrainingJobCancelled`.

    Only one active job is permitted per project. Finished records are retained
    newest-first up to ``max_history``; their raw results remain accessible via
    :meth:`get_result` until the record is evicted.
    """

    DEFAULT_MAX_WORKERS = max(1, min(2, os.cpu_count() or 1))
    MAX_WORKERS_LIMIT = 4

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_history: int = 100,
        max_outstanding_jobs: int = 32,
        thread_name_prefix: str = "aivoraml-training",
    ) -> None:
        if not 1 <= max_workers <= self.MAX_WORKERS_LIMIT:
            raise ValueError(
                f"max_workers must be between 1 and {self.MAX_WORKERS_LIMIT}."
            )
        if max_history < 1:
            raise ValueError("max_history must be at least 1.")
        if max_outstanding_jobs < max_workers:
            raise ValueError(
                "max_outstanding_jobs must be at least as large as max_workers."
            )

        self._max_history = max_history
        self._max_outstanding_jobs = max_outstanding_jobs
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, _TrainingJob] = OrderedDict()
        self._active_by_project: dict[str, str] = {}
        self._shutdown = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )

    def submit(
        self,
        project_id: str,
        runner: TrainingRunner,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        message: str = "Waiting for an available training worker.",
    ) -> dict[str, Any]:
        """Create and queue a job, returning its public snapshot.

        Raises :class:`ActiveTrainingJobError` if the project already owns an
        active job and :class:`TrainingJobQueueFullError` when the bounded
        outstanding-job capacity has been reached.
        """

        normalized_project_id = str(project_id).strip()
        if not normalized_project_id:
            raise ValueError("project_id must be a non-empty string.")
        if not callable(runner):
            raise TypeError("runner must be callable.")

        with self._lock:
            if self._shutdown:
                raise TrainingJobManagerShutdownError(
                    "The training job manager has been shut down."
                )
            active_job_id = self._active_by_project.get(normalized_project_id)
            if active_job_id is not None:
                raise ActiveTrainingJobError(
                    normalized_project_id,
                    active_job_id,
                )
            outstanding = sum(
                job.state in ACTIVE_STATES for job in self._jobs.values()
            )
            if outstanding >= self._max_outstanding_jobs:
                raise TrainingJobQueueFullError(
                    "The training queue is full. Try again after an active job "
                    "finishes."
                )

            now = _utcnow()
            job = _TrainingJob(
                job_id=uuid.uuid4().hex,
                project_id=normalized_project_id,
                state=TrainingJobState.QUEUED,
                progress=0.0,
                stage="queued",
                message=_clean_text(message, limit=500) or "Queued.",
                created_at=now,
                updated_at=now,
                metadata=dict(metadata or {}),
            )
            self._jobs[job.job_id] = job
            self._active_by_project[job.project_id] = job.job_id

            try:
                future = self._executor.submit(self._execute, job.job_id, runner)
            except Exception:
                self._jobs.pop(job.job_id, None)
                self._active_by_project.pop(job.project_id, None)
                raise
            job.future = future
            future.add_done_callback(
                lambda completed, job_id=job.job_id: self._on_future_done(
                    job_id,
                    completed,
                )
            )
            return self._snapshot_locked(job)

    # A name that reads naturally from route handlers while retaining submit's
    # familiar executor vocabulary.
    create_job = submit

    def report_progress(
        self,
        job_id: str,
        progress: float,
        stage: Optional[str] = None,
        message: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update an active job's monotonic progress and return a snapshot."""

        if isinstance(progress, bool):
            raise TypeError("progress must be a number, not bool.")
        numeric_progress = float(progress)
        if not math.isfinite(numeric_progress):
            raise ValueError("progress must be finite.")
        numeric_progress = max(0.0, min(99.9, numeric_progress))

        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state in TERMINAL_STATES:
                return self._snapshot_locked(job)
            job.progress = max(job.progress, numeric_progress)
            clean_stage = _clean_text(stage, limit=120)
            clean_message = _clean_text(message, limit=500)
            if clean_stage:
                job.stage = clean_stage
            if clean_message:
                job.message = clean_message
            job.updated_at = _utcnow()
            return self._snapshot_locked(job)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        """Request cooperative cancellation and return the updated snapshot."""

        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state in TERMINAL_STATES:
                return self._snapshot_locked(job)

            now = _utcnow()
            job.cancel_event.set()
            job.state = TrainingJobState.CANCEL_REQUESTED
            job.stage = "cancelling"
            job.message = "Cancellation requested; finishing the current step."
            job.updated_at = now

            # A queued future can be cancelled immediately. A running future
            # returns False and observes cancel_event cooperatively instead.
            if job.future is not None:
                job.future.cancel()
            return self._snapshot_locked(job)

    cancel = request_cancel

    def is_cancel_requested(self, job_id: str) -> bool:
        """Return whether cancellation has been requested for ``job_id``."""

        with self._lock:
            return self._require_job_locked(job_id).cancel_event.is_set()

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return a JSON-safe public snapshot for one job."""

        with self._lock:
            return self._snapshot_locked(self._require_job_locked(job_id))

    def get_active_job(self, project_id: str) -> Optional[dict[str, Any]]:
        """Return the project's active job, or ``None`` when it has none."""

        with self._lock:
            job_id = self._active_by_project.get(str(project_id).strip())
            if job_id is None:
                return None
            job = self._jobs.get(job_id)
            return self._snapshot_locked(job) if job is not None else None

    def list_jobs(
        self,
        *,
        project_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """List public snapshots newest-first, optionally filtered by project."""

        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative.")
        normalized_project = (
            str(project_id).strip() if project_id is not None else None
        )
        with self._lock:
            jobs = reversed(self._jobs.values())
            snapshots = [
                self._snapshot_locked(job)
                for job in jobs
                if normalized_project is None
                or job.project_id == normalized_project
            ]
            return snapshots if limit is None else snapshots[:limit]

    def get_result(self, job_id: str) -> Any:
        """Return a succeeded job's raw result for server-side integration."""

        with self._lock:
            job = self._require_job_locked(job_id)
            if job.state is not TrainingJobState.SUCCEEDED:
                raise TrainingJobError(
                    f"Job '{job_id}' has no successful result "
                    f"(state: {job.state.value})."
                )
            return job.result

    def wait(
        self,
        job_id: str,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Wait for a job to settle, returning its snapshot without re-raising."""

        with self._lock:
            job = self._require_job_locked(job_id)
            future = job.future
        if future is not None:
            try:
                future.result(timeout=timeout)
            except CancelledError:
                pass
            except Exception:
                # Failure information is normalized into the job snapshot.
                pass
        return self.get_job(job_id)

    def shutdown(
        self,
        *,
        wait: bool = True,
        cancel_futures: bool = False,
    ) -> None:
        """Stop accepting work and release executor threads.

        With ``cancel_futures=True``, all active records first receive a
        cooperative cancellation signal and futures that have not started are
        cancelled by the executor.
        """

        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            if cancel_futures:
                for job in self._jobs.values():
                    if job.state in ACTIVE_STATES:
                        job.cancel_event.set()
                        job.state = TrainingJobState.CANCEL_REQUESTED
                        job.stage = "cancelling"
                        job.message = "Cancellation requested during shutdown."
                        job.updated_at = _utcnow()
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> "TrainingJobManager":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.shutdown(wait=True, cancel_futures=True)

    def _execute(self, job_id: str, runner: TrainingRunner) -> Any:
        with self._lock:
            job = self._require_job_locked(job_id)
            if job.cancel_event.is_set():
                self._mark_cancelled_locked(job)
                return None
            now = _utcnow()
            job.state = TrainingJobState.RUNNING
            job.stage = "starting"
            job.message = "Training started."
            job.started_at = now
            job.updated_at = now

        def progress_callback(
            progress: float,
            stage: Optional[str] = None,
            message: Optional[str] = None,
        ) -> None:
            self.report_progress(job_id, progress, stage, message)

        def cancellation_predicate() -> bool:
            return job.cancel_event.is_set()

        try:
            result = runner(progress_callback, cancellation_predicate)
        except TrainingJobCancelled:
            with self._lock:
                self._mark_cancelled_locked(job)
            return None
        except Exception as exc:
            with self._lock:
                if job.cancel_event.is_set():
                    self._mark_cancelled_locked(job)
                    return None
                now = _utcnow()
                job.state = TrainingJobState.FAILED
                job.stage = "failed"
                job.message = "Training failed."
                job.error = {
                    "type": type(exc).__name__[:200],
                    "message": str(exc)[:2_000] or "Training failed.",
                }
                job.finished_at = now
                job.updated_at = now
            raise

        with self._lock:
            if job.cancel_event.is_set():
                self._mark_cancelled_locked(job)
                return None
            now = _utcnow()
            job.result = result
            job.error = None
            job.progress = 100.0
            job.state = TrainingJobState.SUCCEEDED
            job.stage = "completed"
            job.message = "Training completed successfully."
            job.finished_at = now
            job.updated_at = now
        return result

    def _on_future_done(self, job_id: str, future: Future[Any]) -> None:
        """Normalize executor completion and release the project's active slot."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if future.cancelled() and job.state not in TERMINAL_STATES:
                self._mark_cancelled_locked(job)
            elif job.state not in TERMINAL_STATES:
                # Defensive fallback for exceptional BaseException paths not
                # handled by _execute's normal Exception branch.
                try:
                    unexpected = future.exception()
                except CancelledError:
                    unexpected = None
                if unexpected is not None:
                    now = _utcnow()
                    job.state = TrainingJobState.FAILED
                    job.stage = "failed"
                    job.message = "Training terminated unexpectedly."
                    job.error = {
                        "type": type(unexpected).__name__[:200],
                        "message": str(unexpected)[:2_000]
                        or "Training terminated unexpectedly.",
                    }
                    job.finished_at = now
                    job.updated_at = now

            if self._active_by_project.get(job.project_id) == job_id:
                self._active_by_project.pop(job.project_id, None)
            self._prune_history_locked()

    def _mark_cancelled_locked(self, job: _TrainingJob) -> None:
        now = _utcnow()
        job.cancel_event.set()
        job.state = TrainingJobState.CANCELLED
        job.stage = "cancelled"
        job.message = "Training was cancelled."
        job.result = None
        job.error = None
        job.finished_at = now
        job.updated_at = now

    def _prune_history_locked(self) -> None:
        terminal_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.state in TERMINAL_STATES
        ]
        excess = len(terminal_ids) - self._max_history
        for job_id in terminal_ids[: max(0, excess)]:
            self._jobs.pop(job_id, None)

    def _require_job_locked(self, job_id: str) -> _TrainingJob:
        job = self._jobs.get(str(job_id))
        if job is None:
            raise TrainingJobNotFoundError(f"Unknown training job '{job_id}'.")
        return job

    def _snapshot_locked(self, job: _TrainingJob) -> dict[str, Any]:
        snapshot = {
            "job_id": job.job_id,
            "project_id": job.project_id,
            "state": job.state.value,
            "progress": round(job.progress, 2),
            "stage": job.stage,
            "message": job.message,
            "metadata": _json_safe(job.metadata),
            "created_at": job.created_at.isoformat(),
            "started_at": (
                job.started_at.isoformat() if job.started_at is not None else None
            ),
            "finished_at": (
                job.finished_at.isoformat() if job.finished_at is not None else None
            ),
            "updated_at": job.updated_at.isoformat(),
            "cancellation_requested": job.cancel_event.is_set(),
            "result": (
                _json_safe(job.result)
                if job.state is TrainingJobState.SUCCEEDED
                else None
            ),
            "error": _json_safe(job.error),
        }
        # Keep this invariant local: callers can hand snapshots directly to a
        # JSON response without relying on a framework-specific encoder.
        return snapshot


__all__ = [
    "ACTIVE_STATES",
    "TERMINAL_STATES",
    "ActiveTrainingJobError",
    "CancellationPredicate",
    "ProgressCallback",
    "TrainingJobCancelled",
    "TrainingJobError",
    "TrainingJobManager",
    "TrainingJobManagerShutdownError",
    "TrainingJobNotFoundError",
    "TrainingJobQueueFullError",
    "TrainingJobState",
    "TrainingRunner",
]
