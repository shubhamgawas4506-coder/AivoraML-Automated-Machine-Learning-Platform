"""Immutable, local-first model version registry.

Artifacts are stored below ``storage/model_versions/<project>/<version>.joblib``
with one atomic JSON manifest per project.  The registry is intentionally
framework-independent so API routes and background jobs can share it safely.

Joblib/pickle files must only be loaded from this application-owned registry.
Every read verifies the recorded SHA-256 checksum before deserialization.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import math
import os
import re
import threading
import uuid
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib


SCHEMA_VERSION = 1
PROJECT_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")
VERSION_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
registry_logger = logging.getLogger("aivoraml.model_registry")


class ModelRegistryError(RuntimeError):
    """Base exception for registry operations."""


class InvalidModelIdentifierError(ModelRegistryError):
    """Raised when a project or version identifier is unsafe."""


class DuplicateModelVersionError(ModelRegistryError):
    """Raised when an immutable version ID has already been used."""


class ModelVersionNotFoundError(ModelRegistryError):
    """Raised when a project or retained model version cannot be found."""


class ModelArtifactError(ModelRegistryError):
    """Raised when an artifact is missing, invalid, or cannot be written."""


class ModelArtifactIntegrityError(ModelArtifactError):
    """Raised when artifact bytes do not match their recorded checksum."""


class ModelManifestError(ModelRegistryError):
    """Raised when a project manifest is corrupt or structurally invalid."""


class ModelMetadataTooLargeError(ModelRegistryError):
    """Raised when public metadata exceeds the configured manifest budget."""


_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}


def _shared_root_lock(root: Path) -> threading.RLock:
    """Share one in-process lock across registry instances for the same root."""

    key = os.path.normcase(str(root))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _json_safe(
    value: Any,
    *,
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
) -> Any:
    """Convert public metadata to a bounded JSON-safe representation."""

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
    tracked = isinstance(value, (Mapping, list, tuple, set, frozenset)) or (
        is_dataclass(value) and not isinstance(value, type)
    )
    if tracked:
        identity = id(value)
        if identity in seen:
            return "<circular reference>"
        seen.add(identity)

    try:
        # NumPy-compatible conversion without making NumPy a registry dependency.
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
                    output["metadata_truncated"] = True
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
        if tracked:
            seen.discard(id(value))

    # Avoid repr(), which may contain secrets, arbitrary user text, or addresses.
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


class ModelRegistry:
    """Store and retrieve immutable, checksum-verified model versions.

    Args:
        root: Registry directory. Defaults to
            ``<project>/storage/model_versions``.
        retention_limit: Maximum live versions retained per project. ``None``
            keeps all versions. Pruned identifiers are tombstoned so they can
            never be silently reused or overwritten.
        max_metadata_bytes: Maximum UTF-8 JSON size of public metadata per run.
    """

    def __init__(
        self,
        root: Optional[Path | str] = None,
        *,
        retention_limit: Optional[int] = 20,
        max_metadata_bytes: int = 256 * 1024,
    ) -> None:
        if retention_limit is not None and retention_limit < 1:
            raise ValueError("retention_limit must be at least 1 or None.")
        if max_metadata_bytes < 1_024:
            raise ValueError("max_metadata_bytes must be at least 1024 bytes.")

        default_root = (
            Path(__file__).resolve().parents[2] / "storage" / "model_versions"
        )
        self.root = Path(root or default_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retention_limit = retention_limit
        self.max_metadata_bytes = max_metadata_bytes
        self._lock = _shared_root_lock(self.root)

    def register(
        self,
        project_id: str,
        version_id: str,
        pipeline: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Atomically register one completed fitted-pipeline artifact.

        Version IDs are immutable. A duplicate is rejected even when its live
        artifact was removed by the configured retention policy.
        """

        project = self.validate_project_id(project_id)
        version = self.validate_version_id(version_id)
        self._validate_pipeline(pipeline)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")
        public_metadata = _json_safe(dict(metadata or {}))
        metadata_json = json.dumps(
            public_metadata,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(metadata_json) > self.max_metadata_bytes:
            raise ModelMetadataTooLargeError(
                f"Public metadata is {len(metadata_json):,} bytes; the limit is "
                f"{self.max_metadata_bytes:,} bytes."
            )

        with self._lock:
            project_dir = self._project_directory(project, create=True)
            manifest = self._read_manifest_locked(project, allow_missing=True)
            live_versions = manifest["versions"]
            retired_versions = manifest["retired_versions"]
            used_ids = {
                item["version_id"]
                for item in [*live_versions, *retired_versions]
            }
            artifact_path = self._artifact_path(project, version)
            if version in used_ids or artifact_path.exists():
                raise DuplicateModelVersionError(
                    f"Model version '{version}' already exists for project "
                    f"'{project}'. Version IDs are immutable."
                )

            registered_at = _utcnow_iso()
            artifact = {
                "schema_version": SCHEMA_VERSION,
                "project_id": project,
                "version_id": version,
                "registered_at": registered_at,
                "pipeline": pipeline,
                "public_metadata": public_metadata,
            }
            temporary_path = project_dir / f".{version}.{uuid.uuid4().hex}.tmp"
            artifact_committed = False
            manifest_committed = False
            try:
                self._dump_artifact(temporary_path, artifact)
                checksum = self._sha256_file(temporary_path)
                size_bytes = temporary_path.stat().st_size
                if artifact_path.exists():
                    raise DuplicateModelVersionError(
                        f"Model version '{version}' already exists for project "
                        f"'{project}'."
                    )
                os.replace(temporary_path, artifact_path)
                artifact_committed = True

                entry = {
                    "schema_version": SCHEMA_VERSION,
                    "project_id": project,
                    "version_id": version,
                    "registered_at": registered_at,
                    "artifact_filename": artifact_path.name,
                    "size_bytes": size_bytes,
                    "sha256": checksum,
                    "metadata": public_metadata,
                }
                retained = [entry, *live_versions]
                pruned: list[dict[str, Any]] = []
                if (
                    self.retention_limit is not None
                    and len(retained) > self.retention_limit
                ):
                    pruned = retained[self.retention_limit :]
                    retained = retained[: self.retention_limit]
                retired = [*retired_versions]
                retired_at = _utcnow_iso()
                retired.extend(
                    {
                        "version_id": item["version_id"],
                        "registered_at": item["registered_at"],
                        "retired_at": retired_at,
                        "sha256": item["sha256"],
                    }
                    for item in pruned
                )
                updated_manifest = {
                    "schema_version": SCHEMA_VERSION,
                    "project_id": project,
                    "updated_at": registered_at,
                    "versions": retained,
                    "retired_versions": retired,
                }
                self._write_manifest_locked(project, updated_manifest)
                manifest_committed = True
            except Exception:
                if temporary_path.exists():
                    temporary_path.unlink()
                if artifact_committed and not manifest_committed:
                    artifact_path.unlink(missing_ok=True)
                raise

            # The manifest is authoritative. Retention cleanup happens only
            # after its atomic replacement, so a crash cannot leave references
            # to missing artifacts.
            for retired_entry in pruned:
                retired_path = self._artifact_path(
                    project,
                    retired_entry["version_id"],
                )
                try:
                    retired_path.unlink(missing_ok=True)
                except OSError:
                    # Registration is already durably committed. Keep the
                    # tombstone authoritative and surface orphan cleanup through
                    # logs instead of reporting a false registration failure.
                    registry_logger.warning(
                        "Could not remove retired model artifact %s",
                        retired_path,
                        exc_info=True,
                    )
            return self._public_copy(entry)

    register_completed_run = register

    def list_versions(
        self,
        project_id: str,
        *,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return retained model metadata newest-first."""

        project = self.validate_project_id(project_id)
        if limit is not None and limit < 0:
            raise ValueError("limit cannot be negative.")
        with self._lock:
            manifest = self._read_manifest_locked(project, allow_missing=True)
            versions = manifest["versions"]
            selected = versions if limit is None else versions[:limit]
            return [self._public_copy(entry) for entry in selected]

    def get_metadata(
        self,
        project_id: str,
        version_id: str,
    ) -> dict[str, Any]:
        """Return public metadata for one retained immutable version."""

        project = self.validate_project_id(project_id)
        version = self.validate_version_id(version_id)
        with self._lock:
            entry = self._find_entry_locked(project, version)
            return self._public_copy(entry)

    get_version = get_metadata

    def verify_artifact(
        self,
        project_id: str,
        version_id: str,
    ) -> dict[str, Any]:
        """Verify file size and SHA-256, returning the version metadata."""

        project = self.validate_project_id(project_id)
        version = self.validate_version_id(version_id)
        with self._lock:
            entry = self._find_entry_locked(project, version)
            self._verified_artifact_bytes_locked(project, version, entry)
            return self._public_copy(entry)

    def resolve_artifact(
        self,
        project_id: str,
        version_id: str,
    ) -> Path:
        """Resolve a retained artifact path after checksum verification."""

        project = self.validate_project_id(project_id)
        version = self.validate_version_id(version_id)
        with self._lock:
            entry = self._find_entry_locked(project, version)
            self._verified_artifact_bytes_locked(project, version, entry)
            return self._artifact_path(project, version)

    def read_artifact_bytes(
        self,
        project_id: str,
        version_id: str,
    ) -> bytes:
        """Read checksum-verified bytes suitable for an HTTP download response."""

        project = self.validate_project_id(project_id)
        version = self.validate_version_id(version_id)
        with self._lock:
            entry = self._find_entry_locked(project, version)
            return self._verified_artifact_bytes_locked(project, version, entry)

    def load_artifact(
        self,
        project_id: str,
        version_id: str,
    ) -> dict[str, Any]:
        """Verify, deserialize, and validate an application-owned artifact."""

        project = self.validate_project_id(project_id)
        version = self.validate_version_id(version_id)
        with self._lock:
            entry = self._find_entry_locked(project, version)
            artifact_bytes = self._verified_artifact_bytes_locked(
                project,
                version,
                entry,
            )
            try:
                artifact = joblib.load(io.BytesIO(artifact_bytes))
            except Exception as exc:
                raise ModelArtifactError(
                    f"Model version '{version}' could not be deserialized."
                ) from exc
            if not isinstance(artifact, dict):
                raise ModelArtifactError("Registered artifact has an invalid shape.")
            if (
                artifact.get("schema_version") != SCHEMA_VERSION
                or artifact.get("project_id") != project
                or artifact.get("version_id") != version
                or artifact.get("public_metadata") != entry["metadata"]
            ):
                raise ModelArtifactIntegrityError(
                    "Artifact identity or metadata does not match its manifest."
                )
            self._validate_pipeline(artifact.get("pipeline"))
            return artifact

    def load_pipeline(self, project_id: str, version_id: str) -> Any:
        """Return the checksum-verified fitted pipeline for a version."""

        return self.load_artifact(project_id, version_id)["pipeline"]

    @staticmethod
    def validate_project_id(project_id: str) -> str:
        """Validate and canonicalize a 12-character hexadecimal project ID."""

        candidate = str(project_id)
        if candidate != candidate.strip():
            raise InvalidModelIdentifierError(
                "project_id cannot contain leading or trailing whitespace."
            )
        canonical = candidate.lower()
        if PROJECT_ID_PATTERN.fullmatch(canonical) is None:
            raise InvalidModelIdentifierError(
                "project_id must contain exactly 12 hexadecimal characters."
            )
        return canonical

    @staticmethod
    def validate_version_id(version_id: str) -> str:
        """Validate and canonicalize a portable, traversal-safe version ID."""

        candidate = str(version_id)
        if candidate != candidate.strip():
            raise InvalidModelIdentifierError(
                "version_id cannot contain leading or trailing whitespace."
            )
        canonical = candidate.lower()
        if VERSION_ID_PATTERN.fullmatch(canonical) is None:
            raise InvalidModelIdentifierError(
                "version_id must be 1-64 letters, digits, dots, underscores, or "
                "hyphens, beginning with a letter or digit."
            )
        if canonical in {".", ".."}:
            raise InvalidModelIdentifierError("version_id cannot be a dot path.")
        first_component = canonical.split(".", 1)[0]
        if first_component in WINDOWS_RESERVED_NAMES:
            raise InvalidModelIdentifierError(
                f"version_id '{candidate}' is reserved by the operating system."
            )
        return canonical

    def _project_directory(self, project: str, *, create: bool) -> Path:
        candidate = self.root / project
        if create:
            candidate.mkdir(parents=True, exist_ok=True)
        resolved = candidate.resolve(strict=False)
        self._ensure_contained(resolved)
        if candidate.is_symlink():
            raise ModelRegistryError(
                "Symbolic-link project registry directories are not supported."
            )
        return resolved

    def _artifact_path(self, project: str, version: str) -> Path:
        project_dir = self._project_directory(project, create=False)
        candidate = project_dir / f"{version}.joblib"
        resolved = candidate.resolve(strict=False)
        self._ensure_contained(resolved)
        if candidate.exists() and candidate.is_symlink():
            raise ModelArtifactError("Symbolic-link artifacts are not supported.")
        return resolved

    def _manifest_path(self, project: str) -> Path:
        project_dir = self._project_directory(project, create=False)
        candidate = project_dir / "manifest.json"
        resolved = candidate.resolve(strict=False)
        self._ensure_contained(resolved)
        if candidate.exists() and candidate.is_symlink():
            raise ModelManifestError("Symbolic-link manifests are not supported.")
        return resolved

    def _ensure_contained(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ModelRegistryError(
                "Resolved registry path escapes the configured storage root."
            ) from exc

    @staticmethod
    def _validate_pipeline(pipeline: Any) -> None:
        if pipeline is None or not callable(getattr(pipeline, "predict", None)):
            raise ModelArtifactError(
                "A trained pipeline exposing predict() is required."
            )
        # Reject known unfitted scikit-learn estimators while permitting other
        # prediction protocols that joblib can serialize.
        try:
            from sklearn.exceptions import NotFittedError
            from sklearn.utils.validation import check_is_fitted

            try:
                check_is_fitted(pipeline)
            except NotFittedError as exc:
                raise ModelArtifactError("The supplied pipeline is not fitted.") from exc
            except TypeError:
                # A non-sklearn predictor may not implement estimator tags.
                pass
        except ImportError:
            pass

    @staticmethod
    def _dump_artifact(path: Path, artifact: dict[str, Any]) -> None:
        try:
            with path.open("xb") as handle:
                joblib.dump(artifact, handle, compress=3)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:
            path.unlink(missing_ok=True)
            if isinstance(exc, ModelRegistryError):
                raise
            raise ModelArtifactError("The model artifact could not be saved.") from exc

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _empty_manifest(self, project: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "project_id": project,
            "updated_at": None,
            "versions": [],
            "retired_versions": [],
        }

    def _read_manifest_locked(
        self,
        project: str,
        *,
        allow_missing: bool,
    ) -> dict[str, Any]:
        path = self._manifest_path(project)
        if not path.exists():
            if allow_missing:
                return self._empty_manifest(project)
            raise ModelVersionNotFoundError(
                f"No model versions exist for project '{project}'."
            )
        try:
            manifest = json.loads(
                path.read_text(encoding="utf-8"),
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelManifestError(
                f"The model manifest for project '{project}' is unreadable."
            ) from exc
        self._validate_manifest(manifest, project)
        return manifest

    def _validate_manifest(self, manifest: Any, project: str) -> None:
        if not isinstance(manifest, dict):
            raise ModelManifestError("Model manifest must be a JSON object.")
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("project_id") != project
            or not isinstance(manifest.get("versions"), list)
            or not isinstance(manifest.get("retired_versions"), list)
        ):
            raise ModelManifestError(
                "Model manifest schema or project identity is invalid."
            )
        seen: set[str] = set()
        previous_time: Optional[str] = None
        for entry in manifest["versions"]:
            if not isinstance(entry, dict):
                raise ModelManifestError("Manifest version entry must be an object.")
            try:
                version = self.validate_version_id(entry["version_id"])
            except (KeyError, InvalidModelIdentifierError) as exc:
                raise ModelManifestError("Manifest contains an invalid version ID.") from exc
            expected_filename = f"{version}.joblib"
            if (
                entry.get("schema_version") != SCHEMA_VERSION
                or entry.get("project_id") != project
                or entry.get("artifact_filename") != expected_filename
                or not isinstance(entry.get("registered_at"), str)
                or not isinstance(entry.get("size_bytes"), int)
                or entry["size_bytes"] < 1
                or SHA256_PATTERN.fullmatch(str(entry.get("sha256", ""))) is None
                or not isinstance(entry.get("metadata"), dict)
                or version in seen
            ):
                raise ModelManifestError("Manifest contains an invalid version entry.")
            # Entries are persisted newest-first; lexical ISO-8601 ordering is
            # stable because all registry timestamps use the same UTC format.
            if previous_time is not None and entry["registered_at"] > previous_time:
                raise ModelManifestError("Manifest versions are not newest-first.")
            previous_time = entry["registered_at"]
            seen.add(version)
        for retired in manifest["retired_versions"]:
            if not isinstance(retired, dict):
                raise ModelManifestError("Retired version entry must be an object.")
            try:
                version = self.validate_version_id(retired["version_id"])
            except (KeyError, InvalidModelIdentifierError) as exc:
                raise ModelManifestError(
                    "Manifest contains an invalid retired version ID."
                ) from exc
            if (
                version in seen
                or not isinstance(retired.get("registered_at"), str)
                or not isinstance(retired.get("retired_at"), str)
                or SHA256_PATTERN.fullmatch(str(retired.get("sha256", ""))) is None
            ):
                raise ModelManifestError(
                    "Manifest contains an invalid retired version entry."
                )
            seen.add(version)

    def _write_manifest_locked(
        self,
        project: str,
        manifest: dict[str, Any],
    ) -> None:
        self._validate_manifest(manifest, project)
        path = self._manifest_path(project)
        temporary = path.parent / f".manifest.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    manifest,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if isinstance(exc, ModelRegistryError):
                raise
            raise ModelManifestError("Model manifest could not be saved.") from exc

    def _find_entry_locked(self, project: str, version: str) -> dict[str, Any]:
        manifest = self._read_manifest_locked(project, allow_missing=False)
        for entry in manifest["versions"]:
            if entry["version_id"] == version:
                return entry
        if any(
            item["version_id"] == version
            for item in manifest["retired_versions"]
        ):
            raise ModelVersionNotFoundError(
                f"Model version '{version}' was retired by the retention policy."
            )
        raise ModelVersionNotFoundError(
            f"Model version '{version}' does not exist for project '{project}'."
        )

    def _verified_artifact_bytes_locked(
        self,
        project: str,
        version: str,
        entry: Mapping[str, Any],
    ) -> bytes:
        path = self._artifact_path(project, version)
        if not path.is_file():
            raise ModelArtifactIntegrityError(
                f"Artifact for model version '{version}' is missing."
            )
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ModelArtifactError(
                f"Artifact for model version '{version}' could not be read."
            ) from exc
        actual_checksum = hashlib.sha256(data).hexdigest()
        expected_checksum = str(entry["sha256"])
        if len(data) != entry["size_bytes"] or not hmac.compare_digest(
            actual_checksum,
            expected_checksum,
        ):
            raise ModelArtifactIntegrityError(
                f"Checksum verification failed for model version '{version}'."
            )
        return data

    @staticmethod
    def _public_copy(entry: Mapping[str, Any]) -> dict[str, Any]:
        # A JSON round-trip provides a deep defensive copy and verifies the
        # public metadata contract at the service boundary.
        return json.loads(json.dumps(entry, ensure_ascii=False, allow_nan=False))


__all__ = [
    "DuplicateModelVersionError",
    "InvalidModelIdentifierError",
    "ModelArtifactError",
    "ModelArtifactIntegrityError",
    "ModelManifestError",
    "ModelMetadataTooLargeError",
    "ModelRegistry",
    "ModelRegistryError",
    "ModelVersionNotFoundError",
    "SCHEMA_VERSION",
]
