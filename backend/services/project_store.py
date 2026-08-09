"""Durable, local-first storage for isolated AivoraML projects.

The repository owns every path below ``root`` and deliberately exposes no
method that accepts an arbitrary destination.  Project snapshots are written
as immutable generations, checksummed, and then referenced by an atomically
replaced manifest and index.  This keeps normal readers on either the previous
complete generation or the next complete generation.

``joblib`` can execute code while loading.  Consequently, normal loads are
restricted to strictly named files in repository-created project directories,
and the already-open file handle is checksum-verified before it is passed to
``joblib.load``.  Importing the former single-workspace snapshot is a separate,
explicit operation with its own path restrictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib


PROJECT_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")
SNAPSHOT_FILE_PATTERN = re.compile(r"^snapshot-[0-9a-f]{12}\.joblib$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

INDEX_VERSION = 1
MANIFEST_VERSION = 1
INDEX_FILENAME = "index.json"
MANIFEST_FILENAME = "manifest.json"

PUBLIC_METADATA_FIELDS = (
    "id",
    "filename",
    "created_at",
    "updated_at",
    "rows",
    "cols",
    "trained",
    "prediction_count",
    "report_ready",
)


class ProjectStoreError(RuntimeError):
    """Base class for repository errors."""


class InvalidProjectIdError(ProjectStoreError, ValueError):
    """Raised when an ID is not exactly twelve lowercase hexadecimal chars."""


class ProjectNotFoundError(ProjectStoreError, KeyError):
    """Raised when a project is not present in this repository."""


class ProjectAlreadyExistsError(ProjectStoreError):
    """Raised instead of overwriting an existing project."""


class ProjectCapacityError(ProjectStoreError):
    """Raised when the configured repository project bound is reached."""


class SnapshotTooLargeError(ProjectStoreError):
    """Raised when a serialized snapshot exceeds its configured size bound."""


class SnapshotChecksumError(ProjectStoreError):
    """Raised before load when a snapshot checksum does not match its manifest."""


class ProjectStoreCorruptionError(ProjectStoreError):
    """Raised when repository-owned metadata is invalid or inconsistent."""


class UnsafeProjectPathError(ProjectStoreError):
    """Raised when a candidate path escapes its exact repository boundary."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class ProjectRepository:
    """Thread-safe repository for a bounded number of local projects.

    Parameters
    ----------
    root:
        The project repository directory, normally ``storage/projects``.
    max_projects:
        Hard bound that prevents an accidental unbounded project collection.
    max_snapshot_bytes:
        Hard bound checked immediately after serialization and before commit.
    max_delete_files:
        Maximum number of immediate files a project deletion may remove.
        Deletion is never recursive.
    legacy_roots:
        Exact parent directories from which explicit legacy imports are
        allowed.  The default is the repository root's parent (``storage``).
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_projects: int = 100,
        max_snapshot_bytes: int = 512 * 1024 * 1024,
        max_delete_files: int = 256,
        legacy_roots: Optional[list[str | os.PathLike[str]]] = None,
    ) -> None:
        if max_projects < 1:
            raise ValueError("max_projects must be at least 1")
        if max_snapshot_bytes < 1:
            raise ValueError("max_snapshot_bytes must be at least 1")
        if max_delete_files < 2:
            raise ValueError("max_delete_files must be at least 2")

        requested_root = Path(root).expanduser()
        requested_root.mkdir(parents=True, exist_ok=True)
        self.root = requested_root.resolve(strict=True)
        if not self.root.is_dir():
            raise UnsafeProjectPathError(
                f"Project repository root is not a directory: {self.root}"
            )

        self.max_projects = int(max_projects)
        self.max_snapshot_bytes = int(max_snapshot_bytes)
        self.max_delete_files = int(max_delete_files)
        self._lock = threading.RLock()
        self._index_path = self.root / INDEX_FILENAME

        configured_legacy_roots = legacy_roots or [self.root.parent]
        self._legacy_roots = tuple(
            Path(candidate).expanduser().resolve(strict=False)
            for candidate in configured_legacy_roots
        )
        self._index = self._load_or_recover_index()

    @staticmethod
    def validate_project_id(project_id: str) -> str:
        """Return a valid ID or fail before any filesystem access occurs."""
        if not isinstance(project_id, str) or not PROJECT_ID_PATTERN.fullmatch(
            project_id
        ):
            raise InvalidProjectIdError(
                "Project IDs must be exactly 12 lowercase hexadecimal characters."
            )
        return project_id

    @property
    def active_project_id(self) -> Optional[str]:
        with self._lock:
            value = self._index.get("active_project_id")
            return str(value) if value is not None else None

    def create_project(
        self,
        snapshot: Any,
        *,
        project_id: Optional[str] = None,
        filename: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        activate: bool = True,
    ) -> dict[str, Any]:
        """Create and persist a new project without ever overwriting one."""
        with self._lock:
            projects = self._index["projects"]
            if len(projects) >= self.max_projects:
                raise ProjectCapacityError(
                    f"Project capacity ({self.max_projects}) has been reached."
                )

            if project_id is None:
                project_id = self._new_project_id()
            else:
                project_id = self.validate_project_id(project_id)

            project_dir = self._project_dir(project_id)
            if project_id in projects or project_dir.exists():
                raise ProjectAlreadyExistsError(
                    f"Project '{project_id}' already exists."
                )

            project_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
            project_dir = self._verify_exact_project_directory(
                project_id, must_exist=True
            )
            manifest: Optional[dict[str, Any]] = None
            try:
                public_metadata = self._derive_metadata(
                    project_id,
                    snapshot,
                    existing=None,
                    filename=filename,
                    overrides=metadata,
                )
                normalized_snapshot = self._normalize_snapshot(
                    snapshot, public_metadata
                )
                manifest = self._write_snapshot_and_manifest(
                    project_dir,
                    project_id,
                    normalized_snapshot,
                    public_metadata,
                )

                next_index = self._copy_index()
                next_index["projects"][project_id] = self._entry_from_manifest(
                    manifest
                )
                if activate:
                    next_index["active_project_id"] = project_id
                self._commit_index(next_index)
            except Exception:
                # Creation owns this brand-new exact directory.  Remove only
                # its known immediate files; never recurse or touch siblings.
                self._cleanup_failed_creation(project_dir)
                raise

            return dict(manifest["metadata"])

    def save_project(
        self,
        project_id: str,
        snapshot: Any,
        *,
        filename: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        activate: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Atomically replace one project's snapshot generation."""
        with self._lock:
            project_id = self.validate_project_id(project_id)
            entry = self._require_entry(project_id)
            project_dir = self._verify_exact_project_directory(
                project_id, must_exist=True
            )
            previous_manifest = self._read_and_validate_manifest(
                project_id, entry
            )
            public_metadata = self._derive_metadata(
                project_id,
                snapshot,
                existing=previous_manifest["metadata"],
                filename=filename,
                overrides=metadata,
            )
            normalized_snapshot = self._normalize_snapshot(
                snapshot, public_metadata
            )

            new_manifest = self._write_snapshot_and_manifest(
                project_dir,
                project_id,
                normalized_snapshot,
                public_metadata,
            )
            new_snapshot_path = project_dir / new_manifest["snapshot_file"]
            try:
                next_index = self._copy_index()
                next_index["projects"][project_id] = self._entry_from_manifest(
                    new_manifest
                )
                if activate is True:
                    next_index["active_project_id"] = project_id
                elif (
                    activate is False
                    and next_index.get("active_project_id") == project_id
                ):
                    next_index["active_project_id"] = None
                self._commit_index(next_index)
            except Exception:
                # Restore the previous manifest so it continues pointing at
                # the untouched previous immutable snapshot.
                self._atomic_write_json(
                    project_dir / MANIFEST_FILENAME,
                    previous_manifest,
                    expected_parent=project_dir,
                )
                self._unlink_owned_file(new_snapshot_path, project_dir)
                raise

            old_name = previous_manifest["snapshot_file"]
            if old_name != new_manifest["snapshot_file"]:
                self._unlink_owned_file(project_dir / old_name, project_dir)
            return dict(new_manifest["metadata"])

    def list_projects(self) -> list[dict[str, Any]]:
        """Return public metadata only, newest update first."""
        with self._lock:
            projects = [
                dict(entry["metadata"])
                for entry in self._index["projects"].values()
            ]
            projects.sort(
                key=lambda item: (item["updated_at"], item["id"]),
                reverse=True,
            )
            return projects

    def load_project(
        self,
        project_id: Optional[str] = None,
        *,
        activate: bool = True,
    ) -> Any:
        """Checksum-verify and load an app-owned project snapshot."""
        with self._lock:
            if project_id is None:
                project_id = self._index.get("active_project_id")
                if project_id is None:
                    raise ProjectNotFoundError("No active project is set.")
            project_id = self.validate_project_id(project_id)
            entry = self._require_entry(project_id)
            manifest = self._read_and_validate_manifest(project_id, entry)
            project_dir = self._verify_exact_project_directory(
                project_id, must_exist=True
            )
            snapshot_path = self._verified_snapshot_path(
                project_dir, manifest["snapshot_file"]
            )

            expected_size = int(manifest["snapshot_bytes"])
            actual_size = snapshot_path.stat().st_size
            if actual_size != expected_size:
                raise SnapshotChecksumError(
                    f"Snapshot size mismatch for project '{project_id}'."
                )
            if actual_size > self.max_snapshot_bytes:
                raise SnapshotTooLargeError(
                    f"Snapshot for project '{project_id}' exceeds the configured bound."
                )

            # Hash and load the same already-open descriptor.  This avoids a
            # path replacement between checksum validation and joblib.load.
            with snapshot_path.open("rb") as snapshot_file:
                actual_checksum = self._sha256_handle(snapshot_file)
                if actual_checksum != manifest["snapshot_sha256"]:
                    raise SnapshotChecksumError(
                        f"Snapshot checksum mismatch for project '{project_id}'."
                    )
                snapshot_file.seek(0)
                result = joblib.load(snapshot_file)

            if activate and self._index.get("active_project_id") != project_id:
                next_index = self._copy_index()
                next_index["active_project_id"] = project_id
                self._commit_index(next_index)
            return result

    def activate_project(self, project_id: Optional[str]) -> Optional[str]:
        """Persist the active-project pointer without loading the snapshot."""
        with self._lock:
            if project_id is not None:
                project_id = self.validate_project_id(project_id)
                entry = self._require_entry(project_id)
                self._read_and_validate_manifest(project_id, entry)
            next_index = self._copy_index()
            next_index["active_project_id"] = project_id
            self._commit_index(next_index)
            return project_id

    def delete_project(self, project_id: str) -> dict[str, Any]:
        """Delete one exact project directory after a non-recursive preflight.

        The return value always identifies each removed file and any failure.
        A preflight safety failure removes nothing.  The project is removed
        from the index only after its exact directory has been emptied and
        removed successfully.
        """
        with self._lock:
            project_id = self.validate_project_id(project_id)
            self._require_entry(project_id)
            project_dir = self._verify_exact_project_directory(
                project_id, must_exist=True
            )
            result: dict[str, Any] = {
                "project_id": project_id,
                "deleted": False,
                "deleted_files": [],
                "failures": [],
                "active_project_id": self._index.get("active_project_id"),
            }

            try:
                children = list(project_dir.iterdir())
            except OSError as exc:
                result["failures"].append(
                    {"path": str(project_dir), "error": str(exc)}
                )
                return result

            if len(children) > self.max_delete_files:
                result["failures"].append(
                    {
                        "path": str(project_dir),
                        "error": (
                            "Deletion refused: project contains more than "
                            f"{self.max_delete_files} immediate files."
                        ),
                    }
                )
                return result

            # Validate the complete deletion set first so a malicious nested
            # directory or symlink cannot cause a partially deleted project.
            safe_files: list[Path] = []
            for child in children:
                try:
                    if child.is_symlink():
                        raise UnsafeProjectPathError(
                            "Deletion refused for symbolic links."
                        )
                    resolved = child.resolve(strict=True)
                    if resolved.parent != project_dir or not resolved.is_file():
                        raise UnsafeProjectPathError(
                            "Deletion is limited to immediate regular files."
                        )
                    safe_files.append(resolved)
                except (OSError, UnsafeProjectPathError) as exc:
                    result["failures"].append(
                        {"path": str(child), "error": str(exc)}
                    )

            if result["failures"]:
                return result

            # Remove snapshots first and the manifest last.  No recursive
            # command is used, and every path came through the preflight.
            safe_files.sort(key=lambda path: path.name == MANIFEST_FILENAME)
            for owned_file in safe_files:
                try:
                    owned_file.unlink()
                    result["deleted_files"].append(owned_file.name)
                except OSError as exc:
                    result["failures"].append(
                        {"path": str(owned_file), "error": str(exc)}
                    )

            if result["failures"]:
                return result

            try:
                project_dir.rmdir()
            except OSError as exc:
                result["failures"].append(
                    {"path": str(project_dir), "error": str(exc)}
                )
                return result

            next_index = self._copy_index()
            del next_index["projects"][project_id]
            if next_index.get("active_project_id") == project_id:
                next_index["active_project_id"] = None
            self._commit_index(next_index)
            result["deleted"] = True
            result["active_project_id"] = next_index["active_project_id"]
            return result

    def import_legacy_snapshot(
        self,
        legacy_path: str | os.PathLike[str],
        *,
        expected_sha256: Optional[str] = None,
        project_id: Optional[str] = None,
        filename: Optional[str] = None,
        activate: bool = True,
    ) -> dict[str, Any]:
        """Explicitly import the former app-owned ``workspace.joblib``.

        Unlike normal loads this method accepts a source outside ``root``, but
        only when it is a direct regular ``.joblib`` child of a configured
        legacy root.  The same open descriptor is checksummed before loading.
        Pass ``expected_sha256`` when a trusted migration checksum is known.
        """
        with self._lock:
            source = Path(legacy_path).expanduser()
            if source.is_symlink():
                raise UnsafeProjectPathError("Legacy snapshot cannot be a symlink.")
            source = source.resolve(strict=True)
            if not source.is_file() or source.suffix.lower() != ".joblib":
                raise UnsafeProjectPathError(
                    "Legacy snapshot must be a regular .joblib file."
                )
            if source.parent not in self._legacy_roots or _is_relative_to(
                source, self.root
            ):
                raise UnsafeProjectPathError(
                    "Legacy snapshot is outside the configured legacy roots."
                )
            if expected_sha256 is not None:
                expected_sha256 = str(expected_sha256).lower()
                if not SHA256_PATTERN.fullmatch(expected_sha256):
                    raise SnapshotChecksumError(
                        "Expected legacy checksum must be a SHA-256 hex digest."
                    )

            size = source.stat().st_size
            if size > self.max_snapshot_bytes:
                raise SnapshotTooLargeError(
                    "Legacy snapshot exceeds the configured snapshot bound."
                )
            with source.open("rb") as legacy_file:
                checksum = self._sha256_handle(legacy_file)
                if expected_sha256 is not None and checksum != expected_sha256:
                    raise SnapshotChecksumError(
                        "Legacy snapshot checksum does not match the expected digest."
                    )
                legacy_file.seek(0)
                snapshot = joblib.load(legacy_file)

            if not isinstance(snapshot, Mapping):
                raise ProjectStoreCorruptionError(
                    "Legacy workspace snapshot must contain a mapping."
                )
            if snapshot.get("version") not in (None, 1):
                raise ProjectStoreCorruptionError(
                    "Unsupported legacy workspace snapshot version."
                )

            selected_id = project_id
            if selected_id is None:
                legacy_id = snapshot.get("project_id")
                if (
                    isinstance(legacy_id, str)
                    and PROJECT_ID_PATTERN.fullmatch(legacy_id)
                    and legacy_id not in self._index["projects"]
                    and not self._project_dir(legacy_id).exists()
                ):
                    selected_id = legacy_id

            return self.create_project(
                snapshot,
                project_id=selected_id,
                filename=filename,
                activate=activate,
            )

    # ------------------------------------------------------------------
    # Repository metadata and path validation

    def _new_project_id(self) -> str:
        for _ in range(128):
            candidate = uuid.uuid4().hex[:12]
            if (
                candidate not in self._index["projects"]
                and not self._project_dir(candidate).exists()
            ):
                return candidate
        raise ProjectAlreadyExistsError(
            "Could not allocate a unique project ID after 128 attempts."
        )

    def _project_dir(self, project_id: str) -> Path:
        project_id = self.validate_project_id(project_id)
        candidate = self.root / project_id
        # IDs cannot contain separators, but retain the lexical check as a
        # second boundary should the validation ever be changed.
        if candidate.parent != self.root:
            raise UnsafeProjectPathError("Project path escapes repository root.")
        return candidate

    def _verify_exact_project_directory(
        self, project_id: str, *, must_exist: bool
    ) -> Path:
        candidate = self._project_dir(project_id)
        if not candidate.exists():
            if must_exist:
                raise ProjectNotFoundError(f"Project '{project_id}' does not exist.")
            return candidate
        if candidate.is_symlink():
            raise UnsafeProjectPathError("Project directories cannot be symlinks.")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != self.root or resolved.name != project_id:
            raise UnsafeProjectPathError("Resolved project path escapes repository root.")
        if not resolved.is_dir():
            raise UnsafeProjectPathError("Project path is not a directory.")
        return resolved

    def _verified_snapshot_path(self, project_dir: Path, filename: str) -> Path:
        if not isinstance(filename, str) or not SNAPSHOT_FILE_PATTERN.fullmatch(
            filename
        ):
            raise ProjectStoreCorruptionError("Manifest snapshot filename is invalid.")
        candidate = project_dir / filename
        if candidate.is_symlink():
            raise UnsafeProjectPathError("Snapshot files cannot be symlinks.")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != project_dir or not resolved.is_file():
            raise UnsafeProjectPathError("Snapshot path escapes its project directory.")
        return resolved

    def _require_entry(self, project_id: str) -> dict[str, Any]:
        entry = self._index["projects"].get(project_id)
        if entry is None:
            raise ProjectNotFoundError(f"Project '{project_id}' does not exist.")
        return entry

    def _copy_index(self) -> dict[str, Any]:
        # The index is JSON-only, so this also provides a cheap defensive deep
        # copy without sharing nested metadata dictionaries.
        return json.loads(json.dumps(self._index))

    def _commit_index(self, next_index: dict[str, Any]) -> None:
        self._validate_index_shape(next_index)
        self._atomic_write_json(
            self._index_path, next_index, expected_parent=self.root
        )
        self._index = next_index

    def _load_or_recover_index(self) -> dict[str, Any]:
        with self._lock:
            if self._index_path.exists() or self._index_path.is_symlink():
                if self._index_path.is_symlink():
                    raise UnsafeProjectPathError("Repository index cannot be a symlink.")
                loaded = self._read_json(
                    self._index_path, expected_parent=self.root
                )
                self._validate_index_shape(loaded)
                active_project_id = loaded.get("active_project_id")
            else:
                loaded = {
                    "version": INDEX_VERSION,
                    "active_project_id": None,
                    "projects": {},
                }
                active_project_id = None

            # Manifests are authoritative for project entries.  Reconciliation
            # heals an interrupted deletion or a manifest committed just before
            # an index replacement, without loading any joblib data.
            recovered_projects: dict[str, Any] = {}
            for child in self.root.iterdir():
                if child.name == INDEX_FILENAME or child.name.startswith("."):
                    continue
                if not PROJECT_ID_PATTERN.fullmatch(child.name):
                    continue
                project_dir = self._verify_exact_project_directory(
                    child.name, must_exist=True
                )
                manifest_path = project_dir / MANIFEST_FILENAME
                if not manifest_path.exists():
                    raise ProjectStoreCorruptionError(
                        f"Project '{child.name}' has no manifest."
                    )
                manifest = self._read_and_validate_manifest(child.name, None)
                recovered_projects[child.name] = self._entry_from_manifest(manifest)

            if len(recovered_projects) > self.max_projects:
                raise ProjectCapacityError(
                    "Existing repository exceeds the configured project capacity."
                )
            if active_project_id not in recovered_projects:
                active_project_id = None
            recovered = {
                "version": INDEX_VERSION,
                "active_project_id": active_project_id,
                "projects": recovered_projects,
            }
            if recovered != loaded or not self._index_path.exists():
                self._atomic_write_json(
                    self._index_path, recovered, expected_parent=self.root
                )
            return recovered

    @staticmethod
    def _validate_index_shape(index: Any) -> None:
        if not isinstance(index, dict) or index.get("version") != INDEX_VERSION:
            raise ProjectStoreCorruptionError("Unsupported repository index.")
        projects = index.get("projects")
        if not isinstance(projects, dict):
            raise ProjectStoreCorruptionError("Repository index has no project map.")
        active = index.get("active_project_id")
        if active is not None:
            ProjectRepository.validate_project_id(active)
        for project_id, entry in projects.items():
            ProjectRepository.validate_project_id(project_id)
            if not isinstance(entry, dict):
                raise ProjectStoreCorruptionError("Repository project entry is invalid.")
            metadata = entry.get("metadata")
            ProjectRepository._validate_public_metadata(metadata, project_id)
            snapshot_file = entry.get("snapshot_file")
            checksum = entry.get("snapshot_sha256")
            size = entry.get("snapshot_bytes")
            if not isinstance(snapshot_file, str) or not SNAPSHOT_FILE_PATTERN.fullmatch(
                snapshot_file
            ):
                raise ProjectStoreCorruptionError("Index snapshot filename is invalid.")
            if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
                raise ProjectStoreCorruptionError("Index snapshot checksum is invalid.")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ProjectStoreCorruptionError("Index snapshot size is invalid.")

    @staticmethod
    def _entry_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "metadata": dict(manifest["metadata"]),
            "snapshot_file": manifest["snapshot_file"],
            "snapshot_sha256": manifest["snapshot_sha256"],
            "snapshot_bytes": manifest["snapshot_bytes"],
        }

    # ------------------------------------------------------------------
    # Snapshot and manifest operations

    def _write_snapshot_and_manifest(
        self,
        project_dir: Path,
        project_id: str,
        snapshot: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        generation = uuid.uuid4().hex[:12]
        snapshot_name = f"snapshot-{generation}.joblib"
        snapshot_path = project_dir / snapshot_name
        checksum, snapshot_bytes = self._atomic_dump_joblib(
            snapshot_path, snapshot, expected_parent=project_dir
        )
        manifest = {
            "version": MANIFEST_VERSION,
            "project_id": project_id,
            "metadata": dict(metadata),
            "snapshot_file": snapshot_name,
            "snapshot_sha256": checksum,
            "snapshot_bytes": snapshot_bytes,
        }
        try:
            self._atomic_write_json(
                project_dir / MANIFEST_FILENAME,
                manifest,
                expected_parent=project_dir,
            )
        except Exception:
            self._unlink_owned_file(snapshot_path, project_dir)
            raise
        return manifest

    def _read_and_validate_manifest(
        self,
        project_id: str,
        index_entry: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        project_dir = self._verify_exact_project_directory(
            project_id, must_exist=True
        )
        manifest = self._read_json(
            project_dir / MANIFEST_FILENAME, expected_parent=project_dir
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != MANIFEST_VERSION
            or manifest.get("project_id") != project_id
        ):
            raise ProjectStoreCorruptionError(
                f"Manifest for project '{project_id}' is invalid."
            )
        self._validate_public_metadata(manifest.get("metadata"), project_id)
        snapshot_file = manifest.get("snapshot_file")
        checksum = manifest.get("snapshot_sha256")
        size = manifest.get("snapshot_bytes")
        if not isinstance(snapshot_file, str) or not SNAPSHOT_FILE_PATTERN.fullmatch(
            snapshot_file
        ):
            raise ProjectStoreCorruptionError("Manifest snapshot filename is invalid.")
        if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
            raise ProjectStoreCorruptionError("Manifest checksum is invalid.")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ProjectStoreCorruptionError("Manifest snapshot size is invalid.")
        if index_entry is not None and self._entry_from_manifest(manifest) != dict(
            index_entry
        ):
            raise ProjectStoreCorruptionError(
                f"Manifest and index disagree for project '{project_id}'."
            )
        return manifest

    def _atomic_dump_joblib(
        self,
        destination: Path,
        snapshot: Any,
        *,
        expected_parent: Path,
    ) -> tuple[str, int]:
        self._validate_write_destination(destination, expected_parent)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=".snapshot-", suffix=".tmp", dir=str(expected_parent)
        )
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            dumped_files = joblib.dump(snapshot, temp_path, compress=3)
            if len(dumped_files) != 1 or Path(dumped_files[0]) != temp_path:
                raise ProjectStoreError(
                    "Snapshot serialization unexpectedly produced multiple files."
                )
            # Open read/write for the durability flush because Windows does
            # not permit ``fsync`` on a read-only descriptor.
            with temp_path.open("r+b") as serialized:
                checksum = self._sha256_handle(serialized)
                snapshot_bytes = serialized.tell()
                os.fsync(serialized.fileno())
            if snapshot_bytes > self.max_snapshot_bytes:
                raise SnapshotTooLargeError(
                    f"Serialized snapshot exceeds {self.max_snapshot_bytes} bytes."
                )
            os.replace(temp_path, destination)
            return checksum, snapshot_bytes
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _atomic_write_json(
        self,
        destination: Path,
        value: Mapping[str, Any],
        *,
        expected_parent: Path,
    ) -> None:
        self._validate_write_destination(destination, expected_parent)
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(expected_parent),
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as temp:
                json.dump(
                    value,
                    temp,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                temp.write("\n")
                temp.flush()
                os.fsync(temp.fileno())
            os.replace(temp_path, destination)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _sha256_handle(file_handle: Any) -> str:
        digest = hashlib.sha256()
        while True:
            block = file_handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _validate_write_destination(destination: Path, expected_parent: Path) -> None:
        if destination.parent != expected_parent:
            raise UnsafeProjectPathError("Write destination escapes its owner directory.")
        if destination.exists() and destination.is_symlink():
            raise UnsafeProjectPathError("Repository files cannot be symlinks.")
        resolved_parent = expected_parent.resolve(strict=True)
        if resolved_parent != expected_parent:
            raise UnsafeProjectPathError("Write parent changed during validation.")

    def _read_json(self, path: Path, *, expected_parent: Path) -> dict[str, Any]:
        if path.parent != expected_parent or path.is_symlink():
            raise UnsafeProjectPathError("JSON metadata path escapes its owner.")
        resolved = path.resolve(strict=True)
        if resolved.parent != expected_parent or not resolved.is_file():
            raise UnsafeProjectPathError("JSON metadata path escapes its owner.")
        if resolved.stat().st_size > 4 * 1024 * 1024:
            raise ProjectStoreCorruptionError("JSON metadata exceeds the safety bound.")
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProjectStoreCorruptionError(
                f"Could not read repository metadata '{resolved.name}'."
            ) from exc
        if not isinstance(value, dict):
            raise ProjectStoreCorruptionError("Repository metadata must be an object.")
        return value

    # ------------------------------------------------------------------
    # Public metadata derivation and validation

    def _derive_metadata(
        self,
        project_id: str,
        snapshot: Any,
        *,
        existing: Optional[Mapping[str, Any]],
        filename: Optional[str],
        overrides: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        snapshot_map = snapshot if isinstance(snapshot, Mapping) else {}
        now = _utc_now()
        previous = dict(existing or {})

        inferred_rows = previous.get("rows", 0)
        inferred_cols = previous.get("cols", 0)
        dataframe = snapshot_map.get("df")
        shape = getattr(dataframe, "shape", None)
        if (
            isinstance(shape, tuple)
            and len(shape) >= 2
            and all(isinstance(value, int) for value in shape[:2])
        ):
            inferred_rows, inferred_cols = int(shape[0]), int(shape[1])

        workflow = snapshot_map.get("workflow_result")
        trained = bool(previous.get("trained", False))
        if isinstance(workflow, Mapping):
            trained = workflow.get("_pipeline") is not None

        predictions = snapshot_map.get("predictions")
        prediction_count = previous.get("prediction_count", 0)
        if isinstance(predictions, (list, tuple)):
            prediction_count = len(predictions)

        report_bundle = snapshot_map.get("report_bundle")
        report_ready = bool(previous.get("report_ready", False))
        if isinstance(report_bundle, Mapping):
            report_ready = report_bundle.get("status") == "completed"

        selected_filename = (
            filename
            if filename is not None
            else snapshot_map.get("filename", previous.get("filename", "dataset.csv"))
        )
        metadata: dict[str, Any] = {
            "id": project_id,
            "filename": self._safe_display_filename(selected_filename),
            "created_at": previous.get("created_at")
            or self._safe_timestamp(snapshot_map.get("created_at"), now),
            "updated_at": now,
            "rows": inferred_rows,
            "cols": inferred_cols,
            "trained": trained,
            "prediction_count": prediction_count,
            "report_ready": report_ready,
        }

        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise TypeError("metadata overrides must be a mapping")
            allowed = set(PUBLIC_METADATA_FIELDS) - {
                "id",
                "created_at",
                "updated_at",
            }
            unknown = set(overrides) - allowed
            if unknown:
                raise ValueError(
                    "Unsupported public metadata fields: "
                    + ", ".join(sorted(str(item) for item in unknown))
                )
            for key, value in overrides.items():
                metadata[key] = value
            metadata["filename"] = self._safe_display_filename(
                metadata["filename"]
            )

        self._validate_public_metadata(metadata, project_id)
        return metadata

    @staticmethod
    def _normalize_snapshot(
        snapshot: Any, metadata: Mapping[str, Any]
    ) -> Any:
        if not isinstance(snapshot, Mapping):
            return snapshot
        normalized = dict(snapshot)
        normalized["project_id"] = metadata["id"]
        normalized["filename"] = metadata["filename"]
        normalized["created_at"] = metadata["created_at"]
        normalized["updated_at"] = metadata["updated_at"]
        return normalized

    @staticmethod
    def _safe_display_filename(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("filename must be a string")
        if "\x00" in value:
            raise ValueError("filename contains a NUL byte")
        filename = Path(value).name.strip()
        if not filename or filename in {".", ".."}:
            raise ValueError("filename must have a non-empty basename")
        if len(filename) > 255:
            raise ValueError("filename exceeds 255 characters")
        return filename

    @staticmethod
    def _safe_timestamp(value: Any, fallback: str) -> str:
        if not isinstance(value, str) or not value:
            return fallback
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        if parsed.tzinfo is None:
            return fallback
        return parsed.isoformat()

    @staticmethod
    def _validate_public_metadata(metadata: Any, project_id: str) -> None:
        if not isinstance(metadata, dict):
            raise ProjectStoreCorruptionError("Project metadata is invalid.")
        if set(metadata) != set(PUBLIC_METADATA_FIELDS):
            raise ProjectStoreCorruptionError(
                "Project metadata does not match the public schema."
            )
        if metadata.get("id") != project_id:
            raise ProjectStoreCorruptionError("Project metadata ID mismatch.")
        if not isinstance(metadata.get("filename"), str) or not metadata["filename"]:
            raise ProjectStoreCorruptionError("Project filename is invalid.")
        for timestamp_field in ("created_at", "updated_at"):
            timestamp = metadata.get(timestamp_field)
            if not isinstance(timestamp, str):
                raise ProjectStoreCorruptionError("Project timestamp is invalid.")
            try:
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ProjectStoreCorruptionError(
                    "Project timestamp is invalid."
                ) from exc
            if parsed.tzinfo is None:
                raise ProjectStoreCorruptionError(
                    "Project timestamps must include a timezone."
                )
        for count_field in ("rows", "cols", "prediction_count"):
            count = metadata.get(count_field)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ProjectStoreCorruptionError(
                    f"Project metadata field '{count_field}' is invalid."
                )
        for flag_field in ("trained", "report_ready"):
            if not isinstance(metadata.get(flag_field), bool):
                raise ProjectStoreCorruptionError(
                    f"Project metadata field '{flag_field}' is invalid."
                )

    # ------------------------------------------------------------------
    # Conservative cleanup helpers

    @staticmethod
    def _unlink_owned_file(path: Path, expected_parent: Path) -> None:
        try:
            if path.parent != expected_parent or path.is_symlink():
                return
            resolved = path.resolve(strict=True)
            if resolved.parent == expected_parent and resolved.is_file():
                resolved.unlink()
        except OSError:
            pass

    def _cleanup_failed_creation(self, project_dir: Path) -> None:
        try:
            if project_dir.parent != self.root or project_dir.is_symlink():
                return
            resolved = project_dir.resolve(strict=True)
            if resolved.parent != self.root:
                return
            children = list(resolved.iterdir())
            if len(children) > self.max_delete_files:
                return
            for child in children:
                if child.is_symlink():
                    return
                child_resolved = child.resolve(strict=True)
                if child_resolved.parent != resolved or not child_resolved.is_file():
                    return
            for child in children:
                child.unlink()
            resolved.rmdir()
        except OSError:
            pass


# A concise alias for callers that prefer the module's filename terminology.
ProjectStore = ProjectRepository


__all__ = [
    "InvalidProjectIdError",
    "ProjectAlreadyExistsError",
    "ProjectCapacityError",
    "ProjectNotFoundError",
    "ProjectRepository",
    "ProjectStore",
    "ProjectStoreCorruptionError",
    "ProjectStoreError",
    "SnapshotChecksumError",
    "SnapshotTooLargeError",
    "UnsafeProjectPathError",
]
