"""Filesystem sandbox containing every agent-visible file."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import os
from pathlib import Path
import tempfile
from typing import Final

from .config import DEFAULT_ALLOWED_EXTENSIONS, Settings
from .security import SecurityPolicy


WORLD_AREAS: Final[tuple[str, ...]] = (
    "workspace",
    "drawings",
    "projects",
    "experiments",
    "diary",
    "skills",
)
_WINDOWS_RESERVED_NAMES: Final[frozenset[str]] = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)


class WorldSandbox:
    """Constrained UTF-8 text storage rooted at ``world_data``.

    The root is selected by trusted application code. Every path supplied by an
    agent or tool must be relative and is resolved before access, which also
    catches symlinks whose destination leaves the root.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        settings: Settings | None = None,
        security_policy: SecurityPolicy | None = None,
        allowed_extensions: Sequence[str] | None = None,
        maximum_file_size_bytes: int | None = None,
        maximum_total_world_size_bytes: int | None = None,
    ) -> None:
        if allowed_extensions is not None:
            chosen_extensions = allowed_extensions
        elif settings is not None:
            chosen_extensions = settings.allowed_extensions
        else:
            chosen_extensions = DEFAULT_ALLOWED_EXTENSIONS
        if maximum_file_size_bytes is not None:
            chosen_file_limit = maximum_file_size_bytes
        elif settings is not None:
            chosen_file_limit = settings.maximum_file_size_bytes
        else:
            chosen_file_limit = 100_000
        if maximum_total_world_size_bytes is not None:
            chosen_total_limit = maximum_total_world_size_bytes
        elif settings is not None:
            chosen_total_limit = settings.maximum_total_world_size_bytes
        else:
            chosen_total_limit = 10_000_000

        self._root = Path(root).expanduser().resolve(strict=False)
        self._allowed_extensions = frozenset(chosen_extensions)
        self._maximum_file_size_bytes = chosen_file_limit
        self._maximum_total_world_size_bytes = chosen_total_limit
        self._security = security_policy or SecurityPolicy()
        self._validate_constructor_values()
        self._root.mkdir(parents=True, exist_ok=True)
        if not self._root.is_dir():
            raise ValueError(f"world root is not a directory: {self._root}")
        self.ensure_areas()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def areas(self) -> tuple[str, ...]:
        return WORLD_AREAS

    @property
    def allowed_extensions(self) -> frozenset[str]:
        return self._allowed_extensions

    @property
    def maximum_file_size_bytes(self) -> int:
        return self._maximum_file_size_bytes

    @property
    def maximum_total_world_size_bytes(self) -> int:
        return self._maximum_total_world_size_bytes

    @property
    def total_size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self._iter_safe_files())

    def ensure_areas(self) -> None:
        """Create the fixed top-level areas used by tools."""

        for area in WORLD_AREAS:
            destination = (self._root / area).resolve(strict=False)
            if not self._is_inside(destination):
                self._security.deny_path(
                    "create_world_area", area, "world area resolves outside world_data"
                )
            destination.mkdir(parents=True, exist_ok=True)
            if not destination.is_dir():
                self._security.deny_path(
                    "create_world_area", area, "world area is not a directory"
                )

    def resolve_path(
        self,
        relative_path: str | os.PathLike[str],
        *,
        operation: str = "world_access",
        require_allowed_extension: bool = True,
    ) -> Path:
        """Resolve an untrusted relative path and enforce the world boundary."""

        try:
            raw_path = os.fspath(relative_path)
        except TypeError:
            self._security.deny_path(
                operation, relative_path, "path must be a string or path-like value"
            )
        if not isinstance(raw_path, str):
            self._security.deny_path(operation, raw_path, "path must be text")
        if not raw_path or "\x00" in raw_path:
            self._security.deny_path(operation, raw_path, "path is empty or invalid")

        candidate = Path(raw_path)
        if candidate.is_absolute() or candidate.anchor or candidate.drive:
            self._security.deny_path(operation, raw_path, "absolute paths are forbidden")
        if any(part == ".." for part in candidate.parts):
            self._security.deny_path(operation, raw_path, "path traversal is forbidden")
        if any(":" in part for part in candidate.parts):
            self._security.deny_path(
                operation, raw_path, "drive and alternate-stream syntax is forbidden"
            )
        for part in candidate.parts:
            normalized_part = part.rstrip(" .")
            device_name = normalized_part.split(".", 1)[0].upper()
            if normalized_part != part or device_name in _WINDOWS_RESERVED_NAMES:
                self._security.deny_path(
                    operation, raw_path, "reserved filesystem name is forbidden"
                )

        try:
            resolved = (self._root / candidate).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            self._security.deny_path(
                operation,
                raw_path,
                "path could not be resolved safely",
                details={"error_type": type(exc).__name__},
            )
        if not self._is_inside(resolved) or resolved == self._root:
            self._security.deny_path(
                operation, raw_path, "path resolves outside world_data"
            )
        if require_allowed_extension and (
            candidate.suffix not in self._allowed_extensions
            or resolved.suffix not in self._allowed_extensions
        ):
            self._security.deny_path(
                operation,
                raw_path,
                "file extension is not allowed",
                details={
                    "requested_extension": candidate.suffix,
                    "resolved_extension": resolved.suffix,
                },
            )
        return resolved

    def validate_path(
        self,
        relative_path: str | os.PathLike[str],
        *,
        operation: str = "world_access",
        require_allowed_extension: bool = True,
    ) -> Path:
        """Public validation alias used by simulation-mode tools."""

        return self.resolve_path(
            relative_path,
            operation=operation,
            require_allowed_extension=require_allowed_extension,
        )

    def list_files(self, area: str | None = None) -> list[str]:
        """List agent-readable files without following escaping symlinks."""

        if area is None:
            base = self._root
        else:
            base = self.resolve_path(
                area,
                operation="list_world_files",
                require_allowed_extension=False,
            )
            if not base.exists():
                return []
            if not base.is_dir():
                self._security.deny_path(
                    "list_world_files", area, "list target is not a directory"
                )

        files: list[str] = []
        for path in base.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if not self._is_inside(resolved):
                continue
            if resolved.suffix not in self._allowed_extensions:
                continue
            files.append(path.relative_to(self._root).as_posix())
        return sorted(set(files), key=str.casefold)

    def read_text(self, relative_path: str | os.PathLike[str]) -> str:
        """Read a bounded UTF-8 text file from the world."""

        resolved = self.resolve_path(relative_path, operation="read_world_text")
        self._require_regular_file(resolved, relative_path, "read_world_text")
        size = resolved.stat().st_size
        if size > self._maximum_file_size_bytes:
            self._security.deny_world_limit(
                "read_world_text",
                relative_path,
                "file exceeds the configured size limit",
                details={
                    "size_bytes": size,
                    "maximum_file_size_bytes": self._maximum_file_size_bytes,
                },
            )
        try:
            return resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            self._security.deny_path(
                "read_world_text",
                relative_path,
                "file is not valid UTF-8 text",
                details={"error_type": type(exc).__name__},
            )

    def write_text(
        self, relative_path: str | os.PathLike[str], content: str
    ) -> str:
        """Atomically write bounded UTF-8 text and return its relative path."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        resolved = self.resolve_path(relative_path, operation="write_world_text")
        if resolved.exists() and not resolved.is_file():
            self._security.deny_path(
                "write_world_text", relative_path, "write target is not a regular file"
            )
        encoded = content.encode("utf-8")
        self._check_write_limits(resolved, relative_path, len(encoded))
        self._prepare_parent(resolved, relative_path, "write_world_text")
        self._atomic_write(resolved, encoded)
        return resolved.relative_to(self._root).as_posix()

    def validate_write(
        self, relative_path: str | os.PathLike[str], content: str
    ) -> str:
        """Validate a prospective write without creating or changing a file."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        resolved = self.resolve_path(relative_path, operation="write_world_text")
        if resolved.exists() and not resolved.is_file():
            self._security.deny_path(
                "write_world_text", relative_path, "write target is not a regular file"
            )
        self._check_write_limits(resolved, relative_path, len(content.encode("utf-8")))
        return resolved.relative_to(self._root).as_posix()

    def append_text(
        self, relative_path: str | os.PathLike[str], content: str
    ) -> str:
        """Append UTF-8 text while applying the same limits as a full write."""

        if not isinstance(content, str):
            raise TypeError("content must be a string")
        resolved = self.resolve_path(relative_path, operation="append_world_text")
        existing = b""
        if resolved.exists():
            self._require_regular_file(resolved, relative_path, "append_world_text")
            existing = resolved.read_bytes()
            try:
                existing.decode("utf-8")
            except UnicodeDecodeError as exc:
                self._security.deny_path(
                    "append_world_text",
                    relative_path,
                    "file is not valid UTF-8 text",
                    details={"error_type": type(exc).__name__},
                )
        encoded = existing + content.encode("utf-8")
        self._check_write_limits(resolved, relative_path, len(encoded))
        self._prepare_parent(resolved, relative_path, "append_world_text")
        self._atomic_write(resolved, encoded)
        return resolved.relative_to(self._root).as_posix()

    def snapshot(self) -> dict[str, object]:
        """Return a small deterministic observation of the visible world."""

        files = self.list_files()
        return {
            "areas": list(WORLD_AREAS),
            "files": files,
            "file_count": len(files),
            "total_size_bytes": self.total_size_bytes,
        }

    def _validate_constructor_values(self) -> None:
        if not self._allowed_extensions:
            raise ValueError("allowed_extensions cannot be empty")
        if not self._allowed_extensions.issubset(set(DEFAULT_ALLOWED_EXTENSIONS)):
            raise ValueError("allowed_extensions contains an unsafe extension")
        if (
            isinstance(self._maximum_file_size_bytes, bool)
            or not isinstance(self._maximum_file_size_bytes, int)
            or self._maximum_file_size_bytes < 1
        ):
            raise ValueError("maximum_file_size_bytes must be a positive integer")
        if (
            isinstance(self._maximum_total_world_size_bytes, bool)
            or not isinstance(self._maximum_total_world_size_bytes, int)
            or self._maximum_total_world_size_bytes < self._maximum_file_size_bytes
        ):
            raise ValueError(
                "maximum_total_world_size_bytes must be an integer no smaller "
                "than maximum_file_size_bytes"
            )

    def _is_inside(self, path: Path) -> bool:
        try:
            path.relative_to(self._root)
        except ValueError:
            return False
        return True

    def _require_regular_file(
        self,
        resolved: Path,
        original: object,
        operation: str,
    ) -> None:
        if not resolved.exists():
            self._security.deny_path(operation, original, "file does not exist")
        if not resolved.is_file():
            self._security.deny_path(operation, original, "path is not a regular file")

    def _prepare_parent(
        self, resolved: Path, original: object, operation: str
    ) -> None:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        try:
            checked_parent = resolved.parent.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            self._security.deny_path(
                operation,
                original,
                "parent directory could not be resolved safely",
                details={"error_type": type(exc).__name__},
            )
        if not self._is_inside(checked_parent):
            self._security.deny_path(
                operation, original, "parent directory escapes world_data"
            )

    def _check_write_limits(
        self, resolved: Path, original: object, new_size: int
    ) -> None:
        if new_size > self._maximum_file_size_bytes:
            self._security.deny_world_limit(
                "write_world_text",
                original,
                "file would exceed the configured size limit",
                details={
                    "size_bytes": new_size,
                    "maximum_file_size_bytes": self._maximum_file_size_bytes,
                },
            )
        existing_size = resolved.stat().st_size if resolved.is_file() else 0
        projected_total = self.total_size_bytes - existing_size + new_size
        if projected_total > self._maximum_total_world_size_bytes:
            self._security.deny_world_limit(
                "write_world_text",
                original,
                "world would exceed the configured total size limit",
                details={
                    "projected_total_size_bytes": projected_total,
                    "maximum_total_world_size_bytes": (
                        self._maximum_total_world_size_bytes
                    ),
                },
            )

    def _iter_safe_files(self) -> Iterator[Path]:
        for path in self._root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if self._is_inside(resolved):
                yield resolved

    @staticmethod
    def _atomic_write(destination: Path, content: bytes) -> None:
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=".orionworld-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)


# Clear alternative name for callers that prefer the domain term.
World = WorldSandbox
