"""The complete, explicit tool surface available to the autonomous agent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from html import escape
from pathlib import Path
import re
from typing import Final

from .database import Database
from .models import JSONValue, ToolResult
from .security import AuditCallback
from .world import WorldSandbox


ALLOWED_TOOL_NAMES: Final[tuple[str, ...]] = (
    "list_world_files",
    "read_world_text",
    "write_world_text",
    "append_diary_entry",
    "create_svg_drawing",
    "inspect_project_status",
    "create_project_note",
)
TOOL_ARGUMENT_SCHEMAS: Final[
    dict[str, tuple[frozenset[str], frozenset[str]]]
] = {
    "list_world_files": (frozenset(), frozenset({"area"})),
    "read_world_text": (frozenset({"path"}), frozenset()),
    "write_world_text": (frozenset({"path", "content"}), frozenset()),
    "append_diary_entry": (frozenset({"content"}), frozenset({"filename"})),
    "create_svg_drawing": (
        frozenset({"filename"}),
        frozenset({"title", "caption"}),
    ),
    "inspect_project_status": (frozenset(), frozenset({"project_name"})),
    "create_project_note": (
        frozenset({"project_name", "title", "content"}),
        frozenset(),
    ),
}


class ToolRegistry:
    """Closed registry for the seven permitted world-only operations."""

    def __init__(
        self,
        world: WorldSandbox,
        *,
        simulation: bool = True,
        audit_callback: AuditCallback | None = None,
        database: Database | None = None,
    ) -> None:
        if type(simulation) is not bool:
            raise TypeError("simulation must be a boolean")
        self._world = world
        self._simulation = simulation
        self._audit_callback = audit_callback
        self._database = database
        self._handlers: dict[str, Callable[..., ToolResult]] = {
            "list_world_files": self.list_world_files,
            "read_world_text": self.read_world_text,
            "write_world_text": self.write_world_text,
            "append_diary_entry": self.append_diary_entry,
            "create_svg_drawing": self.create_svg_drawing,
            "inspect_project_status": self.inspect_project_status,
            "create_project_note": self.create_project_note,
        }

    @property
    def simulation(self) -> bool:
        return self._simulation

    @property
    def tool_names(self) -> tuple[str, ...]:
        return ALLOWED_TOOL_NAMES

    @property
    def names(self) -> tuple[str, ...]:
        """Short alias useful to terminal help and diagnostics."""

        return self.tool_names

    def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, object] | None = None,
    ) -> ToolResult:
        """Execute only a registered tool and contain all ordinary failures."""

        if tool_name not in self._handlers:
            self._record_block(
                "tool_access",
                "tool is not allowed",
                {"tool": str(tool_name), "category": "tool", "blocked": True},
            )
            return ToolResult(
                success=False,
                tool=str(tool_name),
                result=None,
                files_changed=[],
                error="tool is not allowed",
            )
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            return ToolResult(
                success=False,
                tool=tool_name,
                result=None,
                files_changed=[],
                error="tool arguments must be a mapping",
            )
        if any(not isinstance(key, str) for key in arguments):
            return ToolResult(
                success=False,
                tool=tool_name,
                result=None,
                files_changed=[],
                error="tool argument names must be strings",
            )
        try:
            return self._handlers[tool_name](**dict(arguments))
        except Exception as exc:
            return self._failure(tool_name, exc)

    def list_world_files(self, area: str | None = None) -> ToolResult:
        return self._capture(
            "list_world_files",
            lambda: (self._world.list_files(area), []),
        )

    def read_world_text(self, path: str) -> ToolResult:
        return self._capture(
            "read_world_text",
            lambda: (self._world.read_text(path), []),
        )

    def write_world_text(self, path: str, content: str) -> ToolResult:
        def operation() -> tuple[JSONValue, list[str]]:
            if self._simulation:
                intended = self._world.validate_write(path, content)
                return self._simulation_result(intended, content), []
            changed = self._world.write_text(path, content)
            return {"path": changed, "bytes_written": len(content.encode("utf-8"))}, [
                changed
            ]

        return self._capture("write_world_text", operation)

    def append_diary_entry(
        self,
        content: str,
        filename: str | None = None,
    ) -> ToolResult:
        def operation() -> tuple[JSONValue, list[str]]:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content must be a non-empty string")
            chosen_filename = filename or f"{date.today().isoformat()}.md"
            if not isinstance(chosen_filename, str):
                raise TypeError("filename must be a string")
            relative = self._area_path("diary", chosen_filename, ".md")
            entry = "- " + content.strip().replace("\n", "\n  ") + "\n"
            destination = self._world.validate_path(
                relative, operation="append_diary_entry"
            )
            prefix = "# Diário do OrionWorld\n\n" if not destination.exists() else ""
            payload = prefix + entry
            if self._simulation:
                existing = (
                    self._world.read_text(relative) if destination.exists() else ""
                )
                intended = self._world.validate_write(
                    relative,
                    existing + payload,
                )
                return self._simulation_result(intended, payload), []
            changed = self._world.append_text(relative, payload)
            return {"path": changed, "entry": content.strip()}, [changed]

        return self._capture("append_diary_entry", operation)

    def create_svg_drawing(
        self,
        filename: str,
        title: str = "OrionWorld",
        caption: str = "",
    ) -> ToolResult:
        def operation() -> tuple[JSONValue, list[str]]:
            if not isinstance(filename, str):
                raise TypeError("filename must be a string")
            if not isinstance(title, str) or not isinstance(caption, str):
                raise TypeError("title and caption must be strings")
            relative = self._area_path("drawings", filename, ".svg")
            svg = self._safe_svg(title, caption)
            if self._simulation:
                intended = self._world.validate_write(relative, svg)
                return self._simulation_result(intended, svg), []
            changed = self._world.write_text(relative, svg)
            return {"path": changed, "title": title}, [changed]

        return self._capture("create_svg_drawing", operation)

    def inspect_project_status(
        self, project_name: str | None = None
    ) -> ToolResult:
        def operation() -> tuple[JSONValue, list[str]]:
            persisted_by_path: dict[str, dict[str, object]] = {}
            if self._database is not None:
                for project in self._database.list_projects(limit=1_000):
                    metadata = project.get("metadata")
                    if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
                        persisted_by_path[str(metadata["path"])] = project
            if project_name is not None:
                slug = self._project_slug(project_name)
                paths = [f"projects/{slug}.md"]
            else:
                paths = [
                    path
                    for path in self._world.list_files("projects")
                    if Path(path).suffix in {".md", ".json", ".txt"}
                ]
            projects: list[JSONValue] = []
            for path in paths:
                resolved = self._world.validate_path(
                    path, operation="inspect_project_status"
                )
                if not resolved.exists():
                    projects.append(
                        {"name": Path(path).stem, "path": path, "status": "missing"}
                    )
                    continue
                text = self._world.read_text(path)
                persisted = persisted_by_path.get(path)
                display_name = (
                    str(persisted["name"])
                    if persisted is not None
                    else Path(path).stem
                )
                status = self._infer_project_status(text)
                projects.append(
                    {
                        "name": display_name,
                        "path": path,
                        "status": status,
                        "size_bytes": len(text.encode("utf-8")),
                        "preview": text[:240],
                    }
                )
                if self._database is not None:
                    self._database.upsert_project(
                        display_name,
                        text[:500] or f"Projeto observado em {path}",
                        status=status,
                        metadata={"path": path},
                    )
            return {"projects": projects, "count": len(projects)}, []

        return self._capture("inspect_project_status", operation)

    def create_project_note(
        self,
        project_name: str,
        title: str,
        content: str,
    ) -> ToolResult:
        def operation() -> tuple[JSONValue, list[str]]:
            slug = self._project_slug(project_name)
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be a non-empty string")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content must be a non-empty string")
            relative = f"projects/{slug}.md"
            destination = self._world.validate_path(
                relative, operation="create_project_note"
            )
            heading = f"# {project_name.strip()}\n\n" if not destination.exists() else ""
            note = f"{heading}## {title.strip()}\n\n{content.strip()}\n\n"
            if self._simulation:
                existing = (
                    self._world.read_text(relative) if destination.exists() else ""
                )
                intended = self._world.validate_write(
                    relative,
                    existing + note,
                )
                return self._simulation_result(intended, note), []
            changed = self._world.append_text(relative, note)
            if self._database is not None:
                existing_project = self._database.get_project(project_name.strip())
                status = (
                    str(existing_project["status"])
                    if existing_project is not None
                    else "active"
                )
                description = (
                    str(existing_project["description"])
                    if existing_project is not None
                    else content.strip()[:500]
                )
                self._database.upsert_project(
                    project_name.strip(),
                    description,
                    status=status,
                    metadata={"path": changed},
                )
            return {"path": changed, "project": project_name.strip()}, [changed]

        return self._capture("create_project_note", operation)

    def _capture(
        self,
        tool_name: str,
        operation: Callable[[], tuple[JSONValue, list[str]]],
    ) -> ToolResult:
        try:
            result, files_changed = operation()
        except Exception as exc:
            return self._failure(tool_name, exc)
        return ToolResult(
            success=True,
            tool=tool_name,
            result=result,
            files_changed=files_changed,
            error=None,
        )

    def _record_block(
        self,
        operation: str,
        reason: str,
        details: dict[str, object],
    ) -> None:
        if self._audit_callback is None:
            return
        try:
            self._audit_callback(operation, reason, details)
        except Exception:
            # A falha da auditoria não pode transformar uma negação em permissão.
            pass

    @staticmethod
    def _failure(tool_name: str, exc: Exception) -> ToolResult:
        message = str(exc).strip() or type(exc).__name__
        return ToolResult(
            success=False,
            tool=tool_name,
            result=None,
            files_changed=[],
            error=message,
        )

    @staticmethod
    def _simulation_result(path: str, content: str) -> dict[str, JSONValue]:
        return {
            "simulation": True,
            "intended_path": path,
            "bytes": len(content.encode("utf-8")),
        }

    @staticmethod
    def _area_path(area: str, filename: str, expected_suffix: str) -> str:
        if not filename.strip():
            raise ValueError("filename cannot be empty")
        supplied = Path(filename)
        if supplied.suffix != expected_suffix:
            raise ValueError(f"filename must use the {expected_suffix} extension")
        return (Path(area) / supplied).as_posix()

    @staticmethod
    def _project_slug(project_name: str) -> str:
        if not isinstance(project_name, str) or not project_name.strip():
            raise ValueError("project_name must be a non-empty string")
        if Path(project_name).is_absolute() or any(
            part == ".." for part in Path(project_name).parts
        ):
            raise ValueError("project_name cannot contain an external path")
        normalized = re.sub(r"[^a-z0-9_-]+", "_", project_name.strip().lower())
        slug = normalized.strip("_")
        if not slug:
            raise ValueError("project_name must contain letters or numbers")
        return slug[:80]

    @staticmethod
    def _infer_project_status(content: str) -> str:
        lowered = content.casefold()
        if "status: complete" in lowered or "status: concluído" in lowered:
            return "complete"
        if "[ ]" in content or "status: incomplete" in lowered:
            return "incomplete"
        return "active"

    @staticmethod
    def _safe_svg(title: str, caption: str) -> str:
        safe_title = escape(title.strip()[:120])
        safe_caption = escape(caption.strip()[:240])
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" '
            'viewBox="0 0 640 360" role="img">\n'
            f"  <title>{safe_title}</title>\n"
            '  <rect width="640" height="360" fill="#081426"/>\n'
            '  <circle cx="320" cy="168" r="92" fill="none" '
            'stroke="#71d7ff" stroke-width="3"/>\n'
            '  <circle cx="407" cy="137" r="13" fill="#ffd166"/>\n'
            '  <circle cx="320" cy="168" r="30" fill="#8f7cff"/>\n'
            f'  <text x="320" y="300" text-anchor="middle" fill="#eef6ff" '
            f'font-family="sans-serif" font-size="18">{safe_caption}</text>\n'
            "</svg>\n"
        )


# Alternative domain name retained for straightforward dependency injection.
AllowedTools = ToolRegistry
