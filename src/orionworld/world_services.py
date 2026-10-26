"""Safe local services for behavior, projects, observation, journal, and SVG."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from xml.etree import ElementTree

from .audit import AuditLogger
from .config import Settings
from .database import Database
from .models import Activity, JSONValue
from .tools import ALLOWED_TOOL_NAMES
from .world import WorldSandbox


def _normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", value.casefold()).split())


class BehaviorMonitor:
    """Deterministic repetition metrics without external models."""

    def __init__(self, database: Database, settings: Settings, audit: AuditLogger) -> None:
        self.database = database
        self.settings = settings
        self.audit = audit

    def metrics(self, candidate: Activity | None = None) -> dict[str, JSONValue]:
        intentions = list(reversed(self.database.list_intentions(limit=12)))
        actions = list(reversed(self.database.list_actions(limit=12)))
        types = [str(row["activity_type"]) for row in intentions]
        tools = [str(row["tool"]) for row in actions]
        failures = 0
        for row in reversed(actions):
            if bool(row["success"]):
                break
            failures += 1
        alternating = len(types) >= 6 and len(set(types[-6:])) == 2 and all(
            types[index] == types[index - 2] for index in range(len(types) - 4, len(types))
        )
        candidate_similarity = 0.0
        if candidate is not None:
            candidate_text = _normalize_text(
                " ".join((candidate.title, candidate.motivation, candidate.description))
            )
            for row in intentions[-6:]:
                previous = _normalize_text(
                    " ".join((str(row["title"]), str(row["motivation"])))
                )
                candidate_similarity = max(
                    candidate_similarity,
                    SequenceMatcher(None, candidate_text, previous).ratio(),
                )
        consecutive_type = 0
        if types:
            for value in reversed(types):
                if value != types[-1]:
                    break
                consecutive_type += 1
        return {
            "same_activity_type": consecutive_type,
            "same_tool": sum(1 for value in tools[-6:] if tools and value == tools[-1]),
            "consecutive_failures": failures,
            "alternating_pair": alternating,
            "candidate_similarity": round(candidate_similarity, 6),
        }

    def penalize(self, candidates: list[Activity], cycle_id: int) -> bool:
        detected = False
        for candidate in candidates:
            values = self.metrics(candidate)
            excessive = (
                int(values["same_activity_type"]) >= self.settings.maximum_consecutive_similar_activities
                or float(values["candidate_similarity"]) >= 0.88
                or bool(values["alternating_pair"])
            )
            if excessive and candidate.type not in {"continue_incomplete_project", "respond_to_user_message"}:
                candidate.score = round(max(0.0, candidate.score * 0.55), 6)
                detected = True
        if detected:
            self.audit.event(
                "repetition_detected",
                "Padrão repetitivo detectado; candidatas semelhantes foram penalizadas.",
                cycle_id=cycle_id,
                metadata=self.metrics(),
            )
            for candidate in candidates:
                if candidate.type == "idle":
                    candidate.score = max(candidate.score, 0.72)
        return detected


class ProjectService:
    VALID_STATUSES = {
        "idea", "planning", "active", "blocked", "paused", "completed",
        "abandoned", "archived",
    }

    def __init__(self, database: Database, settings: Settings, audit: AuditLogger) -> None:
        self.database = database
        self.settings = settings
        self.audit = audit

    def propose(
        self,
        *,
        name: str,
        description: str,
        objective: str,
        success_criteria: str,
        origin: str = "heuristic",
        steps: list[dict[str, str]] | None = None,
    ) -> int:
        validated_steps: list[tuple[str, str, str]] = []
        for step in steps or ():
            tool = str(step.get("required_tool", "none"))
            if tool not in frozenset(ALLOWED_TOOL_NAMES) | {"none"}:
                raise ValueError("plano contém ferramenta não permitida")
            title = str(step.get("title", "Etapa"))
            description_value = str(step.get("description", ""))
            if not title.strip() or len(title) > 160 or len(description_value) > 1_000:
                raise ValueError("etapa inválida")
            validated_steps.append((title, description_value, tool))
        project_id = self.database.create_managed_project(
            name,
            description,
            origin=origin,
            objective=objective,
            success_criteria=success_criteria,
            maximum_active=self.settings.maximum_active_projects,
        )
        for position, (title, description_value, tool) in enumerate(validated_steps, 1):
            self.database.add_project_step(
                project_id,
                position,
                title,
                description_value,
                tool,
                metadata={"arguments_deferred": True},
            )
        self.audit.event(
            "project_proposed",
            "Projeto validado e persistido sem conceder novas permissões.",
            metadata={"project_id": project_id, "origin": origin},
        )
        return project_id

    def perform_lifecycle(self, activity_type: str, arguments: dict[str, object]) -> dict[str, JSONValue]:
        project_id = int(arguments.get("project_id", 0))
        if activity_type == "propose_project":
            created = self.propose(
                name=str(arguments.get("name", "Projeto Orion")),
                description=str(arguments.get("description", "Exploração local persistente.")),
                objective=str(arguments.get("objective", "Produzir observações locais seguras.")),
                success_criteria=str(arguments.get("success_criteria", "Registrar três observações distintas.")),
                origin=str(arguments.get("origin", "heuristic")),
                steps=[
                    {
                        "title": "Registrar observação",
                        "description": "Acrescentar uma nota local ao projeto.",
                        "required_tool": "create_project_note",
                    }
                ],
            )
            return {"project_id": created, "status": "idea"}
        status_by_activity = {
            "plan_project": ("planning", "planning", None),
            "review_project": ("active", "review", None),
            "pause_project": ("paused", "paused", None),
            "complete_project": ("completed", "completed", None),
            "abandon_project": (
                "abandoned", "abandoned", str(arguments.get("reason", "Sem continuidade útil após revisão.")),
            ),
        }
        if activity_type not in status_by_activity or project_id < 1:
            raise ValueError("atividade de projeto inválida")
        status, stage, reason = status_by_activity[activity_type]
        self.database.update_project_lifecycle(
            project_id,
            status,
            current_stage=stage,
            progress=1.0 if status == "completed" else None,
            abandoned_reason=reason,
        )
        self.audit.event(
            "project_lifecycle_updated",
            "Estado do projeto atualizado sem excluir arquivos.",
            metadata={"project_id": project_id, "status": status},
        )
        return {"project_id": project_id, "status": status}


class WorldObserver:
    """Metadata-only world snapshots with relative paths and change detection."""

    def __init__(self, world: WorldSandbox, database: Database) -> None:
        self.world = world
        self.database = database

    def observe(self, cycle_id: int | None) -> dict[str, JSONValue]:
        previous_rows = self.database.list_world_snapshots(limit=1)
        previous_files = {}
        if previous_rows:
            summary = previous_rows[0]["summary"]
            if isinstance(summary, dict):
                previous_files = {
                    str(item["path"]): item
                    for item in summary.get("files", [])
                    if isinstance(item, dict) and isinstance(item.get("path"), str)
                }
        files: list[dict[str, JSONValue]] = []
        for relative in self.world.list_files():
            resolved = self.world.validate_path(relative, operation="world_snapshot")
            content = self.world.read_text(relative)
            files.append(
                {
                    "path": relative,
                    "area": relative.split("/", 1)[0],
                    "size": len(content.encode("utf-8")),
                    "extension": Path(relative).suffix,
                    "modified_ns": resolved.stat().st_mtime_ns,
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "untrusted": True,
                }
            )
        current = {str(item["path"]): item for item in files}
        changes: list[dict[str, JSONValue]] = []
        for path in sorted(current.keys() - previous_files.keys()):
            changes.append({"type": "new_file", "path": path})
        for path in sorted(previous_files.keys() - current.keys()):
            changes.append({"type": "missing_file", "path": path})
        for path in sorted(current.keys() & previous_files.keys()):
            if current[path].get("sha256") != previous_files[path].get("sha256"):
                changes.append({"type": "changed_file", "path": path})
        payload: dict[str, JSONValue] = {
            "files": files,
            "changes": changes,
            "file_count": len(files),
            "projects": len(self.database.list_projects(limit=10_000)),
            "drawings": sum(1 for item in files if item["area"] == "drawings"),
            "messages": len(self.database.list_messages(limit=10_000)),
            "interests": len(self.database.list_interests(limit=10_000)),
        }
        fingerprint = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        snapshot_id = self.database.save_world_snapshot(
            cycle_id, payload, fingerprint, metadata={"content_untrusted": True}
        )
        payload["snapshot_id"] = snapshot_id
        payload["fingerprint"] = fingerprint
        return payload


class SvgRejected(ValueError):
    pass


class SafeSvgRenderer:
    ALLOWED_TAGS = {
        "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline",
        "polygon", "text", "title", "desc", "defs", "linearGradient",
        "radialGradient", "stop", "clipPath",
    }
    ALLOWED_ATTRIBUTES = {
        "xmlns", "width", "height", "viewBox", "x", "y", "x1", "x2", "y1",
        "y2", "cx", "cy", "r", "rx", "ry", "d", "points", "fill", "stroke",
        "stroke-width", "opacity", "transform", "offset", "stop-color",
        "stop-opacity", "clip-path", "id", "font-size", "text-anchor",
    }

    def __init__(self, *, maximum_bytes: int = 100_000, maximum_elements: int = 500) -> None:
        self.maximum_bytes = maximum_bytes
        self.maximum_elements = maximum_elements

    def sanitize(self, source: str) -> str:
        encoded = source.encode("utf-8")
        lowered = source.casefold()
        if len(encoded) > self.maximum_bytes:
            raise SvgRejected("SVG excedeu o limite de tamanho")
        if "<!doctype" in lowered or "<!entity" in lowered:
            raise SvgRejected("entidades externas não são permitidas")
        try:
            root = ElementTree.fromstring(source)
        except ElementTree.ParseError as exc:
            raise SvgRejected("SVG malformado") from exc
        elements = list(root.iter())
        if len(elements) > self.maximum_elements:
            raise SvgRejected("SVG excedeu o limite de elementos")
        for element in elements:
            tag = element.tag.rsplit("}", 1)[-1]
            if tag not in self.ALLOWED_TAGS or tag in {"script", "foreignObject"}:
                raise SvgRejected(f"elemento SVG não permitido: {tag}")
            for raw_name, value in element.attrib.items():
                name = raw_name.rsplit("}", 1)[-1]
                if name.casefold().startswith("on") or name not in self.ALLOWED_ATTRIBUTES:
                    raise SvgRejected(f"atributo SVG não permitido: {name}")
                folded = value.strip().casefold()
                if "url(" in folded and not re.fullmatch(r"url\(#[A-Za-z0-9_-]+\)", value.strip()):
                    raise SvgRejected("referência SVG remota não permitida")
                if folded.startswith(("http:", "https:", "//", "data:")):
                    raise SvgRejected("URL SVG externa não permitida")
        return ElementTree.tostring(root, encoding="unicode")


_ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|/(?:home|users|etc|var)/)")
_SECRET = re.compile(r"(?i)(api[_-]?key|password|secret|bearer\s+|token\s*[=:])")


class CognitiveJournal:
    def __init__(self, database: Database) -> None:
        self.database = database

    def add(
        self,
        *,
        cycle_id: int | None,
        entry_type: str,
        title: str,
        content: str,
        confidence: float,
        related_project_id: int | None = None,
        related_memory_id: int | None = None,
    ) -> int:
        if _ABSOLUTE_PATH.search(content) or _SECRET.search(content):
            raise ValueError("entrada do diário contém dado não permitido")
        if entry_type == "hypothesis" and "hipótese" not in title.casefold():
            title = "Hipótese: " + title
        return self.database.save_journal_entry(
            cycle_id, entry_type, title, content, confidence,
            related_project_id=related_project_id,
            related_memory_id=related_memory_id,
            metadata={
                "epistemic_status": "hypothesis" if entry_type == "hypothesis" else "observed_or_summary",
                "private_reasoning": False,
            },
        )


__all__ = [
    "BehaviorMonitor", "CognitiveJournal", "ProjectService", "SafeSvgRenderer",
    "SvgRejected", "WorldObserver",
]
