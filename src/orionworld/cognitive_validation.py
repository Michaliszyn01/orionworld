"""Strict validation for all untrusted cognitive model responses."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from .cognitive_models import CognitiveProposal, CognitiveReflection
from .models import JSONValue
from .tools import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS
from .world import WorldSandbox


class CognitiveValidationError(ValueError):
    """Raised when model text fails the closed cognitive contract."""


_PROPOSAL_FIELDS = frozenset(
    {
        "should_act",
        "activity_type",
        "title",
        "description",
        "motivation",
        "expected_value",
        "novelty",
        "estimated_cost",
        "risk",
        "requested_tool",
        "tool_arguments",
        "plan",
        "expected_result",
    }
)
_REFLECTION_FIELDS = frozenset(
    {
        "summary",
        "outcome_assessment",
        "learned_information",
        "detected_problem",
        "confidence",
        "suggested_interest",
        "should_remember",
    }
)
_TOOL_SCHEMAS = {**TOOL_ARGUMENT_SCHEMAS, "none": (frozenset(), frozenset())}
_RISK_VALUES = frozenset({"low", "medium", "high"})
_SAFE_NAME = re.compile(r"^[\w .-]+$", re.UNICODE)


def sanitized_preview(text: object, limit: int = 180) -> str:
    """Return a single-line preview without control characters or full payloads."""

    value = text if isinstance(text, str) else type(text).__name__
    clean = " ".join(value.replace("\x00", "").split())
    clean = re.sub(
        r"(?i)(api[_-]?key|password|secret|token)(\s*[=:]\s*|\"\s*:\s*\")[^\s,}\"]+",
        r"\1\2[redacted]",
        clean,
    )
    clean = re.sub(r"(?i)[a-z]:[\\/][^\s]+", "[absolute-path-redacted]", clean)
    return clean[:limit]


def parse_cognitive_proposal(
    text: str,
    *,
    maximum_characters: int,
    world: WorldSandbox,
) -> CognitiveProposal:
    payload = _decode_exact_object(text, maximum_characters)
    _exact_fields(payload, _PROPOSAL_FIELDS, "proposal")

    should_act = _bool(payload, "should_act")
    activity_type = _text(payload, "activity_type", 80)
    title = _text(payload, "title", 120)
    description = _text(payload, "description", 800)
    motivation = _text(payload, "motivation", 500)
    expected_value = _unit_number(payload, "expected_value")
    novelty = _unit_number(payload, "novelty")
    estimated_cost = _unit_number(payload, "estimated_cost")
    risk = _text(payload, "risk", 6)
    if risk not in _RISK_VALUES:
        raise CognitiveValidationError("risk must be low, medium, or high")
    requested_tool = _text(payload, "requested_tool", 80)
    allowed_tools = frozenset(ALLOWED_TOOL_NAMES) | {"none"}
    if requested_tool not in allowed_tools:
        raise CognitiveValidationError("requested_tool is not allowed")
    arguments = payload["tool_arguments"]
    if not isinstance(arguments, dict) or any(
        not isinstance(key, str) for key in arguments
    ):
        raise CognitiveValidationError("tool_arguments must be an object")
    typed_arguments: dict[str, JSONValue] = dict(arguments)
    _validate_tool_arguments(requested_tool, typed_arguments, world)
    plan_raw = payload["plan"]
    if not isinstance(plan_raw, list) or not 1 <= len(plan_raw) <= 8:
        raise CognitiveValidationError("plan must contain between 1 and 8 steps")
    plan = tuple(_limited_string(item, "plan step", 240) for item in plan_raw)
    expected_result = _text(payload, "expected_result", 800)
    if not should_act and (requested_tool != "none" or typed_arguments):
        raise CognitiveValidationError("inactive proposal cannot request a tool")
    no_tool_activities = {
        "respond_to_user_message", "propose_project", "plan_project",
        "review_project", "pause_project", "complete_project", "abandon_project",
    }
    if should_act and requested_tool == "none" and activity_type not in no_tool_activities:
        raise CognitiveValidationError("active proposal must request an allowed tool")
    return CognitiveProposal(
        should_act=should_act,
        activity_type=activity_type,
        title=title,
        description=description,
        motivation=motivation,
        expected_value=expected_value,
        novelty=novelty,
        estimated_cost=estimated_cost,
        risk=risk,
        requested_tool=requested_tool,
        tool_arguments=typed_arguments,
        plan=plan,
        expected_result=expected_result,
    )


def parse_cognitive_reflection(
    text: str, *, maximum_characters: int
) -> CognitiveReflection:
    payload = _decode_exact_object(text, maximum_characters)
    _exact_fields(payload, _REFLECTION_FIELDS, "reflection")
    detected_problem = _optional_text(payload, "detected_problem", 500)
    suggested_interest = _optional_text(payload, "suggested_interest", 120)
    return CognitiveReflection(
        summary=_text(payload, "summary", 800),
        outcome_assessment=_text(payload, "outcome_assessment", 500),
        learned_information=_text(payload, "learned_information", 1_000),
        detected_problem=detected_problem,
        confidence=_unit_number(payload, "confidence"),
        suggested_interest=suggested_interest,
        should_remember=_bool(payload, "should_remember"),
    )


def normalize_interest(topic: str) -> str:
    """Normalize interest identity for stable, accent-insensitive deduplication."""

    decomposed = unicodedata.normalize("NFKD", topic.strip().casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_marks.split())


def _decode_exact_object(text: str, maximum_characters: int) -> dict[str, Any]:
    if not isinstance(text, str):
        raise CognitiveValidationError("model response must be text")
    if len(text) > maximum_characters:
        raise CognitiveValidationError("model response exceeded the size limit")
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise CognitiveValidationError("model response must be one JSON object") from exc
    if not isinstance(payload, dict):
        raise CognitiveValidationError("model response must be one JSON object")
    return payload


def _exact_fields(payload: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise CognitiveValidationError(
            f"{label} fields do not match the required contract"
        )


def _limited_string(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CognitiveValidationError(f"{label} must be a non-empty string")
    clean = value.strip()
    if len(clean) > maximum:
        raise CognitiveValidationError(f"{label} exceeded its length limit")
    return clean


def _text(payload: Mapping[str, Any], key: str, maximum: int) -> str:
    return _limited_string(payload[key], key, maximum)


def _optional_text(payload: Mapping[str, Any], key: str, maximum: int) -> str | None:
    value = payload[key]
    if value is None:
        return None
    return _limited_string(value, key, maximum)


def _bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload[key]
    if type(value) is not bool:
        raise CognitiveValidationError(f"{key} must be a boolean")
    return value


def _unit_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CognitiveValidationError(f"{key} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise CognitiveValidationError(f"{key} must be between 0 and 1")
    return number


def _validate_tool_arguments(
    tool: str, arguments: dict[str, JSONValue], world: WorldSandbox
) -> None:
    required, optional = _TOOL_SCHEMAS[tool]
    actual = set(arguments)
    if not required.issubset(actual) or actual.difference(required | optional):
        raise CognitiveValidationError("tool_arguments do not match the tool schema")
    nullable = {"area", "filename", "project_name"}
    for key, value in arguments.items():
        if value is None and key in nullable:
            continue
        if not isinstance(value, str):
            raise CognitiveValidationError(f"tool argument {key} must be a string")
        if len(value) > (10_000 if key == "content" else 240):
            raise CognitiveValidationError(f"tool argument {key} is too long")
    try:
        if tool == "list_world_files" and arguments.get("area") is not None:
            world.validate_path(
                str(arguments["area"]),
                operation="cognitive_validation",
                require_allowed_extension=False,
            )
        elif tool == "read_world_text":
            world.validate_path(str(arguments["path"]), operation="cognitive_validation")
        elif tool == "write_world_text":
            world.validate_write(str(arguments["path"]), str(arguments["content"]))
        elif tool == "append_diary_entry":
            filename = arguments.get("filename") or "model_entry.md"
            normalized = _safe_filename(str(filename), ".md")
            world.validate_write(f"diary/{normalized}", str(arguments["content"]))
        elif tool == "create_svg_drawing":
            filename = str(arguments["filename"])
            normalized = _safe_filename(filename, ".svg")
            world.validate_write(f"drawings/{normalized}", "<svg></svg>")
        elif tool == "inspect_project_status" and arguments.get("project_name") is not None:
            _safe_label(str(arguments["project_name"]))
        elif tool == "create_project_note":
            _safe_label(str(arguments["project_name"]))
            _limited_string(arguments["title"], "title", 240)
            _limited_string(arguments["content"], "content", 10_000)
    except (OSError, TypeError, ValueError) as exc:
        raise CognitiveValidationError("tool arguments failed sandbox validation") from exc


def _safe_filename(value: str, suffix: str) -> str:
    if not _SAFE_NAME.fullmatch(value) or "/" in value or "\\" in value:
        raise CognitiveValidationError("filename is unsafe")
    dot_suffix = "." in value
    if dot_suffix and not value.casefold().endswith(suffix):
        raise CognitiveValidationError(f"filename must end with {suffix}")
    return value if dot_suffix else value + suffix


def _safe_label(value: str) -> None:
    _limited_string(value, "project_name", 120)
    if not _SAFE_NAME.fullmatch(value):
        raise CognitiveValidationError("project_name contains unsupported characters")


__all__ = [
    "CognitiveValidationError",
    "normalize_interest",
    "parse_cognitive_proposal",
    "parse_cognitive_reflection",
    "sanitized_preview",
]
