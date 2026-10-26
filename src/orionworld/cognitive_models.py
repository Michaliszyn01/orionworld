"""Typed contracts for the optional, proposal-only cognitive layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import JSONValue


@dataclass(frozen=True, slots=True)
class CognitiveRequest:
    """Minimal sanitized input passed to a model provider."""

    kind: str
    cycle_id: int
    prompt: str
    internal_state: dict[str, float]
    candidate_summaries: tuple[dict[str, JSONValue], ...] = ()
    recent_activity_types: tuple[str, ...] = ()
    messages: tuple[dict[str, JSONValue], ...] = ()
    interests: tuple[dict[str, JSONValue], ...] = ()
    world_context: tuple[dict[str, JSONValue], ...] = ()
    resource_limits: dict[str, JSONValue] = field(default_factory=dict)
    action_outcome: dict[str, JSONValue] | None = None


@dataclass(frozen=True, slots=True)
class CognitiveResponse:
    """Raw provider output. It has no authority until strictly validated."""

    text: str
    provider: str


@dataclass(frozen=True, slots=True)
class CognitiveProposal:
    """Validated model suggestion; still only a candidate for local scoring."""

    should_act: bool
    activity_type: str
    title: str
    description: str
    motivation: str
    expected_value: float
    novelty: float
    estimated_cost: float
    risk: str
    requested_tool: str
    tool_arguments: dict[str, JSONValue]
    plan: tuple[str, ...]
    expected_result: str


@dataclass(frozen=True, slots=True)
class CognitiveReflection:
    """Validated interpretation of an observed action outcome."""

    summary: str
    outcome_assessment: str
    learned_information: str
    detected_problem: str | None
    confidence: float
    suggested_interest: str | None
    should_remember: bool


__all__ = [
    "CognitiveProposal",
    "CognitiveReflection",
    "CognitiveRequest",
    "CognitiveResponse",
]
