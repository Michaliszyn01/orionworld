"""Modelos de dados pequenos e serializáveis usados pelo OrionWorld."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _bounded(value: float) -> float:
    """Normaliza um número para o intervalo fechado de zero a um."""

    numeric = float(value)
    if not isfinite(numeric):
        return 0.5
    return min(1.0, max(0.0, numeric))


@dataclass(slots=True)
class InternalState:
    """Variáveis computacionais que influenciam a escolha de atividades."""

    curiosity: float = 0.5
    boredom: float = 0.5
    social_need: float = 0.5
    completion_pressure: float = 0.5
    organization_need: float = 0.5
    confidence: float = 0.5
    energy: float = 0.5
    novelty_preference: float = 0.5

    def __post_init__(self) -> None:
        for name in self.field_names():
            setattr(self, name, _bounded(getattr(self, name)))

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return (
            "curiosity",
            "boredom",
            "social_need",
            "completion_pressure",
            "organization_need",
            "confidence",
            "energy",
            "novelty_preference",
        )

    def clamped(self) -> "InternalState":
        """Retorna uma cópia cujos valores estão garantidamente entre 0 e 1."""

        return InternalState(**self.to_dict())

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.field_names()}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "InternalState":
        defaults = cls().to_dict()
        return cls(
            **{
                name: float(values.get(name, default))
                for name, default in defaults.items()
            }
        )


@dataclass(slots=True)
class Activity:
    """Atividade candidata produzida pelo planejador autônomo."""

    id: str
    type: str
    title: str
    description: str
    motivation: str
    expected_value: float
    novelty: float
    estimated_cost: float
    risk: float
    required_tool: str
    origin: str = "heuristic"
    score: float = 0.0
    plan: dict[str, JSONValue] = field(default_factory=dict)
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "motivation": self.motivation,
            "expected_value": float(self.expected_value),
            "novelty": float(self.novelty),
            "estimated_cost": float(self.estimated_cost),
            "risk": float(self.risk),
            "required_tool": self.required_tool,
            "origin": self.origin,
            "score": float(self.score),
            "plan": dict(self.plan),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Activity":
        return cls(
            id=str(values["id"]),
            type=str(values["type"]),
            title=str(values["title"]),
            description=str(values["description"]),
            motivation=str(values["motivation"]),
            expected_value=float(values["expected_value"]),
            novelty=float(values["novelty"]),
            estimated_cost=float(values["estimated_cost"]),
            risk=float(values["risk"]),
            required_tool=str(values["required_tool"]),
            origin=str(values.get("origin", "heuristic")),
            score=float(values.get("score", 0.0)),
            plan=dict(values.get("plan") or {}),
            metadata=dict(values.get("metadata") or {}),
        )


@dataclass(slots=True)
class ToolResult:
    """Resposta estruturada comum a todas as ferramentas permitidas."""

    success: bool
    tool: str
    result: JSONValue = None
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "success": bool(self.success),
            "tool": self.tool,
            "result": self.result,
            "files_changed": list(self.files_changed),
            "error": self.error,
        }
