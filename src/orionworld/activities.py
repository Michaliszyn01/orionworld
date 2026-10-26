"""Candidate generation, scoring and seeded selection for OrionWorld."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import Activity, InternalState


@dataclass(frozen=True, slots=True)
class _ActivitySpec:
    type: str
    title: str
    description: str
    required_tool: str
    expected_value: float
    base_novelty: float
    estimated_cost: float
    risk: float
    curiosity_affinity: float
    boredom_relief: float
    continuity_affinity: float
    need_field: str


_SPECS: tuple[_ActivitySpec, ...] = (
    _ActivitySpec(
        "observe_world",
        "Observar o mundo",
        "Listar os arquivos existentes para notar mudanças no ambiente isolado.",
        "list_world_files",
        0.58,
        0.42,
        0.18,
        0.05,
        0.88,
        0.36,
        0.35,
        "curiosity",
    ),
    _ActivitySpec(
        "write_diary",
        "Escrever no diário",
        "Registrar uma observação curta sobre o ciclo atual.",
        "append_diary_entry",
        0.62,
        0.34,
        0.24,
        0.08,
        0.40,
        0.52,
        0.58,
        "social_need",
    ),
    _ActivitySpec(
        "create_text_experiment",
        "Criar experimento de texto",
        "Produzir um texto breve para explorar uma ideia nova.",
        "write_world_text",
        0.72,
        0.82,
        0.54,
        0.16,
        0.92,
        0.84,
        0.28,
        "novelty_preference",
    ),
    _ActivitySpec(
        "create_svg_drawing",
        "Criar desenho SVG",
        "Compor um desenho geométrico simples dentro do mundo.",
        "create_svg_drawing",
        0.70,
        0.90,
        0.64,
        0.14,
        0.84,
        0.94,
        0.22,
        "novelty_preference",
    ),
    _ActivitySpec(
        "review_recent_memory",
        "Revisar memória recente",
        "Inspecionar registros recentes e recuperar contexto útil.",
        "list_world_files",
        0.66,
        0.28,
        0.30,
        0.05,
        0.56,
        0.34,
        0.68,
        "organization_need",
    ),
    _ActivitySpec(
        "organize_projects",
        "Organizar projetos",
        "Inspecionar a situação dos projetos e identificar próximos passos.",
        "inspect_project_status",
        0.68,
        0.22,
        0.42,
        0.06,
        0.32,
        0.28,
        0.78,
        "organization_need",
    ),
    _ActivitySpec(
        "continue_incomplete_project",
        "Continuar projeto incompleto",
        "Adicionar uma nota segura ao próximo projeto que ainda esteja aberto.",
        "create_project_note",
        0.80,
        0.40,
        0.68,
        0.12,
        0.62,
        0.66,
        1.00,
        "completion_pressure",
    ),
    _ActivitySpec(
        "idle",
        "Permanecer ocioso",
        "Não executar ferramenta neste ciclo e recuperar energia.",
        "none",
        0.28,
        0.08,
        0.02,
        0.00,
        0.05,
        0.04,
        0.12,
        "energy",
    ),
)

PROJECT_ACTIVITY_TYPES: tuple[str, ...] = (
    "propose_project", "plan_project", "continue_project", "review_project",
    "pause_project", "complete_project", "abandon_project",
)
ACTIVITY_TYPES: tuple[str, ...] = tuple(spec.type for spec in _SPECS) + PROJECT_ACTIVITY_TYPES


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _repetition(activity_type: str, recent: Sequence[str]) -> float:
    """Return a recency-weighted repetition value between zero and one."""

    if not recent:
        return 0.0
    weights = tuple(range(1, len(recent) + 1))
    matching = sum(
        weight for weight, previous in zip(weights, recent) if previous == activity_type
    )
    return matching / sum(weights)


class ActivityEngine:
    """Generate all MVP activities and select among the best with seeded RNG."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._random = random.Random(seed)

    def generate(
        self,
        cycle_id: int,
        state: InternalState,
        recent_activity_types: Sequence[str] = (),
        has_incomplete_project: bool = False,
    ) -> list[Activity]:
        """Generate and score the eight autonomous MVP candidates."""

        recent = tuple(recent_activity_types)
        candidates: list[Activity] = []
        for spec in _SPECS:
            novelty = _clamp(spec.base_novelty + 0.24 * (1.0 - _repetition(spec.type, recent)))
            expected_value = spec.expected_value
            if spec.type == "continue_incomplete_project" and not has_incomplete_project:
                expected_value = 0.12

            activity = Activity(
                id=f"cycle-{cycle_id}:{spec.type}",
                type=spec.type,
                title=spec.title,
                description=spec.description,
                motivation=self._motivation(spec, state, has_incomplete_project),
                expected_value=expected_value,
                novelty=novelty,
                estimated_cost=spec.estimated_cost,
                risk=spec.risk,
                required_tool=spec.required_tool,
                score=0.0,
            )
            activity.score = self.score(
                activity,
                state,
                recent,
                has_incomplete_project=has_incomplete_project,
            )
            activity.plan = self.plan_for(activity, cycle_id)
            candidates.append(activity)
        return candidates

    def score(
        self,
        activity: Activity,
        state: InternalState,
        recent_activity_types: Sequence[str] = (),
        *,
        has_incomplete_project: bool = False,
    ) -> float:
        """Calculate a transparent utility score in the inclusive range 0..1.

        Positive terms are expected value, curiosity, boredom relief, novelty,
        continuity, need alignment and available energy.  Cost, risk and recent
        repetition are explicit penalties.
        """

        spec = self._spec_for(activity.type, activity)
        recent = tuple(recent_activity_types)
        repetition = _repetition(activity.type, recent)
        same_as_previous = bool(recent and recent[-1] == activity.type)

        continuity = spec.continuity_affinity if same_as_previous else 0.0
        if activity.type == "continue_incomplete_project" and has_incomplete_project:
            continuity = 1.0

        need_alignment = float(getattr(state, spec.need_field))
        if activity.type == "idle":
            # Low energy increases the usefulness of rest.
            need_alignment = 1.0 - state.energy

        energy_fit = _clamp(
            state.energy - max(0.0, activity.estimated_cost - state.energy)
        )
        curiosity_term = state.curiosity * spec.curiosity_affinity
        boredom_term = state.boredom * spec.boredom_relief
        novelty_term = activity.novelty * (0.40 + 0.60 * state.novelty_preference)

        positive = (
            0.18 * activity.expected_value
            + 0.15 * curiosity_term
            + 0.13 * boredom_term
            + 0.14 * novelty_term
            + 0.12 * continuity
            + 0.12 * need_alignment
            + 0.10 * energy_fit
        )
        cost_penalty = (
            0.08 * activity.estimated_cost * (1.25 - 0.50 * state.energy)
        )
        risk_penalty = 0.06 * activity.risk * (1.20 - 0.40 * state.confidence)
        repetition_penalty = 0.15 * repetition

        return round(
            _clamp(positive - cost_penalty - risk_penalty - repetition_penalty),
            6,
        )

    # Explicit alias reads well in integrations and external tests.
    score_activity = score

    def choose(
        self,
        candidates: Sequence[Activity],
        *,
        top_k: int = 3,
        temperature: float = 0.12,
    ) -> Activity:
        """Choose probabilistically from the best candidates using softmax.

        Only the ``top_k`` scored options participate.  Seeded sampling avoids a
        rigid argmax while preserving exact replay when call order is the same.
        """

        if not candidates:
            raise ValueError("at least one candidate is required")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")

        ranked = sorted(candidates, key=lambda item: (-item.score, item.type, item.id))
        shortlist = ranked[: min(top_k, len(ranked))]
        best_score = shortlist[0].score
        weights = [
            math.exp((candidate.score - best_score) / temperature)
            for candidate in shortlist
        ]
        return self._random.choices(shortlist, weights=weights, k=1)[0]

    def selection_probabilities(
        self,
        candidates: Sequence[Activity],
        *,
        top_k: int = 3,
        temperature: float = 0.12,
    ) -> list[tuple[Activity, float]]:
        """Expose selection probabilities for functional intention logs."""

        if not candidates:
            return []
        if top_k < 1 or temperature <= 0.0:
            raise ValueError("top_k and temperature must be positive")
        ranked = sorted(candidates, key=lambda item: (-item.score, item.type, item.id))
        shortlist = ranked[: min(top_k, len(ranked))]
        best_score = shortlist[0].score
        weights = [math.exp((item.score - best_score) / temperature) for item in shortlist]
        total = sum(weights)
        return [(item, weight / total) for item, weight in zip(shortlist, weights)]

    def plan_for(self, activity: Activity, cycle_id: int) -> dict[str, Any]:
        """Build a JSON-serializable, world-relative action plan for LifeLoop."""

        cycle_label = f"{cycle_id:06d}"
        plans: Mapping[str, tuple[Mapping[str, Any], list[str], list[str]]] = {
            "observe_world": (
                {},
                ["Listar arquivos disponíveis no mundo isolado."],
                [],
            ),
            "write_diary": (
                {
                    "content": f"Ciclo {cycle_id}: observei o estado atual do OrionWorld.",
                    "filename": "orion_diary.md",
                },
                ["Acrescentar uma entrada curta e factual ao diário."],
                ["diary/orion_diary.md"],
            ),
            "create_text_experiment": (
                {
                    "path": f"experiments/experiment_{cycle_label}.md",
                    "content": (
                        f"# Experimento do ciclo {cycle_id}\n\n"
                        "Uma observação breve criada dentro do mundo isolado.\n"
                    ),
                },
                ["Criar um texto curto.", "Gravá-lo na área de experimentos."],
                [f"experiments/experiment_{cycle_label}.md"],
            ),
            "create_svg_drawing": (
                {
                    "filename": f"drawing_{cycle_label}.svg",
                    "title": f"Órbita do ciclo {cycle_id}",
                    "caption": "Uma órbita geométrica criada no mundo isolado.",
                },
                ["Gerar uma composição SVG geométrica simples e segura."],
                [f"drawings/drawing_{cycle_label}.svg"],
            ),
            "review_recent_memory": (
                {"area": "diary"},
                ["Listar registros recentes do diário para recuperar contexto."],
                [],
            ),
            "organize_projects": (
                {},
                ["Inspecionar projetos persistidos e identificar itens incompletos."],
                [],
            ),
            "continue_incomplete_project": (
                {
                    "project_name": "orion_observations",
                    "title": f"Continuidade do ciclo {cycle_id}",
                    "content": f"Continuidade registrada no ciclo {cycle_id}.",
                },
                ["Acrescentar uma nota curta a um projeto incompleto."],
                ["projects/orion_observations.md"],
            ),
            "idle": (
                {},
                ["Não executar ferramenta neste ciclo."],
                [],
            ),
        }
        arguments, steps, intended_files = plans[activity.type]
        return {
            "activity_type": activity.type,
            "tool": activity.required_tool,
            "arguments": dict(arguments),
            "steps": list(steps),
            "intended_files": list(intended_files),
            "risk": activity.risk,
        }

    def reset(self, seed: int | None = None) -> None:
        """Restart the reproducible selection sequence, optionally with a new seed."""

        if seed is not None:
            self.seed = seed
        self._random.seed(self.seed)

    @staticmethod
    def _spec_for(
        activity_type: str, activity: Activity | None = None
    ) -> _ActivitySpec:
        for spec in _SPECS:
            if spec.type == activity_type:
                return spec
        if activity is None:
            raise ValueError(f"unknown activity type: {activity_type}")
        return _ActivitySpec(
            activity_type,
            activity.title,
            activity.description,
            activity.required_tool,
            activity.expected_value,
            activity.novelty,
            activity.estimated_cost,
            activity.risk,
            0.55,
            0.45,
            0.25,
            "curiosity",
        )

    @staticmethod
    def _motivation(
        spec: _ActivitySpec,
        state: InternalState,
        has_incomplete_project: bool,
    ) -> str:
        if spec.type == "idle":
            return f"Recuperar energia, atualmente em {state.energy:.2f}."
        if spec.type == "continue_incomplete_project" and not has_incomplete_project:
            return "Manter a opção visível, embora não exista projeto incompleto agora."
        value = float(getattr(state, spec.need_field))
        labels = {
            "curiosity": "curiosidade",
            "social_need": "necessidade de expressão",
            "novelty_preference": "preferência por novidade",
            "organization_need": "necessidade de organização",
            "completion_pressure": "pressão por conclusão",
        }
        label = labels.get(spec.need_field, spec.need_field)
        return f"Responder à {label}, atualmente em {value:.2f}."


__all__ = ["ACTIVITY_TYPES", "PROJECT_ACTIVITY_TYPES", "ActivityEngine"]
