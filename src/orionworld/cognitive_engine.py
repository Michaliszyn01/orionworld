"""Proposal-only orchestration for OrionWorld's optional cognitive layer."""

from __future__ import annotations

from collections.abc import Sequence

from .activities import ActivityEngine
from .audit import AuditLogger
from .cognitive_models import CognitiveReflection
from .cognitive_validation import (
    CognitiveValidationError,
    parse_cognitive_proposal,
    parse_cognitive_reflection,
    sanitized_preview,
)
from .config import Settings
from .context_builder import ContextBuilder
from .model_providers import ModelProvider, ModelProviderError
from .models import Activity, InternalState, JSONValue
from .world import WorldSandbox
from .persistence_runtime import ResourceBudget


_RISK_SCORE = {"low": 0.15, "medium": 0.5, "high": 0.9}


class CognitiveEngine:
    """Ask a provider for candidates and reflections, never for execution."""

    def __init__(
        self,
        *,
        settings: Settings,
        provider: ModelProvider,
        context_builder: ContextBuilder,
        world: WorldSandbox,
        audit: AuditLogger,
        resource_budget: ResourceBudget | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.context_builder = context_builder
        self.world = world
        self.audit = audit
        self.resource_budget = resource_budget

    @property
    def enabled(self) -> bool:
        return self.settings.cognitive_engine_enabled

    def propose(
        self,
        *,
        cycle_id: int,
        state: InternalState,
        heuristic_candidates: Sequence[Activity],
        recent_activity_types: Sequence[str],
        messages: Sequence[dict[str, JSONValue]],
        interests: Sequence[dict[str, JSONValue]],
        activity_engine: ActivityEngine,
        has_incomplete_project: bool,
    ) -> Activity | None:
        if not self.enabled:
            return None
        if self.resource_budget is not None and not self.resource_budget.consume(
            "model_call", cycle_id=cycle_id
        ):
            return None
        request = self.context_builder.build_proposal_request(
            cycle_id=cycle_id,
            state=state,
            candidates=heuristic_candidates,
            recent_activity_types=recent_activity_types,
            messages=messages,
            interests=interests,
            resource_limits={
                "maximum_file_size_bytes": self.settings.maximum_file_size_bytes,
                "maximum_actions_per_cycle": self.settings.maximum_actions_per_cycle,
                "simulation": self.settings.simulation,
            },
        )
        self.audit.event(
            "cognitive_request_sent",
            "Contexto mínimo enviado ao provedor cognitivo.",
            cycle_id=cycle_id,
            metadata={"provider": self.provider.name, "kind": request.kind},
        )
        try:
            response = self.provider.generate(request)
            self.audit.event(
                "cognitive_response_received",
                "Resposta cognitiva recebida para validação.",
                cycle_id=cycle_id,
                metadata={
                    "provider": response.provider,
                    "characters": len(response.text),
                },
            )
            proposal = parse_cognitive_proposal(
                response.text,
                maximum_characters=self.settings.maximum_model_response_characters,
                world=self.world,
            )
        except (CognitiveValidationError, ModelProviderError, OSError, ValueError) as exc:
            self.audit.event(
                "cognitive_proposal_rejected",
                "Proposta cognitiva rejeitada; heurísticas locais permanecem ativas.",
                cycle_id=cycle_id,
                metadata={
                    "provider": self.provider.name,
                    "error_type": type(exc).__name__,
                    "reason": sanitized_preview(str(exc)),
                    "preview": sanitized_preview(
                        locals().get("response").text
                        if "response" in locals()
                        else "sem resposta"
                    ),
                },
            )
            return None
        if not proposal.should_act:
            self.audit.event(
                "cognitive_proposal_accepted",
                "O modelo propôs não agir; nenhuma candidatura foi adicionada.",
                cycle_id=cycle_id,
                metadata={"should_act": False, "provider": self.provider.name},
            )
            return None
        plan: dict[str, JSONValue] = {
            "activity_type": proposal.activity_type,
            "tool": proposal.requested_tool,
            "arguments": dict(proposal.tool_arguments),
            "steps": list(proposal.plan),
            "intended_files": [],
            "risk_label": proposal.risk,
            "expected_result": proposal.expected_result,
        }
        candidate = Activity(
            id=f"cycle-{cycle_id}:cognitive:{proposal.activity_type}",
            type=proposal.activity_type,
            title=proposal.title,
            description=proposal.description,
            motivation=proposal.motivation,
            expected_value=proposal.expected_value,
            novelty=proposal.novelty,
            estimated_cost=proposal.estimated_cost,
            risk=_RISK_SCORE[proposal.risk],
            required_tool=proposal.requested_tool,
            origin="cognitive",
            plan=plan,
            metadata={"provider": self.provider.name, "risk_label": proposal.risk},
        )
        candidate.score = activity_engine.score_activity(
            candidate,
            state,
            recent_activity_types,
            has_incomplete_project=has_incomplete_project,
        )
        self.audit.event(
            "cognitive_proposal_accepted",
            "Proposta cognitiva validada e adicionada apenas como candidata.",
            cycle_id=cycle_id,
            metadata={
                "activity_type": candidate.type,
                "requested_tool": candidate.required_tool,
                "local_score": candidate.score,
            },
        )
        return candidate

    def reflect(
        self,
        *,
        cycle_id: int,
        state: InternalState,
        activity: Activity,
        outcome: dict[str, JSONValue],
    ) -> CognitiveReflection | None:
        if not self.enabled:
            return None
        if self.resource_budget is not None and not self.resource_budget.consume(
            "model_call", cycle_id=cycle_id
        ):
            return None
        request = self.context_builder.build_reflection_request(
            cycle_id=cycle_id,
            state=state,
            activity=activity,
            outcome=outcome,
        )
        self.audit.event(
            "cognitive_reflection_requested",
            "Reflexão pós-ação solicitada sem autoridade operacional.",
            cycle_id=cycle_id,
            metadata={"provider": self.provider.name},
        )
        try:
            response = self.provider.generate(request)
            self.audit.event(
                "cognitive_reflection_received",
                "Resposta de reflexão recebida para validação.",
                cycle_id=cycle_id,
                metadata={"characters": len(response.text)},
            )
            reflection = parse_cognitive_reflection(
                response.text,
                maximum_characters=self.settings.maximum_model_response_characters,
            )
        except (CognitiveValidationError, ModelProviderError, OSError, ValueError) as exc:
            self.audit.event(
                "cognitive_reflection_rejected",
                "Reflexão cognitiva rejeitada.",
                cycle_id=cycle_id,
                metadata={
                    "error_type": type(exc).__name__,
                    "reason": sanitized_preview(str(exc)),
                },
            )
            return None
        self.audit.event(
            "cognitive_reflection_accepted",
            "Reflexão validada como interpretação do modelo.",
            cycle_id=cycle_id,
            metadata={"confidence": reflection.confidence},
        )
        return reflection


__all__ = ["CognitiveEngine"]
