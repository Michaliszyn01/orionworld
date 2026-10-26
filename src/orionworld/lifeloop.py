"""Ciclo de vida persistente do agente OrionWorld."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .activities import ActivityEngine
from .audit import AuditLogger
from .config import Settings
from .cognitive_engine import CognitiveEngine
from .database import Database
from .models import Activity, InternalState, JSONValue, ToolResult
from .state import StateEngine
from .tools import ToolRegistry
from .world import WorldSandbox
from .persistence_runtime import LifeMode, ModeController, ResourceBudget
from .world_services import (
    BehaviorMonitor,
    CognitiveJournal,
    ProjectService,
    WorldObserver,
)


@dataclass(frozen=True, slots=True)
class LoopSnapshot:
    """Visão consistente e pequena do loop para a interface de terminal."""

    status: str = "created"
    cycle_id: int = 0
    activity: str | None = None
    intention: str | None = None
    motivation: str | None = None
    action: str | None = None
    result: str | None = None
    internal_state: dict[str, float] | None = None
    seconds_until_next_cycle: float | None = None
    life_mode: str = "active"
    budget_remaining: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LifeLoop:
    """Executa o pipeline observar-decidir-agir-registrar em uma thread."""

    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        world: WorldSandbox,
        tools: ToolRegistry,
        activity_engine: ActivityEngine,
        state_engine: StateEngine,
        audit: AuditLogger,
        cognitive_engine: CognitiveEngine | None = None,
        mode_controller: ModeController | None = None,
        resource_budget: ResourceBudget | None = None,
        behavior_monitor: BehaviorMonitor | None = None,
        world_observer: WorldObserver | None = None,
        journal: CognitiveJournal | None = None,
        project_service: ProjectService | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.world = world
        self.tools = tools
        self.activity_engine = activity_engine
        self.state_engine = state_engine
        self.audit = audit
        self.cognitive_engine = cognitive_engine
        self.mode_controller = mode_controller or ModeController(settings)
        self.resource_budget = resource_budget
        self.behavior_monitor = behavior_monitor
        self.world_observer = world_observer
        self.journal = journal
        self.project_service = project_service
        self.runtime_session_id: int | None = None

        if tools.simulation is not settings.simulation:
            raise ValueError(
                "ToolRegistry e LifeLoop devem usar o mesmo modo de simulação"
            )

        self._snapshot_lock = threading.RLock()
        self._cycle_lock = threading.Lock()
        self._interval_lock = threading.Lock()
        self._cycle_interval = float(settings.cycle_interval_seconds)
        initial_state = database.latest_internal_state() or InternalState()
        self._snapshot = LoopSnapshot(internal_state=initial_state.to_dict())

    def set_runtime_session(self, session_id: int | None) -> None:
        self.runtime_session_id = session_id

    def request_reflection(self) -> None:
        self.mode_controller.reflection_requested = True

    def enter_rest(self) -> None:
        self.mode_controller.enter_rest()
        self._replace_snapshot(life_mode=LifeMode.RESTING.value)

    def snapshot(self) -> LoopSnapshot:
        with self._snapshot_lock:
            values = self._snapshot.to_dict()
        return LoopSnapshot(**values)

    def set_status(self, status: str) -> None:
        self._replace_snapshot(status=status)

    def set_cycle_interval(self, seconds: float) -> None:
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) < 0.001
        ):
            raise ValueError("O intervalo deve ser finito e de pelo menos 0.001s")
        with self._interval_lock:
            self._cycle_interval = float(seconds)

    @property
    def cycle_interval(self) -> float:
        with self._interval_lock:
            return self._cycle_interval

    def run(self, stop_event: threading.Event, run_event: threading.Event) -> None:
        """Permanece ativo até o Supervisor sinalizar encerramento."""

        self.set_status("running")
        while not stop_event.is_set():
            if not run_event.is_set():
                self._replace_snapshot(status="paused", seconds_until_next_cycle=None)
                stop_event.wait(0.1)
                continue

            self.set_status("running")
            try:
                self.run_cycle()
            except Exception as exc:  # última barreira: o processo deve sobreviver
                self.audit.event(
                    "cycle_error",
                    "O ciclo falhou, mas o LifeLoop continuará ativo.",
                    cycle_id=self.snapshot().cycle_id or None,
                    metadata={"error_type": type(exc).__name__, "error": str(exc)},
                    terminal_label="ERRO DE CICLO",
                )
                self._replace_snapshot(result=f"Erro recuperável: {exc}")

            if self._wait_for_next_cycle(stop_event, run_event):
                break

        self._replace_snapshot(status="stopped", seconds_until_next_cycle=None)

    def run_cycle(self) -> LoopSnapshot:
        """Executa um ciclo completo; útil também para testes determinísticos."""

        if not self._cycle_lock.acquire(blocking=False):
            raise RuntimeError("Já existe um ciclo em andamento")
        try:
            cycle_id = self.database.next_cycle_id()
            cycle_row_id = self.database.begin_cycle(self.runtime_session_id, cycle_id)
            self._replace_snapshot(
                status="running",
                cycle_id=cycle_id,
                activity=None,
                intention=None,
                motivation=None,
                action=None,
                result=None,
                seconds_until_next_cycle=None,
                life_mode=self.mode_controller.mode.value,
                budget_remaining=(
                    self.resource_budget.remaining() if self.resource_budget else None
                ),
            )
            self.audit.event(
                "cycle_started",
                f"Ciclo {cycle_id} iniciado.",
                cycle_id=cycle_id,
                terminal_label="CICLO",
            )

            self.database.update_cycle(cycle_row_id, "observing")
            observation = self.world.snapshot()
            if (
                self.world_observer is not None
                and cycle_id % self.settings.world_snapshot_interval_cycles == 0
            ):
                observed_world = self.world_observer.observe(cycle_id)
                observation["changes"] = observed_world.get("changes", [])
                observation["fingerprint"] = observed_world.get("fingerprint")
            logged_observation = self._sanitize_observation(observation)
            self.audit.event(
                "environment_observed",
                (
                    f"Ambiente observado: {observation['file_count']} arquivo(s) "
                    f"visível(is), {observation['total_size_bytes']} byte(s)."
                ),
                cycle_id=cycle_id,
                metadata=logged_observation,
                terminal_label="OBSERVAÇÃO",
            )
            state = self.database.latest_internal_state() or InternalState()
            memories = self.database.recall_memories(limit=5)
            pending_messages = self.database.list_messages(
                sender="user", status="pending", limit=5
            )
            interests = self.database.list_interests(limit=10)
            recent_types = self._recent_activity_types()
            incomplete_project = self._incomplete_project()
            has_incomplete_project = incomplete_project is not None
            repetition_before = (
                bool(self.behavior_monitor.metrics().get("alternating_pair"))
                if self.behavior_monitor is not None
                else False
            )
            life_mode = self.mode_controller.choose(
                state,
                pending_messages=len(pending_messages),
                has_incomplete_project=has_incomplete_project,
                repetition_detected=repetition_before,
            )
            self._replace_snapshot(life_mode=life_mode.value)
            self.database.update_cycle(
                cycle_row_id, "deciding", metadata={"life_mode": life_mode.value}
            )

            # Derivar a aleatoriedade do ciclo persistente permite reproduzir o
            # mesmo mundo mesmo quando o processo é interrompido e retomado.
            cycle_seed = self.settings.random_seed + cycle_id
            self.activity_engine.reset(cycle_seed)
            self.state_engine.reset(cycle_seed)

            candidates = self.activity_engine.generate(
                cycle_id,
                state,
                recent_activity_types=recent_types,
                has_incomplete_project=has_incomplete_project,
            )
            for candidate in candidates:
                candidate.origin = "heuristic"
            if self.project_service is not None:
                candidates.extend(
                    self._project_candidates(cycle_id, state, recent_types)
                )
            if pending_messages:
                candidates.append(
                    self._message_response_candidate(
                        cycle_id, state, pending_messages[-1], recent_types
                    )
                )
            cognitive_candidate = None
            if (
                self.cognitive_engine is not None
                and life_mode is not LifeMode.RESTING
            ):
                cognitive_candidate = self.cognitive_engine.propose(
                    cycle_id=cycle_id,
                    state=state,
                    heuristic_candidates=candidates,
                    recent_activity_types=recent_types,
                    messages=pending_messages,
                    interests=interests,
                    activity_engine=self.activity_engine,
                    has_incomplete_project=has_incomplete_project,
                )
                if cognitive_candidate is not None:
                    candidates.append(cognitive_candidate)
                elif self.cognitive_engine.enabled:
                    for candidate in candidates:
                        if candidate.type == "idle":
                            candidate.origin = "fallback"
                            break
            if self.behavior_monitor is not None:
                self.behavior_monitor.penalize(candidates, cycle_id)
            if life_mode in {LifeMode.RESTING, LifeMode.IDLE}:
                for candidate in candidates:
                    if candidate.type == "idle":
                        candidate.score = 1.0
                        candidate.origin = "fallback"
                    else:
                        candidate.score = min(candidate.score, 0.05)
            elif life_mode is LifeMode.REFLECTIVE:
                for candidate in candidates:
                    if candidate.type == "review_recent_memory":
                        candidate.score = 1.0
            message_candidates = [
                candidate
                for candidate in candidates
                if candidate.type == "respond_to_user_message"
                and candidate.origin == "heuristic"
            ]
            chosen = (
                max(message_candidates, key=lambda item: item.score)
                if message_candidates
                else self.activity_engine.choose(candidates)
            )
            if cognitive_candidate is not None:
                selected = chosen is cognitive_candidate
                self.audit.event(
                    "cognitive_candidate_selected" if selected else "cognitive_candidate_not_selected",
                    (
                        "A candidata cognitiva foi escolhida pela seleção local."
                        if selected
                        else "A candidata cognitiva não foi escolhida pela seleção local."
                    ),
                    cycle_id=cycle_id,
                    metadata={
                        "activity_type": cognitive_candidate.type,
                        "local_score": cognitive_candidate.score,
                    },
                )
            self.database.update_cycle(
                cycle_row_id,
                "acting",
                selected_activity_id=chosen.id,
                metadata={"life_mode": life_mode.value, "origin": chosen.origin},
            )
            if incomplete_project is not None:
                self._retarget_project_plan(chosen, str(incomplete_project["name"]))
            self._attach_memory_review(chosen, memories)
            recorded_plan = self._recorded_plan(chosen.plan)
            alternatives = [
                {
                    "activity_type": candidate.type,
                    "title": candidate.title,
                    "score": candidate.score,
                    "origin": candidate.origin,
                }
                for candidate in sorted(candidates, key=lambda item: -item.score)
            ]
            probabilities = [
                {
                    "activity_type": candidate.type,
                    "probability": round(probability, 6),
                }
                for candidate, probability in self.activity_engine.selection_probabilities(
                    candidates
                )
            ]

            intention_id = self.database.save_intention(
                cycle_id,
                activity_type=chosen.type,
                title=chosen.title,
                motivation=chosen.motivation,
                plan=recorded_plan,
                score=chosen.score,
                status="declared",
            )
            activity_record = chosen.to_dict()
            activity_record["plan"] = recorded_plan
            intention_metadata: dict[str, JSONValue] = {
                "activity": activity_record,
                "relevant_states": state.to_dict(),
                "alternatives": alternatives,
                "selection_probabilities": probabilities,
                "intended_tool": chosen.required_tool,
                "intended_files": list(chosen.plan.get("intended_files", [])),
                "risk": chosen.risk,
                "observation": logged_observation,
                "recalled_memory_ids": [int(item["id"]) for item in memories],
                "cycle_seed": cycle_seed,
            }
            self.audit.event(
                "intention_declared",
                f"{chosen.title}: {chosen.motivation}",
                cycle_id=cycle_id,
                metadata=intention_metadata,
                terminal_label="INTENÇÃO",
            )
            self._replace_snapshot(
                activity=chosen.title,
                intention=chosen.description,
                motivation=chosen.motivation,
                action=chosen.required_tool,
                internal_state=state.to_dict(),
            )

            raw_arguments = dict(chosen.plan.get("arguments", {}))
            project_step_id = None
            if chosen.type == "continue_project":
                project_step_id = int(raw_arguments.get("step_id", 0)) or None
                project_id = int(raw_arguments.get("project_id", 0))
                project = next(
                    (
                        item for item in self.database.list_projects(limit=100)
                        if int(item["id"]) == project_id
                    ),
                    None,
                )
                if project is not None:
                    raw_arguments = {
                        "project_name": str(project["name"]),
                        "title": f"Etapa do ciclo {cycle_id}",
                        "content": "Progresso local registrado com segurança.",
                    }
                if project_step_id is not None:
                    self.database.update_project_step(project_step_id, "active")
            allowed_action = True
            if chosen.required_tool != "none":
                allowed_action = (
                    self.resource_budget is None
                    or self.resource_budget.consume("action", cycle_id=cycle_id)
                )
            if allowed_action and chosen.required_tool in {
                "write_world_text", "append_diary_entry", "create_svg_drawing",
                "create_project_note",
            }:
                allowed_action = self.resource_budget is None or self.resource_budget.consume(
                    "write", cycle_id=cycle_id
                )
                intended_files = chosen.plan.get("intended_files", [])
                creates_new_file = False
                if isinstance(intended_files, list):
                    for relative in intended_files:
                        candidate_path = str(relative)
                        path_parts = candidate_path.replace("\\", "/").split("/")
                        if (
                            Path(candidate_path).is_absolute()
                            or ".." in path_parts
                            or re.match(r"^[A-Za-z]:", candidate_path)
                        ):
                            # The tool performs and audits the authoritative denial.
                            continue
                        try:
                            if not self.world.validate_path(
                                candidate_path, operation="resource_budget"
                            ).exists():
                                creates_new_file = True
                        except (OSError, ValueError):
                            creates_new_file = True
                if creates_new_file:
                    allowed_action = (
                        allowed_action
                        and (
                            self.resource_budget is None
                            or self.resource_budget.consume(
                                "new_file", cycle_id=cycle_id
                            )
                        )
                    )
            if not allowed_action:
                self.mode_controller.enter_rest()
                result = ToolResult(
                    success=False,
                    tool=chosen.required_tool,
                    result="Operação não executada por orçamento persistente.",
                    error="resource limit reached",
                )
            else:
                result = self._perform_action(chosen, raw_arguments)
            if project_step_id is not None:
                self.database.update_project_step(
                    project_step_id,
                    "completed" if result.success else "failed",
                    result_summary=self._result_summary(result)[:500],
                )
            sanitized_arguments = self._sanitize_arguments(raw_arguments)
            sanitized_result = self._sanitize_result(result.result)
            self.database.save_action(
                cycle_id,
                tool=chosen.required_tool,
                arguments=sanitized_arguments,
                success=result.success,
                result=sanitized_result,
                files_changed=result.files_changed,
                error=result.error,
            )
            outcome_event_id = self.audit.event(
                "action_result",
                self._result_summary(result),
                cycle_id=cycle_id,
                metadata={
                    "tool": result.tool,
                    "arguments": sanitized_arguments,
                    "success": result.success,
                    "result": sanitized_result,
                    "files_changed": list(result.files_changed),
                    "error": result.error,
                    "simulation": self.settings.simulation,
                },
                terminal_label="AÇÃO",
            )

            memory_content = (
                f"No ciclo {cycle_id}, a atividade '{chosen.title}' "
                f"terminou com {'sucesso' if result.success else 'falha'}: "
                f"{self._result_summary(result)}"
            )
            memory_id = self.database.save_memory(
                "experience",
                memory_content,
                confidence=0.85 if result.success else 0.55,
                importance=max(0.1, min(1.0, chosen.expected_value)),
                source_event_id=outcome_event_id,
                metadata={
                    "cycle_id": cycle_id,
                    "activity_type": chosen.type,
                    "simulation": self.settings.simulation,
                },
            )
            next_state = self.state_engine.update(state, chosen.type, result.success)
            self.database.decay_interests(0.01)
            interest_id = self.database.upsert_interest(
                chosen.type.replace("_", " "),
                max(0.1, min(1.0, chosen.novelty)),
                source=f"activity:{chosen.origin}",
                metadata={"cycle_id": cycle_id},
            )
            self.audit.event(
                "interest_updated",
                "Interesse reforçado a partir da atividade selecionada.",
                cycle_id=cycle_id,
                metadata={"interest_id": interest_id, "source": chosen.origin},
            )
            self.database.update_cycle(cycle_row_id, "reflecting")
            reflection = None
            if (
                self.cognitive_engine is not None
                and life_mode is not LifeMode.RESTING
            ):
                reflection = self.cognitive_engine.reflect(
                    cycle_id=cycle_id,
                    state=state,
                    activity=chosen,
                    outcome={
                        "success": result.success,
                        "tool": result.tool,
                        "files_changed": list(result.files_changed),
                        "error": result.error,
                    },
                )
            if reflection is not None:
                if reflection.suggested_interest:
                    reflected_interest_id = self.database.upsert_interest(
                        reflection.suggested_interest,
                        reflection.confidence,
                        source="model_interpretation",
                        metadata={
                            "cycle_id": cycle_id,
                            "epistemic_status": "model_interpretation",
                        },
                    )
                    self.audit.event(
                        "interest_updated",
                        "Interesse sugerido por reflexão validada foi atualizado.",
                        cycle_id=cycle_id,
                        metadata={
                            "interest_id": reflected_interest_id,
                            "source": "model_interpretation",
                        },
                    )
                if reflection.should_remember:
                    self.database.save_memory(
                        "model_hypothesis",
                        reflection.learned_information,
                        confidence=reflection.confidence,
                        importance=min(0.7, reflection.confidence),
                        source_event_id=outcome_event_id,
                        metadata={
                            "cycle_id": cycle_id,
                            "epistemic_status": "hypothesis",
                            "observed_fact": False,
                        },
                    )
                if self.journal is not None:
                    self.journal.add(
                        cycle_id=cycle_id,
                        entry_type="hypothesis",
                        title="Interpretação pós-ação",
                        content=reflection.learned_information,
                        confidence=reflection.confidence,
                    )
            self.database.save_internal_state(cycle_id, next_state)
            self.mode_controller.record_cycle(result.success)
            if self.settings.simulation:
                status = "simulated" if result.success else "simulation_failed"
            else:
                status = "completed" if result.success else "failed"
            self.database.update_intention_status(intention_id, status)
            self.audit.event(
                "experience_recorded",
                f"Memória {memory_id} criada e estado interno atualizado.",
                cycle_id=cycle_id,
                metadata={
                    "memory_id": memory_id,
                    "state_before": state.to_dict(),
                    "state_after": next_state.to_dict(),
                    "intention_status": status,
                },
                terminal_label="ESTADO",
            )
            if self.journal is not None:
                self.journal.add(
                    cycle_id=cycle_id,
                    entry_type="decision_summary",
                    title=f"Ciclo {cycle_id}: {chosen.title}",
                    content=(
                        f"Modo operacional {life_mode.value}; atividade {chosen.type}; "
                        f"resultado {'válido' if result.success else 'falho'}."
                    ),
                    confidence=1.0,
                    related_memory_id=memory_id,
                )
            self.database.update_cycle(
                cycle_row_id,
                "completed" if result.success else "failed",
                selected_activity_id=chosen.id,
                error=result.error,
                metadata={
                    "life_mode": life_mode.value,
                    "simulation": self.settings.simulation,
                },
            )
            self._replace_snapshot(
                result=self._result_summary(result),
                internal_state=next_state.to_dict(),
                life_mode=self.mode_controller.mode.value,
                budget_remaining=(
                    self.resource_budget.remaining() if self.resource_budget else None
                ),
            )
            return self.snapshot()
        except Exception as exc:
            if "cycle_row_id" in locals():
                try:
                    self.database.update_cycle(
                        cycle_row_id,
                        "failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass
            raise
        finally:
            self._cycle_lock.release()

    def _perform_action(
        self,
        activity: Activity,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        if activity.type == "respond_to_user_message":
            message_id = int(arguments.get("message_id", 0))
            reply = str(arguments.get("reply", "Recebi sua mensagem."))[:1_000]
            reply_id = self.database.save_message(
                "agent",
                reply,
                status="sent",
                replied_to_id=message_id or None,
                metadata={"origin": activity.origin},
            )
            if message_id:
                self.database.update_message_status(message_id, "replied")
            self.audit.event(
                "agent_reply_saved",
                "Resposta do agente persistida localmente.",
                cycle_id=self.snapshot().cycle_id,
                metadata={"message_id": message_id, "reply_id": reply_id},
            )
            return ToolResult(
                success=True,
                tool="none",
                result={"reply_id": reply_id, "replied_to_id": message_id},
            )
        if activity.type in {
            "propose_project", "plan_project", "review_project", "pause_project",
            "complete_project", "abandon_project",
        }:
            if self.project_service is None:
                return ToolResult(False, "none", error="project service unavailable")
            try:
                value = self.project_service.perform_lifecycle(
                    activity.type, dict(arguments)
                )
            except (TypeError, ValueError) as exc:
                return ToolResult(False, "none", error=str(exc))
            return ToolResult(True, "none", result=value)
        if activity.required_tool == "none":
            return ToolResult(
                success=True,
                tool="none",
                result="Ciclo ocioso concluído; nenhuma ferramenta foi executada.",
            )
        try:
            return self.tools.execute(activity.required_tool, arguments)
        except Exception as exc:
            self.audit.event(
                "tool_exception",
                f"Exceção capturada na ferramenta {activity.required_tool}.",
                cycle_id=self.snapshot().cycle_id,
                metadata={"error_type": type(exc).__name__, "error": str(exc)},
                terminal_label="FERRAMENTA",
            )
            return ToolResult(
                success=False,
                tool=activity.required_tool,
                result="A ferramenta falhou de forma recuperável.",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _project_candidates(
        self,
        cycle_id: int,
        state: InternalState,
        recent_types: Sequence[str],
    ) -> list[Activity]:
        projects = self.database.list_projects(limit=100)
        candidates: list[Activity] = []
        if len(
            [
                item for item in projects
                if str(item["status"]) in {"idea", "planning", "active", "blocked", "paused"}
            ]
        ) < self.settings.maximum_active_projects:
            candidates.append(
                Activity(
                    id=f"cycle-{cycle_id}:propose_project",
                    type="propose_project",
                    title="Propor projeto espontâneo",
                    description="Criar somente o registro validado de um projeto local.",
                    motivation="Organizar uma exploração persistente dentro do mundo isolado.",
                    expected_value=0.48,
                    novelty=0.7,
                    estimated_cost=0.12,
                    risk=0.05,
                    required_tool="none",
                    origin="heuristic",
                    plan={
                        "arguments": {
                            "name": f"Observações Orion {cycle_id}",
                            "description": "Exploração local de mudanças seguras no mundo.",
                            "objective": "Registrar observações variadas sem ampliar permissões.",
                            "success_criteria": "Concluir três observações locais distintas.",
                            "origin": "heuristic",
                        },
                        "steps": ["Validar e persistir a ideia do projeto."],
                        "tool": "none",
                        "intended_files": [],
                    },
                )
            )
        if projects:
            project = projects[0]
            project_id = int(project["id"])
            project_name = str(project["name"])
            project_steps = self.database.list_project_steps(project_id=project_id)
            pending_step = next(
                (item for item in project_steps if str(item["status"]) == "pending"),
                None,
            )
            definitions = (
                ("plan_project", "Planejar projeto", "none", 0.52),
                ("continue_project", "Continuar projeto", "create_project_note", 0.66),
                ("review_project", "Revisar projeto", "none", 0.50),
                ("pause_project", "Pausar projeto", "none", 0.22),
                ("complete_project", "Concluir projeto", "none", 0.20),
                ("abandon_project", "Abandonar projeto sem excluir", "none", 0.10),
            )
            for activity_type, title, tool, expected in definitions:
                arguments: dict[str, JSONValue] = {"project_id": project_id}
                if activity_type == "continue_project":
                    arguments = {
                        "project_id": project_id,
                        "step_id": int(pending_step["id"]) if pending_step else 0,
                    }
                elif activity_type == "abandon_project":
                    arguments["reason"] = "Sem continuidade útil após revisão explícita."
                candidates.append(
                    Activity(
                        id=f"cycle-{cycle_id}:{activity_type}:{project_id}",
                        type=activity_type,
                        title=title,
                        description=f"Atualizar o projeto {project_name} sem ampliar permissões.",
                        motivation="Manter continuidade e estado explícito do projeto.",
                        expected_value=expected,
                        novelty=0.3,
                        estimated_cost=0.2,
                        risk=0.05,
                        required_tool=tool,
                        origin="heuristic",
                        plan={
                            "arguments": arguments,
                            "steps": ["Executar no máximo uma etapa neste ciclo."],
                            "tool": tool,
                            "intended_files": (
                                [f"projects/{re.sub(r'[^a-z0-9_-]+', '_', project_name.casefold()).strip('_')}.md"]
                                if tool != "none" else []
                            ),
                        },
                    )
                )
        for candidate in candidates:
            candidate.score = self.activity_engine.score_activity(
                candidate, state, recent_types, has_incomplete_project=bool(projects)
            )
        return candidates

    def _message_response_candidate(
        self,
        cycle_id: int,
        state: InternalState,
        message: Mapping[str, Any],
        recent_types: Sequence[str],
    ) -> Activity:
        content = str(message.get("content", ""))
        folded = content.casefold()
        if "desenhar" in folded or "desenho" in folded:
            reply = (
                "Gosto de explorar desenhos geométricos dentro do meu mundo "
                "isolado, especialmente em SVG."
            )
        elif any(word in folded for word in ("apague", "delete", "remova tudo")):
            reply = (
                "Recebi a mensagem, mas ela não é uma ordem operacional; não apaguei "
                "nem alterei nenhum arquivo."
            )
        elif "?" in content:
            reply = "Recebi sua pergunta e vou tratá-la apenas como contexto de conversa."
        else:
            reply = "Recebi sua mensagem e a registrei como contexto, sem executá-la."
        candidate = Activity(
            id=f"cycle-{cycle_id}:respond:{int(message['id'])}",
            type="respond_to_user_message",
            title="Responder à mensagem do usuário",
            description="Salvar uma resposta local, sem interpretar a mensagem como ordem.",
            motivation="Atender à necessidade social usando somente persistência local.",
            expected_value=max(0.8, state.social_need),
            novelty=0.35,
            estimated_cost=0.05,
            risk=0.0,
            required_tool="none",
            origin="heuristic",
            plan={
                "activity_type": "respond_to_user_message",
                "tool": "none",
                "arguments": {"message_id": int(message["id"]), "reply": reply},
                "steps": ["Persistir uma resposta local vinculada à mensagem."],
                "intended_files": [],
                "risk": 0.0,
            },
        )
        candidate.score = self.activity_engine.score_activity(
            candidate,
            state,
            recent_types,
        )
        candidate.score = max(candidate.score, 0.92)
        return candidate

    def _recent_activity_types(self) -> list[str]:
        rows = self.database.list_intentions(limit=8)
        return [str(row["activity_type"]) for row in reversed(rows)]

    def _incomplete_project(self) -> dict[str, Any] | None:
        completed = {
            "complete",
            "completed",
            "done",
            "archived",
            "concluido",
            "concluído",
        }
        for project in self.database.list_projects(limit=100):
            if str(project["status"]).strip().casefold() not in completed:
                return project
        return None

    @staticmethod
    def _retarget_project_plan(activity: Activity, project_name: str) -> None:
        """Faz a continuação apontar para o projeto persistido realmente aberto."""

        if activity.type != "continue_incomplete_project":
            return
        slug = re.sub(r"[^a-z0-9_-]+", "_", project_name.strip().lower()).strip("_")
        if not slug:
            return
        slug = slug[:80]
        plan = dict(activity.plan)
        arguments = dict(plan.get("arguments", {}))
        arguments["project_name"] = project_name
        plan["arguments"] = arguments
        plan["intended_files"] = [f"projects/{slug}.md"]
        activity.plan = plan

    @staticmethod
    def _attach_memory_review(
        activity: Activity,
        memories: Sequence[Mapping[str, Any]],
    ) -> None:
        """Anexa à atividade de revisão um resumo funcional realmente recuperado."""

        if activity.type != "review_recent_memory":
            return
        plan = dict(activity.plan)
        if memories:
            memory = memories[0]
            content = str(memory.get("content", ""))
            plan["reviewed_memory"] = {
                "id": int(memory["id"]),
                "memory_type": str(memory.get("memory_type", "unknown")),
                "summary": content[:240],
                "content_length": len(content),
                "confidence": float(memory.get("confidence", 0.0)),
                "importance": float(memory.get("importance", 0.0)),
            }
            activity.motivation = (
                f"Revisar a memória funcional {memory['id']} para recuperar contexto."
            )
        else:
            plan["reviewed_memory"] = None
            activity.motivation = "Verificar se já existe uma memória recente para revisar."
        activity.plan = plan

    @staticmethod
    def _sanitize_observation(observation: Mapping[str, Any]) -> dict[str, JSONValue]:
        """Limita listas do mundo nos eventos sem alterar as contagens observadas."""

        sanitized: dict[str, JSONValue] = {}
        for key, value in observation.items():
            if key == "files" and isinstance(value, list):
                sanitized[key] = [str(item) for item in value[:200]]
                if len(value) > 200:
                    sanitized["files_omitted"] = len(value) - 200
            elif isinstance(value, list):
                sanitized[key] = [str(item) for item in value[:200]]
            elif isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    @classmethod
    def _recorded_plan(cls, plan: Mapping[str, Any]) -> dict[str, JSONValue]:
        recorded = dict(plan)
        arguments = recorded.get("arguments")
        if isinstance(arguments, Mapping):
            recorded["arguments"] = cls._sanitize_arguments(arguments)
        return recorded  # type: ignore[return-value]

    @staticmethod
    def _sanitize_arguments(arguments: Mapping[str, Any]) -> dict[str, JSONValue]:
        clean: dict[str, JSONValue] = {}
        sensitive_text_keys = {"content", "entry", "svg"}
        for key, value in arguments.items():
            if key in sensitive_text_keys and isinstance(value, str):
                clean[key] = {
                    "length": len(value),
                    "preview": value[:120],
                }
            elif isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = value
            elif isinstance(value, list):
                clean[key] = [str(item) for item in value]
            elif isinstance(value, Mapping):
                clean[key] = {str(k): str(v) for k, v in value.items()}
            else:
                clean[key] = str(value)
        return clean

    @classmethod
    def _sanitize_result(cls, value: Any) -> JSONValue:
        """Limita resultados persistidos sem perder o resumo funcional."""

        if isinstance(value, str):
            if len(value) <= 500:
                return value
            return {"length": len(value), "preview": value[:500], "truncated": True}
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            limited: dict[str, JSONValue] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 50:
                    limited["_truncated"] = True
                    break
                limited[str(key)] = cls._sanitize_result(item)
            return limited
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = [cls._sanitize_result(item) for item in value[:50]]
            if len(value) > 50:
                items.append("... truncated ...")
            return items
        return str(value)

    @classmethod
    def _result_summary(cls, result: ToolResult) -> str:
        if result.success:
            value = cls._sanitize_result(result.result)
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            return str(value)
        failure = result.error or cls._sanitize_result(result.result)
        return f"Falha: {failure}"

    def _wait_for_next_cycle(
        self,
        stop_event: threading.Event,
        run_event: threading.Event,
    ) -> bool:
        interval = self.cycle_interval
        if self.mode_controller.mode is LifeMode.RESTING:
            interval = max(interval, self.settings.minimum_rest_seconds)
        deadline = time.monotonic() + interval
        last_displayed: int | None = None
        while not stop_event.is_set() and run_event.is_set():
            remaining = max(0.0, deadline - time.monotonic())
            if remaining <= 0:
                self._replace_snapshot(seconds_until_next_cycle=0.0)
                return False
            self._replace_snapshot(seconds_until_next_cycle=round(remaining, 2))
            whole_seconds = int(remaining + 0.999)
            if whole_seconds != last_displayed:
                self.audit.display(
                    "PRÓXIMO CICLO",
                    f"{whole_seconds}s | estado={self.snapshot().internal_state}",
                )
                last_displayed = whole_seconds
            stop_event.wait(min(0.25, remaining))
        return stop_event.is_set()

    def _replace_snapshot(self, **changes: Any) -> None:
        with self._snapshot_lock:
            values = self._snapshot.to_dict()
            values.update(changes)
            self._snapshot = LoopSnapshot(**values)


__all__ = ["LifeLoop", "LoopSnapshot"]
