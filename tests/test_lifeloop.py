"""Integration tests for the persistent observe-decide-act life loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from orionworld.audit import AuditLogger
from orionworld.config import Settings
from orionworld.database import Database
from orionworld.lifeloop import LifeLoop
from orionworld.models import Activity, InternalState, ToolResult
from orionworld.security import SecurityPolicy
from orionworld.state import StateEngine
from orionworld.tools import ToolRegistry
from orionworld.world import WorldSandbox


class FixedActivityEngine:
    """Small deterministic planner implementing the LifeLoop planner contract."""

    def __init__(self, *, tool: str = "write_world_text") -> None:
        self.tool = tool
        self.last_seed: int | None = None

    def reset(self, seed: int | None = None) -> None:
        self.last_seed = seed

    def generate(
        self,
        cycle_id: int,
        state: InternalState,
        recent_activity_types: Sequence[str] = (),
        has_incomplete_project: bool = False,
    ) -> list[Activity]:
        del state, recent_activity_types, has_incomplete_project
        intended_path = f"workspace/cycle-{cycle_id}.txt"
        return [
            Activity(
                id=f"cycle-{cycle_id}:experiment",
                type="create_text_experiment",
                title=f"Experimento {cycle_id}",
                description="Registrar uma observação curta no mundo isolado.",
                motivation="Testar o ciclo persistente de forma observável.",
                expected_value=0.8,
                novelty=0.7,
                estimated_cost=0.2,
                risk=0.05,
                required_tool=self.tool,
                score=0.75,
                plan={
                    "activity_type": "create_text_experiment",
                    "tool": self.tool,
                    "arguments": {
                        "path": intended_path,
                        "content": f"conteúdo funcional do ciclo {cycle_id}",
                    },
                    "steps": ["Registrar a intenção.", "Executar a ferramenta."],
                    "intended_files": [intended_path],
                },
            )
        ]

    @staticmethod
    def choose(candidates: Sequence[Activity]) -> Activity:
        return candidates[0]

    @staticmethod
    def selection_probabilities(
        candidates: Sequence[Activity],
    ) -> list[tuple[Activity, float]]:
        return [(candidates[0], 1.0)]


class ScriptedTools:
    def __init__(self, outcomes: Sequence[ToolResult | BaseException]) -> None:
        self.simulation = False
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((tool_name, dict(arguments)))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class UnsafeActivityEngine(FixedActivityEngine):
    def generate(self, *args: object, **kwargs: object) -> list[Activity]:
        candidates = super().generate(*args, **kwargs)  # type: ignore[arg-type]
        candidates[0].plan["arguments"] = {
            "path": "../escape.txt",
            "content": "must remain blocked",
        }
        candidates[0].plan["intended_files"] = ["../escape.txt"]
        return candidates


class ReviewActivityEngine(FixedActivityEngine):
    def generate(self, *args: object, **kwargs: object) -> list[Activity]:
        candidates = super().generate(*args, **kwargs)  # type: ignore[arg-type]
        activity = candidates[0]
        activity.type = "review_recent_memory"
        activity.title = "Revisar memória recente"
        activity.required_tool = "list_world_files"
        activity.plan = {
            "activity_type": activity.type,
            "tool": activity.required_tool,
            "arguments": {"area": "diary"},
            "steps": ["Revisar memória recuperada."],
            "intended_files": [],
        }
        return candidates


class ContinueActivityEngine(FixedActivityEngine):
    def generate(self, *args: object, **kwargs: object) -> list[Activity]:
        candidates = super().generate(*args, **kwargs)  # type: ignore[arg-type]
        activity = candidates[0]
        activity.type = "continue_incomplete_project"
        activity.title = "Continuar projeto incompleto"
        activity.required_tool = "create_project_note"
        activity.plan = {
            "activity_type": activity.type,
            "tool": activity.required_tool,
            "arguments": {
                "project_name": "placeholder",
                "title": "Próximo passo",
                "content": "Continuidade segura.",
            },
            "steps": ["Continuar o projeto persistido."],
            "intended_files": ["projects/placeholder.md"],
        }
        return candidates


def _settings(*, simulation: bool) -> Settings:
    return Settings(
        simulation=simulation,
        cycle_interval_seconds=0.01,
        random_seed=73,
    )


def _build_loop(
    tmp_path: Path,
    *,
    settings: Settings,
    tools: object | None = None,
) -> tuple[LifeLoop, Database, WorldSandbox]:
    world_root = tmp_path / "world_data"
    database = Database(world_root / "orion_world.sqlite3")
    audit = AuditLogger(database, terminal_enabled=False)
    policy = SecurityPolicy(audit_callback=audit.blocked)
    world = WorldSandbox(world_root, settings=settings, security_policy=policy)
    selected_tools = tools or ToolRegistry(world, simulation=settings.simulation)
    loop = LifeLoop(
        settings=settings,
        database=database,
        world=world,
        tools=selected_tools,  # type: ignore[arg-type]
        activity_engine=FixedActivityEngine(),  # type: ignore[arg-type]
        state_engine=StateEngine(seed=settings.random_seed),
        audit=audit,
    )
    return loop, database, world


def _content_snapshot(world: WorldSandbox) -> dict[str, bytes]:
    return {
        relative: (world.root / relative).read_bytes()
        for relative in world.list_files()
    }


def test_simulation_persists_complete_cycle_without_changing_content(
    tmp_path: Path,
) -> None:
    settings = _settings(simulation=True)
    loop, database, world = _build_loop(
        tmp_path,
        settings=settings,
    )
    try:
        world.write_text("workspace/existing.txt", "conteúdo que deve permanecer")
        content_before = _content_snapshot(world)

        snapshot = loop.run_cycle()

        assert snapshot.cycle_id == 1
        assert snapshot.result is not None
        assert json.loads(snapshot.result) == {
            "bytes": len("conteúdo funcional do ciclo 1".encode("utf-8")),
            "intended_path": "workspace/cycle-1.txt",
            "simulation": True,
        }
        assert _content_snapshot(world) == content_before
        assert not (world.root / "workspace" / "cycle-1.txt").exists()

        intentions = database.list_intentions(cycle_id=1)
        actions = database.list_actions(cycle_id=1)
        memories = database.list_memories(memory_type="experience")
        events = {
            row["event_type"]: row for row in database.list_events(cycle_id=1)
        }

        assert len(intentions) == len(actions) == len(memories) == 1
        assert intentions[0]["status"] == "simulated"
        assert intentions[0]["plan"]["arguments"] == {
            "path": "workspace/cycle-1.txt",
            "content": {
                "length": len("conteúdo funcional do ciclo 1"),
                "preview": "conteúdo funcional do ciclo 1",
            },
        }
        assert actions[0]["tool"] == "write_world_text"
        assert actions[0]["success"] is True
        assert actions[0]["files_changed"] == []
        assert actions[0]["error"] is None
        assert actions[0]["arguments"] == intentions[0]["plan"]["arguments"]
        assert memories[0]["source_event_id"] == events["action_result"]["id"]
        assert memories[0]["metadata"] == {
            "activity_type": "create_text_experiment",
            "cycle_id": 1,
            "simulation": True,
        }
        assert database.latest_internal_state() is not None

        intention_metadata = events["intention_declared"]["metadata"]
        assert set(intention_metadata) == {
            "activity",
            "relevant_states",
            "alternatives",
            "selection_probabilities",
            "intended_tool",
            "intended_files",
            "risk",
            "observation",
            "recalled_memory_ids",
            "cycle_seed",
        }
        assert intention_metadata["cycle_seed"] == settings.random_seed + 1
        assert intention_metadata["intended_tool"] == "write_world_text"
        assert intention_metadata["intended_files"] == ["workspace/cycle-1.txt"]
        assert intention_metadata["selection_probabilities"] == [
            {"activity_type": "create_text_experiment", "probability": 1.0}
        ]
        observation = intention_metadata["observation"]
        assert observation["areas"] == list(world.areas)
        assert observation["files"] == ["workspace/existing.txt"]
        assert observation["file_count"] == 1
        # The SQLite persistence files also live below world_data but are not
        # agent-visible content; their bytes may vary by SQLite version.
        assert observation["total_size_bytes"] >= len(
            "conteúdo que deve permanecer".encode("utf-8")
        )
        assert events["environment_observed"]["metadata"] == observation

        action_metadata = events["action_result"]["metadata"]
        assert set(action_metadata) == {
            "tool",
            "arguments",
            "success",
            "result",
            "files_changed",
            "error",
            "simulation",
        }
        assert action_metadata["arguments"] == actions[0]["arguments"]
        assert action_metadata["success"] is True
        assert action_metadata["files_changed"] == []
        assert action_metadata["simulation"] is True
        assert action_metadata["result"] == json.loads(actions[0]["result"])
        assert set(events) == {
            "cycle_started",
            "environment_observed",
            "intention_declared",
            "action_result",
                "experience_recorded",
                "interest_updated",
            }
    finally:
        database.close()


def test_tool_failure_and_exception_are_recoverable_and_next_cycle_runs(
    tmp_path: Path,
) -> None:
    settings = _settings(simulation=False)
    tools = ScriptedTools(
        [
            ToolResult(
                success=False,
                tool="write_world_text",
                result="falha controlada",
                error="falha esperada",
            ),
            RuntimeError("exceção controlada"),
            ToolResult(
                success=True,
                tool="write_world_text",
                result="recuperado no ciclo seguinte",
            ),
        ]
    )
    loop, database, world = _build_loop(tmp_path, settings=settings, tools=tools)
    content_before = _content_snapshot(world)
    try:
        snapshots = [loop.run_cycle() for _ in range(3)]

        assert [item.cycle_id for item in snapshots] == [1, 2, 3]
        assert snapshots[0].result == "Falha: falha esperada"
        assert snapshots[1].result == "Falha: RuntimeError: exceção controlada"
        assert snapshots[2].result == "recuperado no ciclo seguinte"
        assert len(tools.calls) == 3
        assert _content_snapshot(world) == content_before

        actions = sorted(database.list_actions(), key=lambda row: row["cycle_id"])
        assert [row["success"] for row in actions] == [False, False, True]
        assert [row["error"] for row in actions] == [
            "falha esperada",
            "RuntimeError: exceção controlada",
            None,
        ]
        intentions = sorted(
            database.list_intentions(), key=lambda row: row["cycle_id"]
        )
        assert [row["status"] for row in intentions] == [
            "failed",
            "failed",
            "completed",
        ]
        assert len(database.list_memories(memory_type="experience")) == 3
        assert database.latest_internal_state() is not None
        exception_events = database.list_events(event_type="tool_exception")
        assert len(exception_events) == 1
        assert exception_events[0]["cycle_id"] == 2
        assert exception_events[0]["metadata"] == {
            "error": "exceção controlada",
            "error_type": "RuntimeError",
        }
        assert len(database.list_events(event_type="action_result")) == 3
    finally:
        database.close()


def test_simulation_still_blocks_unsafe_plan_and_marks_failure(tmp_path: Path) -> None:
    settings = _settings(simulation=True)
    loop, database, world = _build_loop(tmp_path, settings=settings)
    loop.activity_engine = UnsafeActivityEngine()  # type: ignore[assignment]
    try:
        snapshot = loop.run_cycle()

        assert snapshot.result is not None and snapshot.result.startswith("Falha:")
        assert database.list_actions(cycle_id=1)[0]["success"] is False
        assert database.list_intentions(cycle_id=1)[0]["status"] == (
            "simulation_failed"
        )
        assert len(database.list_events(event_type="security_block")) == 1
        assert not (tmp_path / "escape.txt").exists()
        assert world.list_files() == []
    finally:
        database.close()


def test_review_activity_attaches_a_real_recalled_memory_summary(
    tmp_path: Path,
) -> None:
    settings = _settings(simulation=True)
    loop, database, _ = _build_loop(tmp_path, settings=settings)
    loop.activity_engine = ReviewActivityEngine()  # type: ignore[assignment]
    memory_id = database.save_memory(
        "experience",
        "Resumo funcional recente para orientar a revisão.",
        confidence=0.8,
        importance=0.9,
    )
    try:
        loop.run_cycle()

        reviewed = database.list_intentions(cycle_id=1)[0]["plan"][
            "reviewed_memory"
        ]
        assert reviewed["id"] == memory_id
        assert reviewed["memory_type"] == "experience"
        assert reviewed["summary"] == (
            "Resumo funcional recente para orientar a revisão."
        )
        assert reviewed["content_length"] == len(reviewed["summary"])
    finally:
        database.close()


def test_continue_activity_targets_the_actual_incomplete_project(
    tmp_path: Path,
) -> None:
    settings = _settings(simulation=True)
    loop, database, _ = _build_loop(tmp_path, settings=settings)
    loop.activity_engine = ContinueActivityEngine()  # type: ignore[assignment]
    database.create_project(
        "Projeto Real",
        "Projeto persistido que ainda está em andamento.",
        status="incomplete",
        metadata={"path": "projects/projeto_real.md"},
    )
    try:
        loop.run_cycle()

        plan = database.list_intentions(cycle_id=1)[0]["plan"]
        assert plan["arguments"]["project_name"] == "Projeto Real"
        assert plan["intended_files"] == ["projects/projeto_real.md"]
    finally:
        database.close()
