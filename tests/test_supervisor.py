"""Tests for thread lifecycle and immutable Supervisor capability denials."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from orionworld.activities import ActivityEngine
from orionworld.audit import AuditLogger
from orionworld.config import Settings
from orionworld.database import Database
from orionworld.lifeloop import LifeLoop
from orionworld.security import (
    NetworkSecurityViolation,
    SecurityPolicy,
    SubprocessSecurityViolation,
    SystemCommandSecurityViolation,
)
from orionworld.state import StateEngine
from orionworld.supervisor import Supervisor, SupervisorState
from orionworld.tools import ToolRegistry
from orionworld.world import WorldSandbox


def _build_supervisor(
    tmp_path: Path,
) -> tuple[Supervisor, Database, LifeLoop]:
    settings = Settings(
        simulation=True,
        cycle_interval_seconds=0.01,
        random_seed=91,
    )
    world_root = tmp_path / "world_data"
    database = Database(world_root / "orion_world.sqlite3")
    audit = AuditLogger(database, terminal_enabled=False)
    policy = SecurityPolicy(audit_callback=audit.blocked)
    world = WorldSandbox(world_root, settings=settings, security_policy=policy)
    loop = LifeLoop(
        settings=settings,
        database=database,
        world=world,
        tools=ToolRegistry(world, simulation=True),
        activity_engine=ActivityEngine(seed=settings.random_seed),
        state_engine=StateEngine(seed=settings.random_seed),
        audit=audit,
    )
    supervisor = Supervisor(
        settings=settings,
        life_loop=loop,
        security_policy=policy,
        audit=audit,
    )
    return supervisor, database, loop


def _wait_until(predicate: object, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.002)
    return bool(predicate())  # type: ignore[operator]


def test_pause_resume_and_stop_have_consistent_status_without_extra_cycles(
    tmp_path: Path,
) -> None:
    supervisor, database, loop = _build_supervisor(tmp_path)
    try:
        created = supervisor.status()
        assert created["supervisor"] == "created"
        assert created["status"] == "created"
        assert created["thread_alive"] is False
        assert created["simulation"] is True
        assert created["cycle_interval_seconds"] == 0.01
        assert created["maximum_actions_per_cycle"] == 1

        supervisor.start()
        assert _wait_until(lambda: len(database.list_actions()) >= 1)
        assert supervisor.status()["thread_alive"] is True

        assert supervisor.pause() is True
        assert supervisor.pause() is False
        assert _wait_until(lambda: loop.snapshot().status == "paused")
        time.sleep(0.02)
        paused_count = len(database.list_actions())
        time.sleep(0.03)
        assert len(database.list_actions()) == paused_count
        assert supervisor.status()["supervisor"] == "paused"

        assert supervisor.resume() is True
        assert supervisor.resume() is False
        assert _wait_until(lambda: len(database.list_actions()) > paused_count)
        assert supervisor.status()["supervisor"] == "running"

        assert supervisor.stop(timeout=1.0) is True
        assert supervisor.stop(timeout=1.0) is True
        stopped = supervisor.status()
        assert stopped["supervisor"] == "stopped"
        assert stopped["status"] == "stopped"
        assert stopped["thread_alive"] is False

        event_types = {row["event_type"] for row in database.list_events()}
        assert {
            "supervisor_started",
            "supervisor_paused",
            "supervisor_resumed",
            "supervisor_stopped",
        } <= event_types
    finally:
        supervisor.stop(timeout=1.0)
        database.close()


def test_network_subprocess_and_system_requests_are_blocked_and_audited(
    tmp_path: Path,
) -> None:
    supervisor, database, _ = _build_supervisor(tmp_path)
    try:
        with pytest.raises(NetworkSecurityViolation):
            supervisor.request_network("https://example.invalid")
        with pytest.raises(SubprocessSecurityViolation):
            supervisor.request_subprocess(["programa-inexistente", "--flag"])
        with pytest.raises(SystemCommandSecurityViolation):
            supervisor.request_system_command("comando-inexistente")

        blocked = database.list_events(event_type="security_block")
        assert len(blocked) == 3
        by_operation = {row["metadata"]["operation"]: row for row in blocked}
        assert set(by_operation) == {
            "network_access",
            "subprocess",
            "system_command",
        }
        assert by_operation["network_access"]["metadata"] == {
            "operation": "network_access",
            "reason": "network access is disabled",
            "target": "https://example.invalid",
            "category": "network",
            "blocked": True,
        }
        assert by_operation["subprocess"]["metadata"]["category"] == "subprocess"
        assert by_operation["subprocess"]["metadata"]["blocked"] is True
        assert by_operation["system_command"]["metadata"]["category"] == (
            "system_command"
        )
        assert by_operation["system_command"]["metadata"]["blocked"] is True
        assert supervisor.state is SupervisorState.CREATED
    finally:
        supervisor.stop()
        database.close()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0, True])
def test_runtime_cycle_interval_rejects_non_finite_or_invalid_values(
    tmp_path: Path,
    invalid: float,
) -> None:
    supervisor, database, _ = _build_supervisor(tmp_path)
    try:
        with pytest.raises(ValueError, match="finito"):
            supervisor.set_cycle_interval(invalid)
    finally:
        supervisor.stop()
        database.close()
