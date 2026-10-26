"""Tests for the fixed, local-only terminal command surface."""

from __future__ import annotations

import io
import time
from pathlib import Path

from orionworld.activities import ActivityEngine
from orionworld.audit import AuditLogger
from orionworld.cli import Runtime, TerminalInterface
from orionworld.config import Settings
from orionworld.database import Database
from orionworld.lifeloop import LifeLoop
from orionworld.security import SecurityPolicy
from orionworld.state import StateEngine
from orionworld.supervisor import Supervisor
from orionworld.tools import ToolRegistry
from orionworld.world import WorldSandbox


def _runtime(tmp_path: Path) -> Runtime:
    settings = Settings(
        simulation=True,
        cycle_interval_seconds=0.01,
        random_seed=37,
    )
    world_root = tmp_path / "world_data"
    database = Database(world_root / "orion_world.sqlite3")
    audit = AuditLogger(database, terminal_enabled=False)
    policy = SecurityPolicy(audit_callback=audit.blocked)
    world = WorldSandbox(world_root, settings=settings, security_policy=policy)
    tools = ToolRegistry(world, simulation=True)
    loop = LifeLoop(
        settings=settings,
        database=database,
        world=world,
        tools=tools,
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
    return Runtime(
        settings=settings,
        database=database,
        audit=audit,
        security_policy=policy,
        world=world,
        tools=tools,
        life_loop=loop,
        supervisor=supervisor,
    )


def _execute(interface: TerminalInterface, stream: io.StringIO, command: str) -> str:
    stream.seek(0)
    stream.truncate(0)
    assert interface.execute(command) is True
    return stream.getvalue()


def test_cli_read_commands_report_status_and_persisted_cycle_data(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    stream = io.StringIO()
    interface = TerminalInterface(runtime, stream=stream)
    try:
        runtime.world.write_text("workspace/existing.txt", "visível apenas no mundo")
        runtime.life_loop.run_cycle()

        expected_fragments = {
            "help": "Comandos:",
            "status": '"supervisor": "created"',
            "history": '"event_type": "intention_declared"',
            "intentions": '"status": "simulated"',
            "actions": '"success": true',
            "memories": '"memory_type": "experience"',
            "state": '"curiosity":',
            "files": '"workspace/existing.txt"',
        }
        for command, expected in expected_fragments.items():
            assert expected in _execute(interface, stream, command)

        assert _execute(interface, stream, "   ") == ""
        unknown = _execute(interface, stream, "execute-algo-fora-da-lista")
        assert "Comando desconhecido" in unknown
        assert "Use 'help'" in unknown
    finally:
        runtime.close()


def test_cli_pause_resume_and_stop_commands_control_the_supervisor(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    stream = io.StringIO()
    interface = TerminalInterface(runtime, stream=stream)
    try:
        runtime.supervisor.start()
        deadline = time.monotonic() + 1.0
        while not runtime.database.list_actions() and time.monotonic() < deadline:
            time.sleep(0.002)
        assert runtime.database.list_actions()

        assert _execute(interface, stream, "pause") == "Agente pausado.\n"
        assert _execute(interface, stream, "resume") == "Agente retomado.\n"

        stream.seek(0)
        stream.truncate(0)
        assert interface.execute("stop") is False
        assert stream.getvalue() == "Agente encerrado.\n"
        status = runtime.supervisor.status()
        assert status["supervisor"] == "stopped"
        assert status["thread_alive"] is False
    finally:
        runtime.close()
