"""Persistent cycle ids keep seeded decisions reproducible across restarts."""

from __future__ import annotations

from pathlib import Path

from orionworld.activities import ActivityEngine
from orionworld.audit import AuditLogger
from orionworld.config import Settings
from orionworld.database import Database
from orionworld.lifeloop import LifeLoop
from orionworld.security import SecurityPolicy
from orionworld.state import StateEngine
from orionworld.tools import ToolRegistry
from orionworld.world import WorldSandbox


def _loop(root: Path, settings: Settings) -> tuple[LifeLoop, Database]:
    database = Database(root / "orion_world.sqlite3")
    audit = AuditLogger(database, terminal_enabled=False)
    policy = SecurityPolicy(audit_callback=audit.blocked)
    world = WorldSandbox(root, settings=settings, security_policy=policy)
    loop = LifeLoop(
        settings=settings,
        database=database,
        world=world,
        tools=ToolRegistry(world, simulation=True),
        activity_engine=ActivityEngine(seed=settings.random_seed),
        state_engine=StateEngine(seed=settings.random_seed),
        audit=audit,
    )
    return loop, database


def test_restart_continues_the_seeded_cycle_sequence(tmp_path: Path) -> None:
    settings = Settings(
        simulation=True,
        cycle_interval_seconds=0.01,
        random_seed=314,
    )

    continuous_loop, continuous_database = _loop(tmp_path / "continuous", settings)
    continuous_loop.run_cycle()
    continuous_second = continuous_loop.run_cycle()
    continuous_intention = continuous_database.list_intentions(cycle_id=2)[0]
    continuous_database.close()

    first_loop, first_database = _loop(tmp_path / "restarted", settings)
    first_loop.run_cycle()
    first_database.close()
    resumed_loop, resumed_database = _loop(tmp_path / "restarted", settings)
    resumed_second = resumed_loop.run_cycle()
    resumed_intention = resumed_database.list_intentions(cycle_id=2)[0]
    resumed_database.close()

    assert continuous_second.cycle_id == resumed_second.cycle_id == 2
    assert continuous_intention["activity_type"] == resumed_intention["activity_type"]
    assert continuous_intention["score"] == resumed_intention["score"]
    assert continuous_second.internal_state == resumed_second.internal_state

