"""Behavioral tests for the persistent SQLite repository and audit log."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3

import pytest

from orionworld.activities import ActivityEngine
from orionworld.audit import AuditLogger
from orionworld.database import Database
from orionworld.models import InternalState, ToolResult


EXPECTED_SCHEMA = {
    "events": (
        "id",
        "timestamp",
        "cycle_id",
        "event_type",
        "content",
        "metadata_json",
    ),
    "memories": (
        "id",
        "timestamp",
        "memory_type",
        "content",
        "confidence",
        "importance",
        "source_event_id",
        "times_recalled",
        "last_recalled_at",
        "metadata_json",
    ),
    "internal_states": ("id", "timestamp", "cycle_id", "state_json"),
    "intentions": (
        "id",
        "timestamp",
        "cycle_id",
        "activity_type",
        "title",
        "motivation",
        "plan_json",
        "score",
        "status",
    ),
    "actions": (
        "id",
        "timestamp",
        "cycle_id",
        "tool",
        "arguments_json",
        "success",
        "result",
        "files_changed_json",
        "error",
    ),
    "projects": (
        "id",
        "name",
        "description",
        "status",
        "created_at",
        "updated_at",
        "metadata_json",
        "origin",
        "current_stage",
        "objective",
        "success_criteria",
        "last_activity_at",
        "priority",
        "progress",
        "abandoned_reason",
    ),
}


def _schema_columns(database_path: Path) -> dict[str, tuple[str, ...]]:
    with sqlite3.connect(database_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?",
                ("table",),
            )
        }
        assert set(EXPECTED_SCHEMA) <= tables
        return {
            table: tuple(
                str(row[1])
                for row in connection.execute(
                    "SELECT * FROM pragma_table_info(?) ORDER BY cid",
                    (table,),
                )
            )
            for table in EXPECTED_SCHEMA
        }


def test_schema_and_structured_records_persist_after_reopen(tmp_path: Path) -> None:
    database_path = tmp_path / "world_data" / "orion_world.sqlite3"
    state = InternalState(
        curiosity=0.71,
        boredom=0.42,
        social_need=0.33,
        completion_pressure=0.64,
        organization_need=0.58,
        confidence=0.69,
        energy=0.77,
        novelty_preference=0.81,
    )
    activity = next(
        item
        for item in ActivityEngine(seed=29).generate(4, state)
        if item.type == "write_diary"
    )
    result = ToolResult(
        success=True,
        tool="append_diary_entry",
        result={"summary": "entry stored"},
        files_changed=["diary/orion_diary.md"],
    )

    with Database(database_path) as database:
        event_id = database.log_event(
            4,
            "observation",
            "Observed one safe file.",
            {"file_count": 1},
        )
        memory_id = database.save_memory(
            "experience",
            "A short functional summary.",
            confidence=0.8,
            importance=0.9,
            source_event_id=event_id,
            metadata={"outcome": "success"},
        )
        state_id = database.save_internal_state(4, state)
        intention_id = database.save_intention(4, activity=activity)
        action_id = database.save_action(
            4,
            arguments={"filename": "orion_diary.md"},
            tool_result=result,
        )
        assert database.update_intention_status(intention_id, "completed")
        assert database.next_cycle_id() == 5
        assert all(
            identifier > 0
            for identifier in (event_id, memory_id, state_id, intention_id, action_id)
        )

    assert _schema_columns(database_path) == EXPECTED_SCHEMA

    with Database(database_path) as reopened:
        events = reopened.list_events(cycle_id=4, event_type="observation")
        intentions = reopened.list_intentions(cycle_id=4, status="completed")
        actions = reopened.list_actions(cycle_id=4, success=True)
        memories = reopened.list_memories(memory_type="experience")

        assert len(events) == 1
        assert events[0]["content"] == "Observed one safe file."
        assert events[0]["metadata"] == {"file_count": 1}

        assert len(intentions) == 1
        assert intentions[0]["activity_type"] == "write_diary"
        assert intentions[0]["motivation"] == activity.motivation
        assert intentions[0]["plan"] == activity.plan
        assert intentions[0]["score"] == activity.score

        assert len(actions) == 1
        assert actions[0]["tool"] == "append_diary_entry"
        assert actions[0]["arguments"] == {"filename": "orion_diary.md"}
        assert actions[0]["success"] is True
        assert json.loads(actions[0]["result"]) == {"summary": "entry stored"}
        assert actions[0]["files_changed"] == ["diary/orion_diary.md"]
        assert actions[0]["error"] is None

        assert len(memories) == 1
        assert memories[0]["source_event_id"] == event_id
        assert memories[0]["metadata"] == {"outcome": "success"}
        assert memories[0]["times_recalled"] == 0

        recalled = reopened.recall_memories(
            "functional",
            memory_type="experience",
            min_importance=0.85,
        )
        assert [item["id"] for item in recalled] == [memory_id]
        assert recalled[0]["times_recalled"] == 1
        assert recalled[0]["last_recalled_at"] is not None
        assert reopened.latest_internal_state() == state
        assert reopened.next_cycle_id() == 5


def test_sql_like_input_is_data_and_cannot_change_queries_or_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "orion_world.sqlite3"
    sql_like = "x' OR 1=1; DROP TABLE events; --"

    with Database(database_path) as database:
        safe_id = database.log_event(8, "safe", "ordinary event")
        hostile_id = database.log_event(
            9,
            sql_like,
            sql_like,
            {"literal": sql_like},
        )
        memory_id = database.save_memory(
            sql_like,
            f"Remember literally: {sql_like}",
            confidence=0.7,
            importance=0.8,
        )
        project_id = database.create_project(sql_like, sql_like)

        assert [item["id"] for item in database.list_events(event_type=sql_like)] == [
            hostile_id
        ]
        assert [item["id"] for item in database.list_events(event_type="safe")] == [
            safe_id
        ]
        assert [item["id"] for item in database.list_memories(memory_type=sql_like)] == [
            memory_id
        ]
        assert [item["id"] for item in database.recall_memories(sql_like)] == [
            memory_id
        ]
        project = database.get_project(sql_like)
        assert project is not None
        assert project["id"] == project_id

        follow_up_id = database.log_event(10, "after_input", "schema still works")
        assert follow_up_id > hostile_id

    assert _schema_columns(database_path) == EXPECTED_SCHEMA


def test_audit_logger_persists_readable_events_without_private_reasoning(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    with Database(tmp_path / "audit.sqlite3") as database:
        logger = AuditLogger(database, stream=stream)
        intention_event = logger.event(
            "intention_log",
            "Plan: inspect files using the allowed listing tool.",
            cycle_id=12,
            metadata={"tool": "list_world_files", "risk": 0.05},
            terminal_label="INTENÇÃO",
        )
        blocked_event = logger.blocked(
            "write_world_text",
            "path traversal is forbidden",
            {"path": "../outside.txt", "category": "path"},
            cycle_id=12,
        )

        records = database.list_events(cycle_id=12)

    assert [record["id"] for record in records] == [blocked_event, intention_event]
    assert records[0]["event_type"] == "security_block"
    assert records[0]["metadata"] == {
        "operation": "write_world_text",
        "reason": "path traversal is forbidden",
        "path": "../outside.txt",
        "category": "path",
    }
    assert records[1]["event_type"] == "intention_log"
    assert records[1]["metadata"] == {"tool": "list_world_files", "risk": 0.05}

    terminal_output = stream.getvalue()
    assert "[INTENÇÃO] Plan: inspect files" in terminal_output
    assert "[BLOQUEIO] Operação bloqueada: write_world_text" in terminal_output
    assert "private reasoning" not in terminal_output.lower()


def test_database_path_must_stay_inside_declared_world_root(tmp_path: Path) -> None:
    world_root = tmp_path / "world_data"
    outside = tmp_path / "outside.sqlite3"

    with pytest.raises(ValueError, match="world_data"):
        Database(outside, allowed_root=world_root)


def test_database_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    world_root = tmp_path / "world_data"
    world_root.mkdir()
    outside = tmp_path / "outside.sqlite3"
    outside.touch()
    linked_database = world_root / "orion_world.sqlite3"
    try:
        linked_database.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable in this environment: {exc}")

    with pytest.raises(ValueError, match="link simbólico"):
        Database(linked_database, allowed_root=world_root)
