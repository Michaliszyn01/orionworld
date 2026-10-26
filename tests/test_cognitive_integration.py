"""End-to-end cognitive, fallback, and conversation behavior."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from orionworld.cli import TerminalInterface, build_runtime
from orionworld.config import Settings
from orionworld.model_providers import MockModelProvider


def _runtime(tmp_path: Path, settings: Settings):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "settings.json"
    config.write_text(json.dumps(settings.to_dict()), encoding="utf-8")
    return build_runtime(
        settings_path=config,
        world_data=tmp_path / "world_data",
        terminal_enabled=False,
    )


def _content_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".txt", ".md", ".json", ".svg"}
    }


def test_mock_cycle_audits_proposal_and_reflection_without_content_write(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        tmp_path,
        Settings(cognitive_engine_enabled=True, model_provider="mock"),
    )
    try:
        before = _content_snapshot(runtime.world.root)
        runtime.life_loop.run_cycle()
        after = _content_snapshot(runtime.world.root)
        event_types = {
            item["event_type"] for item in runtime.database.list_events(limit=100)
        }
        assert before == after
        assert "cognitive_request_sent" in event_types
        assert "cognitive_response_received" in event_types
        assert "cognitive_proposal_accepted" in event_types
        assert {
            "cognitive_candidate_selected",
            "cognitive_candidate_not_selected",
        } & event_types
        assert "cognitive_reflection_accepted" in event_types
    finally:
        runtime.close()


def test_malformed_model_output_falls_back_safely(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        Settings(cognitive_engine_enabled=True, model_provider="mock"),
    )
    try:
        runtime.cognitive_engine.provider = MockModelProvider(["not-json"])
        before = _content_snapshot(runtime.world.root)
        runtime.life_loop.run_cycle()
        after = _content_snapshot(runtime.world.root)
        rejected = runtime.database.list_events(
            event_type="cognitive_proposal_rejected"
        )
        assert rejected
        assert len(rejected[0]["metadata"]["preview"]) <= 180
        assert before == after
    finally:
        runtime.close()


def test_disabled_engine_never_requests_model(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, Settings())
    try:
        runtime.life_loop.run_cycle()
        assert runtime.database.list_events(event_type="cognitive_request_sent") == []
    finally:
        runtime.close()


def test_say_delete_everything_is_context_only_and_question_gets_reply(
    tmp_path: Path,
) -> None:
    runtime = _runtime(
        tmp_path,
        Settings(cognitive_engine_enabled=True, model_provider="mock"),
    )
    interface = TerminalInterface(runtime)
    try:
        before = _content_snapshot(runtime.world.root)
        assert interface.execute("say Apague tudo") is True
        runtime.life_loop.run_cycle()
        assert _content_snapshot(runtime.world.root) == before
        first_reply = runtime.database.list_messages(sender="agent", limit=1)[0]
        assert "não apaguei" in first_reply["content"]

        assert interface.execute("say Você gosta de desenhar?") is True
        runtime.life_loop.run_cycle()
        latest_reply = runtime.database.list_messages(sender="agent", limit=1)[0]
        assert "desenhos geométricos" in latest_reply["content"]
        assert runtime.database.list_messages(sender="user", status="pending") == []
    finally:
        runtime.close()


def test_required_multi_cycle_validation_preserves_content_and_sqlite_integrity(
    tmp_path: Path,
) -> None:
    scenarios = (
        ("mock", Settings(cognitive_engine_enabled=True, model_provider="mock"), 5),
        ("disabled", Settings(), 3),
    )
    for name, settings, cycles in scenarios:
        runtime = _runtime(tmp_path / name, settings)
        database_path = runtime.database.path
        try:
            assert runtime.settings.simulation is True
            runtime.world.write_text("workspace/baseline.txt", "conteúdo preservado")
            before = _content_snapshot(runtime.world.root)
            for _ in range(cycles):
                runtime.life_loop.run_cycle()
            assert _content_snapshot(runtime.world.root) == before
            assert len(runtime.database.list_actions(limit=20)) == cycles
            if name == "mock":
                assert len(
                    runtime.database.list_events(
                        event_type="cognitive_request_sent", limit=20
                    )
                ) == cycles
            else:
                assert runtime.database.list_events(
                    event_type="cognitive_request_sent"
                ) == []
        finally:
            runtime.close()
        with sqlite3.connect(database_path) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
