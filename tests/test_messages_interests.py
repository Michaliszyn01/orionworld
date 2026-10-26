"""Persistence tests for conversation and bounded interests."""

from __future__ import annotations

from pathlib import Path

from orionworld.database import Database


def test_message_reply_lifecycle(tmp_path: Path) -> None:
    database = Database(tmp_path / "world_data" / "orion_world.sqlite3")
    try:
        user_id = database.save_message("user", "Você gosta de desenhar?")
        reply_id = database.save_message(
            "agent", "Sim, de forma local.", status="sent", replied_to_id=user_id
        )
        assert database.update_message_status(user_id, "replied") is True
        pending = database.list_messages(sender="user", status="pending")
        replies = database.list_messages(sender="agent")
        assert pending == []
        assert replies[0]["id"] == reply_id
        assert replies[0]["replied_to_id"] == user_id
    finally:
        database.close()


def test_interests_normalize_deduplicate_decay_and_cap_active(tmp_path: Path) -> None:
    database = Database(tmp_path / "world_data" / "orion_world.sqlite3")
    try:
        first = database.upsert_interest(" Desenho ", 0.4, source="message")
        duplicate = database.upsert_interest("DESENHO", 0.7, source="activity")
        assert first == duplicate
        for index in range(5):
            database.upsert_interest(
                f"tema {index}", index / 10, source="test", maximum_active=3
            )
        active = database.list_interests(limit=20)
        assert len(active) <= 3
        before = active[0]["strength"]
        database.decay_interests(0.1)
        after = database.list_interests(limit=20)[0]["strength"]
        assert after == max(0.0, before - 0.1)
        assert all(0.0 <= item["strength"] <= 1.0 for item in active)
    finally:
        database.close()
