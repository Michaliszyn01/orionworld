"""Configuration is strict and cannot opt into excluded capabilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orionworld.config import Settings, SettingsError, load_settings


def _valid_mapping() -> dict[str, object]:
    return {
        "simulation": True,
        "cycle_interval_seconds": 10,
        "random_seed": 42,
        "maximum_actions_per_cycle": 1,
        "maximum_file_size_bytes": 100_000,
        "maximum_total_world_size_bytes": 10_000_000,
        "allowed_extensions": [".txt", ".md", ".json", ".svg"],
        "network_enabled": False,
        "subprocess_enabled": False,
        "cognitive_engine_enabled": False,
        "model_provider": "disabled",
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_model": "gemma3:4b",
        "model_timeout_seconds": 60,
        "maximum_model_response_characters": 30_000,
        "runtime_enabled": True,
        "dashboard_enabled": True,
        "dashboard_host": "127.0.0.1",
        "dashboard_port": 8765,
        "maximum_model_calls_per_hour": 20,
        "maximum_model_calls_per_day": 100,
        "maximum_actions_per_hour": 30,
        "maximum_writes_per_day": 50,
        "maximum_new_files_per_day": 20,
        "minimum_rest_seconds": 300,
        "maximum_active_cycles_before_rest": 10,
        "maximum_consecutive_failures": 3,
        "maximum_consecutive_similar_activities": 3,
        "maximum_active_projects": 5,
        "world_snapshot_interval_cycles": 1,
    }


def test_settings_load_and_are_deeply_immutable(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(_valid_mapping()), encoding="utf-8")

    settings = load_settings(config_path)

    assert settings == Settings()
    assert isinstance(settings.allowed_extensions, tuple)
    with pytest.raises((AttributeError, TypeError)):
        settings.network_enabled = True  # type: ignore[misc]


@pytest.mark.parametrize(
    ("key", "unsafe_value"),
    (
        ("network_enabled", True),
        ("subprocess_enabled", True),
        ("allowed_extensions", [".txt", ".exe"]),
        ("cycle_interval_seconds", 0),
        ("maximum_actions_per_cycle", 0),
        ("ollama_base_url", "http://localhost:11434"),
        ("model_provider", "remote"),
        ("maximum_model_response_characters", 0),
        ("dashboard_host", "0.0.0.0"),
        ("dashboard_port", 70_000),
        ("maximum_actions_per_hour", 0),
    ),
)
def test_settings_fail_closed(tmp_path: Path, key: str, unsafe_value: object) -> None:
    values = _valid_mapping()
    values[key] = unsafe_value
    config_path = tmp_path / "settings.json"
    config_path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(SettingsError):
        load_settings(config_path)


def test_unknown_or_missing_configuration_is_rejected(tmp_path: Path) -> None:
    values = _valid_mapping()
    values["unexpected"] = "unsafe ambiguity"
    with pytest.raises(SettingsError):
        Settings.from_mapping(values)

    del values["unexpected"]
    del values["network_enabled"]
    with pytest.raises(SettingsError):
        Settings.from_mapping(values)
