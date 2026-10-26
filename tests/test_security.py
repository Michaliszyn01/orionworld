"""Regression tests for the filesystem and capability security boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from orionworld.security import (
    PathSecurityViolation,
    SecurityPolicy,
    WorldLimitViolation,
)
from orionworld.world import WorldSandbox


@pytest.mark.parametrize(
    "untrusted_path",
    (
        "../outside.txt",
        "workspace/../../outside.txt",
        "workspace/../outside.txt",
    ),
)
def test_path_traversal_is_blocked_and_audited(
    tmp_path: Path,
    untrusted_path: str,
) -> None:
    audit_events: list[tuple[str, str, dict[str, object]]] = []
    policy = SecurityPolicy(
        audit_callback=lambda operation, reason, details: audit_events.append(
            (operation, reason, details)
        )
    )
    world = WorldSandbox(tmp_path / "world_data", security_policy=policy)

    with pytest.raises(PathSecurityViolation) as raised:
        world.write_text(untrusted_path, "must not be written")

    assert "traversal" in raised.value.reason
    assert audit_events == [
        (
            "write_world_text",
            "path traversal is forbidden",
            {
                "path": untrusted_path,
                "category": "path",
                "blocked": True,
            },
        )
    ]
    assert not (tmp_path / "outside.txt").exists()


def test_absolute_paths_are_blocked(tmp_path: Path) -> None:
    world = WorldSandbox(tmp_path / "world_data")
    absolute_path = (tmp_path / "absolute.txt").resolve()

    with pytest.raises(PathSecurityViolation) as raised:
        world.write_text(absolute_path, "must not be written")

    assert "absolute" in raised.value.reason
    assert not absolute_path.exists()


@pytest.mark.parametrize(
    "relative_path",
    (
        "workspace/script.py",
        "workspace/program.exe",
        "workspace/archive.zip",
        "workspace/no_extension",
        "workspace/upper.TXT",
    ),
)
def test_only_explicitly_allowed_extensions_can_be_written(
    tmp_path: Path,
    relative_path: str,
) -> None:
    world = WorldSandbox(tmp_path / "world_data")

    with pytest.raises(PathSecurityViolation) as raised:
        world.write_text(relative_path, "blocked")

    assert "extension" in raised.value.reason
    assert not (world.root / relative_path).exists()


def test_symlink_cannot_escape_world_when_supported(tmp_path: Path) -> None:
    world = WorldSandbox(tmp_path / "world_data")
    outside = tmp_path / "outside"
    outside.mkdir()
    escaping_link = world.root / "workspace" / "escape"
    try:
        escaping_link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable in this environment: {exc}")

    with pytest.raises(PathSecurityViolation) as raised:
        world.write_text("workspace/escape/leak.txt", "must stay isolated")

    assert "outside" in raised.value.reason
    assert not (outside / "leak.txt").exists()


def test_per_file_size_limit_is_enforced_without_partial_append(
    tmp_path: Path,
) -> None:
    world = WorldSandbox(
        tmp_path / "world_data",
        maximum_file_size_bytes=8,
        maximum_total_world_size_bytes=16,
    )
    world.write_text("workspace/exact.txt", "12345678")

    with pytest.raises(WorldLimitViolation) as raised:
        world.append_text("workspace/exact.txt", "9")

    assert "file" in raised.value.reason
    assert world.read_text("workspace/exact.txt") == "12345678"
    assert world.total_size_bytes == 8


def test_total_world_size_limit_is_enforced(tmp_path: Path) -> None:
    world = WorldSandbox(
        tmp_path / "world_data",
        maximum_file_size_bytes=8,
        maximum_total_world_size_bytes=10,
    )
    world.write_text("workspace/first.txt", "123456")

    with pytest.raises(WorldLimitViolation) as raised:
        world.write_text("workspace/second.txt", "12345")

    assert "world" in raised.value.reason
    assert world.list_files() == ["workspace/first.txt"]
    assert world.total_size_bytes == 6
