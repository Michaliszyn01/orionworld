"""Behavioral tests for the closed, world-only tool surface."""

from __future__ import annotations

from pathlib import Path

from orionworld.database import Database
from orionworld.models import ToolResult
from orionworld.tools import ALLOWED_TOOL_NAMES, TOOL_ARGUMENT_SCHEMAS, ToolRegistry
from orionworld.world import WorldSandbox


def _assert_structured(result: ToolResult, tool: str) -> None:
    assert result.tool == tool
    assert set(result.to_dict()) == {
        "success",
        "tool",
        "result",
        "files_changed",
        "error",
    }


def test_registry_exposes_exactly_seven_tools_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    blocks: list[tuple[str, str, dict[str, object]]] = []
    registry = ToolRegistry(
        WorldSandbox(tmp_path / "world_data"),
        simulation=False,
        audit_callback=lambda operation, reason, details: blocks.append(
            (operation, reason, details)
        ),
    )

    assert registry.tool_names == ALLOWED_TOOL_NAMES
    assert tuple(TOOL_ARGUMENT_SCHEMAS) == ALLOWED_TOOL_NAMES
    assert len(registry.tool_names) == 7
    result = registry.execute("run_system_command", {"command": "anything"})
    _assert_structured(result, "run_system_command")
    assert result.success is False
    assert result.files_changed == []
    assert result.error == "tool is not allowed"
    assert blocks == [
        (
            "tool_access",
            "tool is not allowed",
            {
                "tool": "run_system_command",
                "category": "tool",
                "blocked": True,
            },
        )
    ]


def test_all_allowed_tools_stay_inside_world_and_return_structured_results(
    tmp_path: Path,
) -> None:
    world = WorldSandbox(tmp_path / "world_data")
    registry = ToolRegistry(world, simulation=False)

    calls = (
        ("list_world_files", {}),
        (
            "write_world_text",
            {"path": "experiments/note.md", "content": "safe experiment"},
        ),
        ("read_world_text", {"path": "experiments/note.md"}),
        (
            "append_diary_entry",
            {"filename": "daily.md", "content": "A short observation."},
        ),
        (
            "create_svg_drawing",
            {"filename": "orbit.svg", "title": "Orbit", "caption": "safe"},
        ),
        (
            "create_project_note",
            {
                "project_name": "First project",
                "title": "Next step",
                "content": "Keep the note local.",
            },
        ),
        ("inspect_project_status", {"project_name": "First project"}),
    )

    for tool_name, arguments in calls:
        result = registry.execute(tool_name, arguments)
        _assert_structured(result, tool_name)
        assert result.success is True, result.error
        assert all(not Path(path).is_absolute() for path in result.files_changed)

    assert world.list_files() == [
        "diary/daily.md",
        "drawings/orbit.svg",
        "experiments/note.md",
        "projects/first_project.md",
    ]


def test_simulation_validates_but_does_not_modify_content_files(
    tmp_path: Path,
) -> None:
    world = WorldSandbox(tmp_path / "world_data")
    registry = ToolRegistry(world, simulation=True)

    intended = registry.execute(
        "write_world_text",
        {"path": "workspace/future.txt", "content": "not written"},
    )
    blocked = registry.execute(
        "write_world_text",
        {"path": "../escape.txt", "content": "not written"},
    )

    assert intended.success is True
    assert intended.files_changed == []
    assert intended.result == {
        "simulation": True,
        "intended_path": "workspace/future.txt",
        "bytes": 11,
    }
    assert blocked.success is False
    assert world.list_files() == []
    assert not (tmp_path / "escape.txt").exists()


def test_project_notes_and_inspection_keep_sqlite_index_in_sync(
    tmp_path: Path,
) -> None:
    world = WorldSandbox(tmp_path / "world_data")
    database = Database(
        world.root / "orion_world.sqlite3",
        allowed_root=world.root,
    )
    registry = ToolRegistry(world, simulation=False, database=database)
    try:
        created = registry.create_project_note(
            "Projeto Atlas",
            "Próximo passo",
            "Manter o projeto local.",
        )
        assert created.success is True
        project = database.get_project("Projeto Atlas")
        assert project is not None
        assert project["status"] == "active"
        assert project["metadata"] == {"path": "projects/projeto_atlas.md"}

        world.write_text(
            "projects/external.md",
            "# External\n\nStatus: incomplete\n\n- [ ] Próximo passo\n",
        )
        inspected = registry.inspect_project_status()
        assert inspected.success is True
        external = database.get_project("external")
        assert external is not None
        assert external["status"] == "incomplete"
        assert external["metadata"] == {"path": "projects/external.md"}
    finally:
        database.close()
