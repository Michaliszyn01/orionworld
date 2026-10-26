"""Tests for bounded state transitions and autonomous activity selection."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from orionworld.activities import ActivityEngine
from orionworld.models import InternalState
from orionworld.state import STATE_FIELDS, StateEngine


EXPECTED_ACTIVITY_TYPES = (
    "observe_world",
    "write_diary",
    "create_text_experiment",
    "create_svg_drawing",
    "review_recent_memory",
    "organize_projects",
    "continue_incomplete_project",
    "idle",
)

REQUIRED_ACTIVITY_FIELDS = {
    "id",
    "type",
    "title",
    "description",
    "motivation",
    "expected_value",
    "novelty",
    "estimated_cost",
    "risk",
    "required_tool",
    "score",
}


def test_state_updates_remain_in_closed_unit_interval(tmp_path: Path) -> None:
    # tmp_path is intentionally the only filesystem fixture used by this module.
    assert tmp_path.is_dir()
    initial = InternalState(
        curiosity=0.0,
        boredom=1.0,
        social_need=0.0,
        completion_pressure=1.0,
        organization_need=0.0,
        confidence=1.0,
        energy=0.0,
        novelty_preference=1.0,
    )
    original_values = initial.to_dict()
    current = initial
    engine = StateEngine(seed=19, jitter=0.05)

    for cycle in range(96):
        current = engine.update(
            current,
            EXPECTED_ACTIVITY_TYPES[cycle % len(EXPECTED_ACTIVITY_TYPES)],
            success=cycle % 3 != 0,
        )
        values = current.to_dict()
        assert tuple(values) == STATE_FIELDS
        assert all(math.isfinite(value) for value in values.values())
        assert all(0.0 <= value <= 1.0 for value in values.values())

    assert initial.to_dict() == original_values
    assert current.to_dict() != original_values


def test_state_sequence_is_reproducible_with_seed(tmp_path: Path) -> None:
    assert tmp_path.is_dir()
    initial = InternalState(
        curiosity=0.73,
        boredom=0.41,
        social_need=0.28,
        completion_pressure=0.66,
        organization_need=0.57,
        confidence=0.62,
        energy=0.81,
        novelty_preference=0.76,
    )
    activities = EXPECTED_ACTIVITY_TYPES * 3
    outcomes = tuple(index % 4 != 0 for index in range(len(activities)))

    def run(seed: int) -> list[dict[str, float]]:
        engine = StateEngine(seed=seed)
        state = initial
        history: list[dict[str, float]] = []
        for activity_type, success in zip(activities, outcomes):
            state = engine.update(state, activity_type, success)
            history.append(state.to_dict())
        return history

    first = run(823)
    second = run(823)

    assert first == second
    assert first != run(824)

    replay_engine = StateEngine(seed=823)
    replay_first = replay_engine.update(initial, activities[0], outcomes[0])
    replay_engine.reset()
    assert replay_engine.update(initial, activities[0], outcomes[0]) == replay_first


def test_generation_produces_exactly_the_eight_specified_activities(
    tmp_path: Path,
) -> None:
    assert tmp_path.is_dir()
    state = InternalState(
        curiosity=0.82,
        boredom=0.64,
        social_need=0.35,
        completion_pressure=0.71,
        organization_need=0.59,
        confidence=0.68,
        energy=0.77,
        novelty_preference=0.84,
    )
    recent = ("write_diary", "observe_world")
    engine = ActivityEngine(seed=101)

    candidates = engine.generate(
        17,
        state,
        recent_activity_types=recent,
        has_incomplete_project=True,
    )

    assert len(candidates) == 8
    assert tuple(candidate.type for candidate in candidates) == EXPECTED_ACTIVITY_TYPES
    assert len({candidate.id for candidate in candidates}) == 8

    for candidate in candidates:
        serialized = candidate.to_dict()
        assert REQUIRED_ACTIVITY_FIELDS <= serialized.keys()
        assert candidate.id == f"cycle-17:{candidate.type}"
        assert candidate.title.strip()
        assert candidate.description.strip()
        assert candidate.motivation.strip()
        assert candidate.required_tool.strip()
        assert candidate.plan["activity_type"] == candidate.type
        assert candidate.plan["tool"] == candidate.required_tool
        assert isinstance(candidate.plan["arguments"], dict)
        assert isinstance(candidate.plan["steps"], list)
        assert isinstance(candidate.plan["intended_files"], list)
        for value in (
            candidate.expected_value,
            candidate.novelty,
            candidate.estimated_cost,
            candidate.risk,
            candidate.score,
        ):
            assert math.isfinite(value)
            assert 0.0 <= value <= 1.0
        assert candidate.score == engine.score_activity(
            candidate,
            state,
            recent,
            has_incomplete_project=True,
        )


def test_score_reflects_state_repetition_and_project_continuity(
    tmp_path: Path,
) -> None:
    assert tmp_path.is_dir()
    engine = ActivityEngine(seed=5)
    low_curiosity = InternalState(curiosity=0.0, energy=0.8)
    high_curiosity = InternalState(curiosity=1.0, energy=0.8)

    low_observation = engine.generate(1, low_curiosity)[0]
    high_observation = engine.generate(1, high_curiosity)[0]
    assert high_observation.score > low_observation.score

    fresh_observation = engine.generate(2, high_curiosity)[0]
    repeated_observation = engine.generate(
        2,
        high_curiosity,
        recent_activity_types=("observe_world",) * 4,
    )[0]
    assert repeated_observation.novelty < fresh_observation.novelty
    assert repeated_observation.score < fresh_observation.score

    without_project = next(
        item
        for item in engine.generate(3, high_curiosity, has_incomplete_project=False)
        if item.type == "continue_incomplete_project"
    )
    with_project = next(
        item
        for item in engine.generate(3, high_curiosity, has_incomplete_project=True)
        if item.type == "continue_incomplete_project"
    )
    assert with_project.expected_value > without_project.expected_value
    assert with_project.score > without_project.score


def test_seeded_choice_is_reproducible_and_not_rigid_argmax(
    tmp_path: Path,
) -> None:
    assert tmp_path.is_dir()
    state = InternalState(
        curiosity=0.79,
        boredom=0.72,
        completion_pressure=0.63,
        organization_need=0.58,
        confidence=0.7,
        energy=0.76,
        novelty_preference=0.81,
    )
    candidates = ActivityEngine(seed=0).generate(
        21,
        state,
        has_incomplete_project=True,
    )
    first_engine = ActivityEngine(seed=442)
    second_engine = ActivityEngine(seed=442)
    probabilities = first_engine.selection_probabilities(
        candidates,
        top_k=3,
        temperature=1.0,
    )

    assert len(probabilities) == 3
    assert all(0.0 < probability < 1.0 for _, probability in probabilities)
    assert sum(probability for _, probability in probabilities) == pytest.approx(1.0)

    first_sequence = [
        first_engine.choose(candidates, top_k=3, temperature=1.0).type
        for _ in range(48)
    ]
    second_sequence = [
        second_engine.choose(candidates, top_k=3, temperature=1.0).type
        for _ in range(48)
    ]
    eligible = {candidate.type for candidate, _ in probabilities}
    argmax_type = probabilities[0][0].type

    assert first_sequence == second_sequence
    assert set(first_sequence) <= eligible
    assert len(set(first_sequence)) > 1
    assert any(activity_type != argmax_type for activity_type in first_sequence)
