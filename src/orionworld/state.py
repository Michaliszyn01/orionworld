"""Deterministic transitions for OrionWorld's artificial internal state.

The values in :class:`InternalState` are computational signals used by the
activity selector.  They are not emotions or claims about consciousness.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from .models import InternalState


STATE_FIELDS: tuple[str, ...] = (
    "curiosity",
    "boredom",
    "social_need",
    "completion_pressure",
    "organization_need",
    "confidence",
    "energy",
    "novelty_preference",
)


# Slow drift applied once per cycle before the selected activity is considered.
_CYCLE_DRIFT: Mapping[str, float] = {
    "curiosity": 0.010,
    "boredom": 0.018,
    "social_need": 0.006,
    "completion_pressure": 0.008,
    "organization_need": 0.006,
    "confidence": -0.002,
    "energy": -0.010,
    "novelty_preference": 0.004,
}


# A successful activity normally satisfies the signals which motivated it.
_ACTIVITY_DELTAS: Mapping[str, Mapping[str, float]] = {
    "observe_world": {
        "curiosity": -0.080,
        "boredom": -0.040,
        "organization_need": 0.015,
        "confidence": 0.010,
        "energy": -0.025,
        "novelty_preference": -0.025,
    },
    "write_diary": {
        "boredom": -0.035,
        "social_need": -0.035,
        "completion_pressure": -0.010,
        "organization_need": -0.025,
        "confidence": 0.015,
        "energy": -0.020,
    },
    "create_text_experiment": {
        "curiosity": -0.100,
        "boredom": -0.110,
        "completion_pressure": 0.010,
        "confidence": 0.025,
        "energy": -0.070,
        "novelty_preference": -0.100,
    },
    "create_svg_drawing": {
        "curiosity": -0.075,
        "boredom": -0.130,
        "confidence": 0.025,
        "energy": -0.085,
        "novelty_preference": -0.120,
    },
    "review_recent_memory": {
        "curiosity": -0.025,
        "boredom": -0.020,
        "completion_pressure": -0.020,
        "organization_need": -0.045,
        "confidence": 0.025,
        "energy": -0.030,
        "novelty_preference": 0.015,
    },
    "organize_projects": {
        "boredom": 0.010,
        "completion_pressure": -0.060,
        "organization_need": -0.140,
        "confidence": 0.035,
        "energy": -0.055,
        "novelty_preference": 0.020,
    },
    "continue_incomplete_project": {
        "curiosity": -0.025,
        "boredom": -0.050,
        "completion_pressure": -0.180,
        "organization_need": -0.025,
        "confidence": 0.050,
        "energy": -0.095,
        "novelty_preference": 0.015,
    },
    "idle": {
        "curiosity": 0.025,
        "boredom": 0.075,
        "social_need": 0.010,
        "completion_pressure": 0.010,
        "confidence": -0.005,
        "energy": 0.180,
        "novelty_preference": 0.020,
    },
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class StateEngine:
    """Update artificial state with seeded, bounded transitions.

    Two engines constructed with the same seed and called with the same inputs
    produce the same sequence of states.  The input object is never mutated.
    """

    def __init__(self, seed: int = 42, *, jitter: float = 0.006) -> None:
        if not 0.0 <= jitter <= 0.05:
            raise ValueError("jitter must be between 0 and 0.05")
        self.seed = seed
        self.jitter = jitter
        self._random = random.Random(seed)

    def update(
        self,
        state: InternalState,
        activity_type: str,
        success: bool,
    ) -> InternalState:
        """Return the state after one cycle and one activity outcome.

        Unknown activity types still receive normal cycle drift and outcome
        feedback.  This keeps recovery paths safe if an old activity is loaded
        from persistent storage.
        """

        values = {name: float(getattr(state, name)) for name in STATE_FIELDS}
        deltas = _ACTIVITY_DELTAS.get(activity_type, {})

        for name in STATE_FIELDS:
            values[name] += _CYCLE_DRIFT[name]
            values[name] += deltas.get(name, 0.0)

        if activity_type != "idle":
            if success:
                values["confidence"] += 0.035
                values["completion_pressure"] -= 0.015
            else:
                # Failed work should not be treated as satisfying the drive.
                for name, delta in deltas.items():
                    if delta < 0.0:
                        values[name] -= 0.65 * delta
                values["confidence"] -= 0.060
                values["completion_pressure"] += 0.040
                values["organization_need"] += 0.015
                values["boredom"] += 0.020

        for name in STATE_FIELDS:
            values[name] = _clamp(
                values[name] + self._random.uniform(-self.jitter, self.jitter)
            )

        return InternalState(**values)

    def reset(self, seed: int | None = None) -> None:
        """Restart the reproducible jitter sequence, optionally with a new seed."""

        if seed is not None:
            self.seed = seed
        self._random.seed(self.seed)


def update_state(
    state: InternalState,
    activity_type: str,
    success: bool,
    *,
    seed: int = 42,
    jitter: float = 0.006,
) -> InternalState:
    """Convenience function for an isolated, reproducible state transition."""

    return StateEngine(seed=seed, jitter=jitter).update(state, activity_type, success)


__all__ = ["STATE_FIELDS", "StateEngine", "update_state"]
