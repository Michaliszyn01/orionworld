"""Persistent runtime primitives: instance lock, recovery, modes, and budgets."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
import uuid

from .audit import AuditLogger
from .config import Settings
from .database import Database, utc_timestamp
from .models import InternalState


RUNTIME_VERSION = "0.3.0"


class InstanceAlreadyRunning(RuntimeError):
    """Raised when the operating-system lock is already held."""


class InstanceLock:
    """One-byte OS lock with Windows support and a portable POSIX fallback."""

    def __init__(self, path: str | Path, *, version: str = RUNTIME_VERSION) -> None:
        self.path = Path(path).resolve(strict=False)
        self.version = version
        self.instance_id = str(uuid.uuid4())
        self._handle = None

    def acquire(self) -> dict[str, object]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise InstanceAlreadyRunning("já existe um runtime ativo para este mundo") from exc
        metadata = {
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "started_at": utc_timestamp(),
            "version": self.version,
        }
        encoded = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        handle.seek(0)
        handle.truncate()
        handle.write(encoded)
        handle.flush()
        self._handle = handle
        return metadata

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None


class LifeMode(StrEnum):
    ACTIVE = "active"
    REFLECTIVE = "reflective"
    IDLE = "idle"
    RESTING = "resting"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass(slots=True)
class ModeController:
    settings: Settings
    mode: LifeMode = LifeMode.ACTIVE
    active_cycles: int = 0
    consecutive_failures: int = 0
    rest_started_at: datetime | None = None
    reflection_requested: bool = False

    def choose(
        self,
        state: InternalState,
        *,
        pending_messages: int,
        has_incomplete_project: bool,
        repetition_detected: bool,
    ) -> LifeMode:
        now = datetime.now(UTC)
        if self.mode is LifeMode.RESTING:
            elapsed = (now - (self.rest_started_at or now)).total_seconds()
            if elapsed < self.settings.minimum_rest_seconds:
                return self.mode
            self.mode = LifeMode.ACTIVE
            self.active_cycles = 0
        if self.reflection_requested or repetition_detected:
            self.reflection_requested = False
            self.mode = LifeMode.REFLECTIVE
        elif self.consecutive_failures >= self.settings.maximum_consecutive_failures:
            self.enter_rest()
        elif self.active_cycles >= self.settings.maximum_active_cycles_before_rest:
            self.enter_rest()
        elif state.energy < 0.2 and not pending_messages:
            self.enter_rest()
        elif pending_messages or has_incomplete_project:
            self.mode = LifeMode.ACTIVE
        elif state.energy < 0.4:
            self.mode = LifeMode.IDLE
        else:
            self.mode = LifeMode.ACTIVE
        return self.mode

    def record_cycle(self, success: bool) -> None:
        self.active_cycles += 1
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1

    def enter_rest(self) -> None:
        self.mode = LifeMode.RESTING
        self.rest_started_at = datetime.now(UTC)


class ResourceBudget:
    """Persistent sliding-window counters backed by SQLite."""

    def __init__(self, database: Database, settings: Settings, audit: AuditLogger) -> None:
        self.database = database
        self.settings = settings
        self.audit = audit

    def allow(self, resource: str, *, cycle_id: int | None = None) -> bool:
        limits = {
            "model_call": (
                (1, self.settings.maximum_model_calls_per_hour),
                (24, self.settings.maximum_model_calls_per_day),
            ),
            "action": ((1, self.settings.maximum_actions_per_hour),),
            "write": ((24, self.settings.maximum_writes_per_day),),
            "new_file": ((24, self.settings.maximum_new_files_per_day),),
        }
        if resource not in limits:
            raise ValueError("recurso desconhecido")
        for hours, limit in limits[resource]:
            if self.database.resource_usage(resource, hours=hours) >= limit:
                self.audit.event(
                    "resource_limit_reached",
                    "Operação bloqueada por orçamento persistente.",
                    cycle_id=cycle_id,
                    metadata={"resource": resource, "window_hours": hours, "limit": limit},
                )
                return False
        return True

    def consume(self, resource: str, *, cycle_id: int | None = None) -> bool:
        if not self.allow(resource, cycle_id=cycle_id):
            return False
        self.database.record_resource_usage(resource, cycle_id=cycle_id)
        return True

    def remaining(self) -> dict[str, int]:
        return {
            "model_calls_hour": max(0, self.settings.maximum_model_calls_per_hour - self.database.resource_usage("model_call", hours=1)),
            "model_calls_day": max(0, self.settings.maximum_model_calls_per_day - self.database.resource_usage("model_call", hours=24)),
            "actions_hour": max(0, self.settings.maximum_actions_per_hour - self.database.resource_usage("action", hours=1)),
            "writes_day": max(0, self.settings.maximum_writes_per_day - self.database.resource_usage("write", hours=24)),
            "new_files_day": max(0, self.settings.maximum_new_files_per_day - self.database.resource_usage("new_file", hours=24)),
        }


class RuntimeRecovery:
    def __init__(self, database: Database, audit: AuditLogger) -> None:
        self.database = database
        self.audit = audit

    def recover(self) -> dict[str, list[int]]:
        self.audit.event("runtime_recovery_started", "Verificação conservadora de recuperação iniciada.")
        result = self.database.recover_incomplete_state()
        for session_id in result["stale_sessions"]:
            self.audit.event("stale_session_detected", "Sessão anterior não foi encerrada de forma limpa.", metadata={"session_id": session_id})
        for cycle_id in result["interrupted_cycles"]:
            self.audit.event("interrupted_cycle_detected", "Ciclo incompleto marcado como interrompido.", metadata={"cycle_row_id": cycle_id})
        for intention_id in result["ambiguous_actions"]:
            self.audit.event("ambiguous_action_detected", "Ação com resultado desconhecido requer revisão e não será repetida.", metadata={"intention_id": intention_id})
        self.audit.event("runtime_recovery_completed", "Verificação de recuperação concluída.", metadata={key: len(value) for key, value in result.items()})
        return result


__all__ = [
    "InstanceAlreadyRunning", "InstanceLock", "LifeMode", "ModeController",
    "ResourceBudget", "RUNTIME_VERSION", "RuntimeRecovery",
]
