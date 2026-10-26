"""Persistência SQLite thread-safe para eventos e memória do OrionWorld."""

from __future__ import annotations

import json
import sqlite3
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import Activity, InternalState, JSONValue, ToolResult


def utc_timestamp(moment: datetime | None = None) -> str:
    """Produz um timestamp ISO 8601 normalizado para UTC."""

    current = moment or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    current = current.astimezone(UTC)
    return current.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_dump(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _unit_interval(name: str, value: float) -> float:
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} deve estar entre 0 e 1")
    return numeric


def _normalized_topic(topic: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(topic).strip().casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(without_marks.split())


class Database:
    """Repositório SQLite compartilhável com segurança entre threads.

    Uma única conexão é protegida por ``RLock``. Cada gravação ocorre dentro de
    uma transação curta, e WAL reduz contenção entre consultas do ciclo e da CLI.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        allowed_root: str | Path | None = None,
    ) -> None:
        candidate = Path(path).expanduser()
        trusted_root: Path | None = None
        if allowed_root is not None:
            trusted_root = Path(allowed_root).expanduser().resolve(strict=False)
            trusted_root.mkdir(parents=True, exist_ok=True)
            sqlite_paths = [candidate] + [
                Path(f"{candidate}{suffix}")
                for suffix in ("-wal", "-shm", "-journal")
            ]
            for sqlite_path in sqlite_paths:
                is_junction = getattr(sqlite_path, "is_junction", lambda: False)
                if sqlite_path.is_symlink() or is_junction():
                    raise ValueError(
                        "Um arquivo SQLite ou auxiliar não pode ser link simbólico"
                    )

        resolved_path = candidate.resolve(strict=False)
        if trusted_root is not None:
            try:
                resolved_path.relative_to(trusted_root)
                candidate.parent.resolve(strict=False).relative_to(trusted_root)
            except ValueError as exc:
                raise ValueError(
                    "O arquivo SQLite deve permanecer dentro de world_data"
                ) from exc
        self.path = resolved_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.path.is_file():
            raise ValueError("O caminho SQLite deve apontar para um arquivo regular")
        self._lock = threading.RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                source_event_id INTEGER,
                times_recalled INTEGER NOT NULL DEFAULT 0,
                last_recalled_at TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (source_event_id) REFERENCES events(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS internal_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                state_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS intentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL,
                title TEXT NOT NULL,
                motivation TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                score REAL NOT NULL,
                status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER NOT NULL,
                tool TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                success INTEGER NOT NULL CHECK (success IN (0, 1)),
                result TEXT NOT NULL,
                files_changed_json TEXT NOT NULL,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                sender TEXT NOT NULL CHECK (sender IN ('user', 'agent')),
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                replied_to_id INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (replied_to_id) REFERENCES messages(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS interests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL UNIQUE,
                strength REAL NOT NULL CHECK (strength >= 0 AND strength <= 1),
                source TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_reinforced_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS runtime_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_id TEXT NOT NULL UNIQUE,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                stop_reason TEXT,
                process_id INTEGER NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                runtime_session_id INTEGER,
                cycle_number INTEGER NOT NULL UNIQUE,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                selected_activity_id TEXT,
                error TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (runtime_session_id) REFERENCES runtime_sessions(id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                amount INTEGER NOT NULL CHECK (amount > 0),
                cycle_id INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS project_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                position INTEGER NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                required_tool TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                result_summary TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(project_id, position),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS world_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER,
                summary_json TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS cognitive_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cycle_id INTEGER,
                entry_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                related_project_id INTEGER,
                related_memory_id INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (related_project_id) REFERENCES projects(id) ON DELETE SET NULL,
                FOREIGN KEY (related_memory_id) REFERENCES memories(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS control_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                command TEXT NOT NULL,
                status TEXT NOT NULL,
                processed_at TEXT,
                result TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_events_cycle ON events(cycle_id, id);
            CREATE INDEX IF NOT EXISTS idx_memories_relevance
                ON memories(importance DESC, confidence DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_states_cycle
                ON internal_states(cycle_id, id);
            CREATE INDEX IF NOT EXISTS idx_intentions_cycle
                ON intentions(cycle_id, id);
            CREATE INDEX IF NOT EXISTS idx_actions_cycle ON actions(cycle_id, id);
            CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status, id);
            CREATE INDEX IF NOT EXISTS idx_messages_status
                ON messages(status, sender, id);
            CREATE INDEX IF NOT EXISTS idx_interests_active
                ON interests(active, strength DESC, id DESC);
            CREATE INDEX IF NOT EXISTS idx_runtime_sessions_status
                ON runtime_sessions(status, id);
            CREATE INDEX IF NOT EXISTS idx_cycles_status ON cycles(status, id);
            CREATE INDEX IF NOT EXISTS idx_resource_usage_window
                ON resource_usage(resource_type, timestamp);
            CREATE INDEX IF NOT EXISTS idx_project_steps_status
                ON project_steps(project_id, status, position);
            CREATE INDEX IF NOT EXISTS idx_world_snapshots_cycle
                ON world_snapshots(cycle_id, id);
            CREATE INDEX IF NOT EXISTS idx_cognitive_journal_type
                ON cognitive_journal(entry_type, id);
            CREATE INDEX IF NOT EXISTS idx_control_commands_status
                ON control_commands(status, id);
            """
        )
        for name, definition in (
            ("origin", "TEXT NOT NULL DEFAULT 'legacy'"),
            ("current_stage", "TEXT NOT NULL DEFAULT 'planning'"),
            ("objective", "TEXT NOT NULL DEFAULT ''"),
            ("success_criteria", "TEXT NOT NULL DEFAULT ''"),
            ("last_activity_at", "TEXT"),
            ("priority", "REAL NOT NULL DEFAULT 0.5"),
            ("progress", "REAL NOT NULL DEFAULT 0.0"),
            ("abandoned_reason", "TEXT"),
        ):
            self._ensure_column("projects", name, definition)
        for name, definition in (
            ("curiosity", "REAL NOT NULL DEFAULT 0.5"),
            ("familiarity", "REAL NOT NULL DEFAULT 0.0"),
            ("evolution_json", "TEXT NOT NULL DEFAULT '[]'"),
        ):
            self._ensure_column("interests", name, definition)
        self._connection.commit()

    def _ensure_column(self, table: str, name: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({table})")
        }
        if name not in columns:
            # Table and column names are internal constants, never user input.
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("O banco de dados já foi fechado")

    @staticmethod
    def _limit(value: int) -> int:
        numeric = int(value)
        if numeric < 1:
            raise ValueError("limit deve ser maior que zero")
        return min(numeric, 10_000)

    def log_event(
        self,
        cycle_id: int | None,
        event_type: str,
        content: str,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO events
                    (timestamp, cycle_id, event_type, content, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(),
                    cycle_id,
                    str(event_type),
                    str(content),
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def save_memory(
        self,
        memory_type: str,
        content: str,
        confidence: float,
        importance: float,
        source_event_id: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        checked_confidence = _unit_interval("confidence", confidence)
        checked_importance = _unit_interval("importance", importance)
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO memories (
                    timestamp, memory_type, content, confidence, importance,
                    source_event_id, times_recalled, last_recalled_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?)
                """,
                (
                    utc_timestamp(),
                    str(memory_type),
                    str(content),
                    checked_confidence,
                    checked_importance,
                    source_event_id,
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def recall_memories(
        self,
        query: str | None = None,
        *,
        memory_type: str | None = None,
        min_importance: float = 0.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Recupera memórias relevantes e registra que elas foram lembradas."""

        search = (query or "").strip()
        checked_minimum = _unit_interval("min_importance", min_importance)
        with self._lock, self._connection:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE (? IS NULL OR memory_type = ?)
                  AND importance >= ?
                  AND (? = '' OR instr(lower(content), lower(?)) > 0)
                ORDER BY importance DESC, confidence DESC, id DESC
                LIMIT ?
                """,
                (
                    memory_type,
                    memory_type,
                    checked_minimum,
                    search,
                    search,
                    self._limit(limit),
                ),
            ).fetchall()
            recalled_at = utc_timestamp()
            self._connection.executemany(
                """
                UPDATE memories
                SET times_recalled = times_recalled + 1,
                    last_recalled_at = ?
                WHERE id = ?
                """,
                [(recalled_at, int(row["id"])) for row in rows],
            )
            results = [self._memory_row(row) for row in rows]
            for result in results:
                result["times_recalled"] = int(result["times_recalled"]) + 1
                result["last_recalled_at"] = recalled_at
            return results

    def save_internal_state(
        self,
        cycle_id: int,
        state: InternalState | Mapping[str, Any],
    ) -> int:
        normalized = (
            state.clamped()
            if isinstance(state, InternalState)
            else InternalState.from_dict(dict(state)).clamped()
        )
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO internal_states (timestamp, cycle_id, state_json)
                VALUES (?, ?, ?)
                """,
                (utc_timestamp(), int(cycle_id), _json_dump(normalized)),
            )
            return int(cursor.lastrowid)

    def latest_internal_state(self) -> InternalState | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT state_json FROM internal_states
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return InternalState.from_dict(json.loads(str(row["state_json"])))

    def save_intention(
        self,
        cycle_id: int,
        activity_type: str | Activity | None = None,
        title: str | None = None,
        motivation: str | None = None,
        plan: Mapping[str, JSONValue] | Sequence[JSONValue] | None = None,
        score: float | None = None,
        status: str = "declared",
        *,
        activity: Activity | None = None,
    ) -> int:
        selected = activity or (
            activity_type if isinstance(activity_type, Activity) else None
        )
        if selected is not None:
            activity_name = selected.type
            title = selected.title if title is None else title
            motivation = selected.motivation if motivation is None else motivation
            plan = selected.plan if plan is None else plan
            score = selected.score if score is None else score
        else:
            activity_name = str(activity_type or "")
        if not activity_name or title is None or motivation is None or score is None:
            raise ValueError("A intenção requer tipo, título, motivação e pontuação")
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO intentions (
                    timestamp, cycle_id, activity_type, title, motivation,
                    plan_json, score, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(),
                    int(cycle_id),
                    activity_name,
                    str(title),
                    str(motivation),
                    _json_dump(plan or {}),
                    float(score),
                    str(status),
                ),
            )
            return int(cursor.lastrowid)

    def update_intention_status(self, intention_id: int, status: str) -> bool:
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                "UPDATE intentions SET status = ? WHERE id = ?",
                (str(status), int(intention_id)),
            )
            return cursor.rowcount > 0

    def save_action(
        self,
        cycle_id: int,
        tool: str | ToolResult | None = None,
        arguments: Mapping[str, JSONValue] | None = None,
        success: bool | None = None,
        result: JSONValue = None,
        files_changed: Sequence[str] | None = None,
        error: str | None = None,
        *,
        tool_result: ToolResult | None = None,
    ) -> int:
        structured = tool_result or (tool if isinstance(tool, ToolResult) else None)
        if structured is not None:
            tool_name = structured.tool
            success = structured.success if success is None else success
            result = structured.result if result is None else result
            files_changed = (
                structured.files_changed if files_changed is None else files_changed
            )
            error = structured.error if error is None else error
        else:
            tool_name = str(tool or "")
        if not tool_name or success is None:
            raise ValueError("A ação requer ferramenta e indicador de sucesso")
        result_text = result if isinstance(result, str) else _json_dump(result)
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO actions (
                    timestamp, cycle_id, tool, arguments_json, success, result,
                    files_changed_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(),
                    int(cycle_id),
                    tool_name,
                    _json_dump(dict(arguments or {})),
                    int(bool(success)),
                    result_text,
                    _json_dump(list(files_changed or [])),
                    error,
                ),
            )
            return int(cursor.lastrowid)

    def list_events(
        self,
        *,
        cycle_id: int | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM events
                WHERE (? IS NULL OR cycle_id = ?)
                  AND (? IS NULL OR event_type = ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    cycle_id,
                    cycle_id,
                    event_type,
                    event_type,
                    self._limit(limit),
                ),
            ).fetchall()
        return [self._event_row(row) for row in rows]

    def list_intentions(
        self,
        *,
        cycle_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM intentions
                WHERE (? IS NULL OR cycle_id = ?)
                  AND (? IS NULL OR status = ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (cycle_id, cycle_id, status, status, self._limit(limit)),
            ).fetchall()
        return [self._intention_row(row) for row in rows]

    def list_actions(
        self,
        *,
        cycle_id: int | None = None,
        success: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        success_value = None if success is None else int(success)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM actions
                WHERE (? IS NULL OR cycle_id = ?)
                  AND (? IS NULL OR success = ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (
                    cycle_id,
                    cycle_id,
                    success_value,
                    success_value,
                    self._limit(limit),
                ),
            ).fetchall()
        return [self._action_row(row) for row in rows]

    def list_memories(
        self,
        *,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE (? IS NULL OR memory_type = ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (memory_type, memory_type, self._limit(limit)),
            ).fetchall()
        return [self._memory_row(row) for row in rows]

    def list_projects(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM projects
                WHERE (? IS NULL OR status = ?)
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (status, status, self._limit(limit)),
            ).fetchall()
        return [self._project_row(row) for row in rows]

    def save_message(
        self,
        sender: str,
        content: str,
        *,
        status: str = "pending",
        replied_to_id: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        """Persist a user message or an agent reply without interpreting it as authority."""

        if sender not in {"user", "agent"}:
            raise ValueError("sender deve ser user ou agent")
        clean = str(content).strip()
        if not clean:
            raise ValueError("content não pode ser vazio")
        if len(clean) > 10_000:
            raise ValueError("content excedeu o limite")
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO messages (
                    timestamp, sender, content, status, replied_to_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(),
                    sender,
                    clean,
                    str(status),
                    replied_to_id,
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def update_message_status(self, message_id: int, status: str) -> bool:
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                "UPDATE messages SET status = ? WHERE id = ?",
                (str(status), int(message_id)),
            )
            return cursor.rowcount > 0

    def list_messages(
        self,
        *,
        sender: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM messages
                WHERE (? IS NULL OR sender = ?)
                  AND (? IS NULL OR status = ?)
                ORDER BY id DESC
                LIMIT ?
                """,
                (sender, sender, status, status, self._limit(limit)),
            ).fetchall()
        return [self._message_row(row) for row in rows]

    def upsert_interest(
        self,
        topic: str,
        strength: float,
        *,
        source: str,
        metadata: Mapping[str, JSONValue] | None = None,
        maximum_active: int = 20,
    ) -> int:
        normalized = _normalized_topic(topic)
        if not normalized or len(normalized) > 120:
            raise ValueError("topic deve conter entre 1 e 120 caracteres")
        checked_strength = _unit_interval("strength", strength)
        if maximum_active < 1:
            raise ValueError("maximum_active deve ser maior que zero")
        now = utc_timestamp()
        with self._lock, self._connection:
            self._ensure_open()
            existing_interest = self._connection.execute(
                "SELECT familiarity, evolution_json FROM interests WHERE topic = ?",
                (normalized,),
            ).fetchone()
            evolution = (
                json.loads(str(existing_interest["evolution_json"]))
                if existing_interest is not None
                else []
            )
            if not isinstance(evolution, list):
                evolution = []
            evolution = evolution[-49:] + [
                {"timestamp": now, "strength": checked_strength, "source": str(source)[:40]}
            ]
            familiarity = min(
                1.0,
                (float(existing_interest["familiarity"]) if existing_interest else 0.0)
                + 0.1,
            )
            self._connection.execute(
                """
                INSERT INTO interests (
                    topic, strength, source, active, created_at, updated_at,
                    last_reinforced_at, metadata_json, curiosity, familiarity,
                    evolution_json
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    strength = MIN(1.0, MAX(interests.strength, excluded.strength)),
                    source = excluded.source,
                    active = 1,
                    updated_at = excluded.updated_at,
                    last_reinforced_at = excluded.last_reinforced_at,
                    metadata_json = excluded.metadata_json,
                    curiosity = excluded.curiosity,
                    familiarity = excluded.familiarity,
                    evolution_json = excluded.evolution_json
                """,
                (
                    normalized,
                    checked_strength,
                    str(source),
                    now,
                    now,
                    now,
                    _json_dump(dict(metadata or {})),
                    checked_strength,
                    familiarity,
                    _json_dump(evolution),
                ),
            )
            self._connection.execute(
                """
                UPDATE interests SET active = 0, updated_at = ?
                WHERE id IN (
                    SELECT id FROM interests WHERE active = 1
                    ORDER BY strength DESC, last_reinforced_at DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (now, int(maximum_active)),
            )
            row = self._connection.execute(
                "SELECT id FROM interests WHERE topic = ?", (normalized,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Falha ao recuperar o interesse persistido")
            return int(row["id"])

    def decay_interests(self, amount: float = 0.02) -> int:
        checked = _unit_interval("amount", amount)
        now = utc_timestamp()
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                UPDATE interests
                SET strength = MAX(0.0, strength - ?),
                    active = CASE WHEN strength - ? > 0 THEN 1 ELSE 0 END,
                    updated_at = ?
                WHERE active = 1
                """,
                (checked, checked, now),
            )
            return cursor.rowcount

    def list_interests(
        self, *, active_only: bool = True, limit: int = 100
    ) -> list[dict[str, Any]]:
        active_value = 1 if active_only else None
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                """
                SELECT * FROM interests
                WHERE (? IS NULL OR active = ?)
                ORDER BY strength DESC, last_reinforced_at DESC, id DESC
                LIMIT ?
                """,
                (active_value, active_value, self._limit(limit)),
            ).fetchall()
        return [self._interest_row(row) for row in rows]

    def create_runtime_session(
        self,
        instance_id: str,
        process_id: int,
        version: str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO runtime_sessions (
                    instance_id, started_at, stopped_at, stop_reason,
                    process_id, version, status, metadata_json
                ) VALUES (?, ?, NULL, NULL, ?, ?, 'running', ?)
                """,
                (
                    str(instance_id),
                    utc_timestamp(),
                    int(process_id),
                    str(version),
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def stop_runtime_session(self, session_id: int, reason: str) -> bool:
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                UPDATE runtime_sessions
                SET stopped_at = ?, stop_reason = ?, status = 'stopped'
                WHERE id = ? AND status = 'running'
                """,
                (utc_timestamp(), str(reason)[:240], int(session_id)),
            )
            return cursor.rowcount > 0

    def list_runtime_sessions(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM runtime_sessions
                WHERE (? IS NULL OR status = ?)
                ORDER BY id DESC LIMIT ?
                """,
                (status, status, self._limit(limit)),
            ).fetchall()
        return [self._with_json_alias(row, "metadata_json", "metadata") for row in rows]

    def mark_stale_session(self, session_id: int) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE runtime_sessions
                SET stopped_at = ?, stop_reason = 'interrupted', status = 'interrupted'
                WHERE id = ? AND status = 'running'
                """,
                (utc_timestamp(), int(session_id)),
            )
            return cursor.rowcount > 0

    def begin_cycle(self, runtime_session_id: int | None, cycle_number: int) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO cycles (
                    runtime_session_id, cycle_number, started_at, completed_at,
                    status, selected_activity_id, error, metadata_json
                ) VALUES (?, ?, ?, NULL, 'pending', NULL, NULL, '{}')
                """,
                (runtime_session_id, int(cycle_number), utc_timestamp()),
            )
            return int(cursor.lastrowid)

    def update_cycle(
        self,
        cycle_id: int,
        status: str,
        *,
        selected_activity_id: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> bool:
        valid = {
            "pending", "observing", "deciding", "acting", "reflecting",
            "completed", "failed", "interrupted", "recovered",
        }
        if status not in valid:
            raise ValueError("estado de ciclo inválido")
        completed_at = utc_timestamp() if status in {
            "completed", "failed", "interrupted", "recovered"
        } else None
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE cycles SET status = ?, selected_activity_id = COALESCE(?, selected_activity_id),
                    error = ?, metadata_json = ?, completed_at = COALESCE(?, completed_at)
                WHERE id = ?
                """,
                (
                    status,
                    selected_activity_id,
                    None if error is None else str(error)[:500],
                    _json_dump(dict(metadata or {})),
                    completed_at,
                    int(cycle_id),
                ),
            )
            return cursor.rowcount > 0

    def list_cycles(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM cycles WHERE (? IS NULL OR status = ?)
                ORDER BY cycle_number DESC LIMIT ?
                """,
                (status, status, self._limit(limit)),
            ).fetchall()
        return [self._with_json_alias(row, "metadata_json", "metadata") for row in rows]

    def recover_incomplete_state(self) -> dict[str, list[int]]:
        """Mark stale work conservatively; ambiguous writes are never replayed."""

        with self._lock, self._connection:
            running = self._connection.execute(
                "SELECT id FROM runtime_sessions WHERE status = 'running' ORDER BY id"
            ).fetchall()
            cycles = self._connection.execute(
                """
                SELECT id FROM cycles
                WHERE status IN ('pending','observing','deciding','acting','reflecting')
                ORDER BY id
                """
            ).fetchall()
            ambiguous = self._connection.execute(
                """
                SELECT id FROM intentions
                WHERE status IN ('declared','running','acting') ORDER BY id
                """
            ).fetchall()
            now = utc_timestamp()
            self._connection.executemany(
                """
                UPDATE runtime_sessions SET status='interrupted', stopped_at=?,
                    stop_reason='recovered_after_unclean_shutdown' WHERE id=?
                """,
                [(now, int(row["id"])) for row in running],
            )
            self._connection.executemany(
                "UPDATE cycles SET status='interrupted', completed_at=? WHERE id=?",
                [(now, int(row["id"])) for row in cycles],
            )
            self._connection.executemany(
                "UPDATE intentions SET status='requires_review' WHERE id=?",
                [(int(row["id"]),) for row in ambiguous],
            )
        return {
            "stale_sessions": [int(row["id"]) for row in running],
            "interrupted_cycles": [int(row["id"]) for row in cycles],
            "ambiguous_actions": [int(row["id"]) for row in ambiguous],
        }

    def record_resource_usage(
        self,
        resource_type: str,
        *,
        amount: int = 1,
        cycle_id: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        if amount < 1:
            raise ValueError("amount deve ser positivo")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO resource_usage
                    (timestamp, resource_type, amount, cycle_id, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(), str(resource_type), int(amount), cycle_id,
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def resource_usage(self, resource_type: str, *, hours: int) -> int:
        threshold = utc_timestamp(datetime.now(UTC) - timedelta(hours=hours))
        with self._lock:
            row = self._connection.execute(
                """
                SELECT COALESCE(SUM(amount), 0) AS total FROM resource_usage
                WHERE resource_type = ? AND timestamp >= ?
                """,
                (str(resource_type), threshold),
            ).fetchone()
        return int(row["total"])

    def create_managed_project(
        self,
        name: str,
        description: str,
        *,
        origin: str,
        objective: str,
        success_criteria: str,
        maximum_active: int,
        priority: float = 0.5,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        clean_name = str(name).strip()
        clean_description = str(description).strip()
        if not clean_name or len(clean_name) > 120:
            raise ValueError("nome do projeto inválido")
        if not clean_description or len(clean_description) > 1_000:
            raise ValueError("descrição do projeto inválida")
        forbidden = ("..", ":\\", ":/", "http://", "https://")
        joined = " ".join((clean_name, clean_description, objective, success_criteria)).casefold()
        if any(value in joined for value in forbidden):
            raise ValueError("projeto contém referência externa ou caminho proibido")
        checked_priority = _unit_interval("priority", priority)
        with self._lock, self._connection:
            count = self._connection.execute(
                """
                SELECT COUNT(*) AS total FROM projects
                WHERE status IN ('idea','planning','active','blocked','paused')
                """
            ).fetchone()
            if int(count["total"]) >= int(maximum_active):
                raise ValueError("limite de projetos ativos alcançado")
            now = utc_timestamp()
            cursor = self._connection.execute(
                """
                INSERT INTO projects (
                    name, description, status, created_at, updated_at, metadata_json,
                    origin, current_stage, objective, success_criteria,
                    last_activity_at, priority, progress, abandoned_reason
                ) VALUES (?, ?, 'idea', ?, ?, ?, ?, 'planning', ?, ?, ?, ?, 0.0, NULL)
                """,
                (
                    clean_name, clean_description, now, now,
                    _json_dump(dict(metadata or {})), str(origin)[:40],
                    str(objective)[:1_000], str(success_criteria)[:1_000], now,
                    checked_priority,
                ),
            )
            return int(cursor.lastrowid)

    def add_project_step(
        self,
        project_id: int,
        position: int,
        title: str,
        description: str,
        required_tool: str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        from .tools import ALLOWED_TOOL_NAMES
        if required_tool not in frozenset(ALLOWED_TOOL_NAMES) | {"none"}:
            raise ValueError("ferramenta da etapa não permitida")
        if not str(title).strip() or len(str(title)) > 160:
            raise ValueError("título da etapa inválido")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO project_steps (
                    project_id, position, title, description, status, required_tool,
                    created_at, started_at, completed_at, result_summary, metadata_json
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    int(project_id), int(position), str(title).strip(),
                    str(description)[:1_000], required_tool, utc_timestamp(),
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def update_project_lifecycle(
        self,
        project_id: int,
        status: str,
        *,
        current_stage: str | None = None,
        progress: float | None = None,
        abandoned_reason: str | None = None,
    ) -> bool:
        valid = {
            "idea", "planning", "active", "blocked", "paused", "completed",
            "abandoned", "archived",
        }
        if status not in valid:
            raise ValueError("estado de projeto inválido")
        checked_progress = None if progress is None else _unit_interval("progress", progress)
        if status == "abandoned" and not str(abandoned_reason or "").strip():
            raise ValueError("abandono requer justificativa")
        now = utc_timestamp()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE projects SET status=?, current_stage=COALESCE(?, current_stage),
                    progress=COALESCE(?, progress), abandoned_reason=?, updated_at=?,
                    last_activity_at=? WHERE id=?
                """,
                (
                    status, current_stage, checked_progress,
                    None if abandoned_reason is None else str(abandoned_reason)[:500],
                    now, now, int(project_id),
                ),
            )
            return cursor.rowcount > 0

    def update_project_step(
        self, step_id: int, status: str, *, result_summary: str | None = None
    ) -> bool:
        valid = {"pending", "active", "completed", "failed", "skipped", "blocked"}
        if status not in valid:
            raise ValueError("estado de etapa inválido")
        now = utc_timestamp()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE project_steps SET status=?,
                    started_at=CASE WHEN ?='active' THEN COALESCE(started_at, ?) ELSE started_at END,
                    completed_at=CASE WHEN ? IN ('completed','failed','skipped','blocked') THEN ? ELSE completed_at END,
                    result_summary=COALESCE(?, result_summary)
                WHERE id=?
                """,
                (status, status, now, status, now, result_summary, int(step_id)),
            )
            return cursor.rowcount > 0

    def list_project_steps(
        self, *, project_id: int | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM project_steps WHERE (? IS NULL OR project_id = ?)
                ORDER BY project_id, position LIMIT ?
                """,
                (project_id, project_id, self._limit(limit)),
            ).fetchall()
        return [self._with_json_alias(row, "metadata_json", "metadata") for row in rows]

    def save_world_snapshot(
        self,
        cycle_id: int | None,
        summary: Mapping[str, JSONValue],
        fingerprint: str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO world_snapshots
                    (timestamp, cycle_id, summary_json, fingerprint, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(), cycle_id, _json_dump(dict(summary)),
                    str(fingerprint), _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def list_world_snapshots(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM world_snapshots ORDER BY id DESC LIMIT ?",
                (self._limit(limit),),
            ).fetchall()
        results = []
        for row in rows:
            item = self._with_json_alias(row, "summary_json", "summary")
            item["metadata"] = json.loads(str(item["metadata_json"]))
            results.append(item)
        return results

    def save_journal_entry(
        self,
        cycle_id: int | None,
        entry_type: str,
        title: str,
        content: str,
        confidence: float,
        *,
        related_project_id: int | None = None,
        related_memory_id: int | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        valid_types = {
            "observation", "reflection", "decision_summary", "project_update",
            "question", "hypothesis", "retrospective",
        }
        if entry_type not in valid_types:
            raise ValueError("tipo de diário inválido")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO cognitive_journal (
                    timestamp, cycle_id, entry_type, title, content, confidence,
                    related_project_id, related_memory_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_timestamp(), cycle_id, entry_type, str(title)[:160],
                    str(content)[:2_000], _unit_interval("confidence", confidence),
                    related_project_id, related_memory_id,
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def list_journal(
        self, *, entry_type: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM cognitive_journal
                WHERE (? IS NULL OR entry_type = ?)
                ORDER BY id DESC LIMIT ?
                """,
                (entry_type, entry_type, self._limit(limit)),
            ).fetchall()
        return [self._with_json_alias(row, "metadata_json", "metadata") for row in rows]

    def enqueue_control_command(
        self,
        command: str,
        *,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        allowed = {"pause", "resume", "request_reflection", "enter_rest", "stop"}
        if command not in allowed:
            raise ValueError("comando de controle não permitido")
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO control_commands
                    (timestamp, command, status, processed_at, result, metadata_json)
                VALUES (?, ?, 'pending', NULL, NULL, ?)
                """,
                (utc_timestamp(), command, _json_dump(dict(metadata or {}))),
            )
            return int(cursor.lastrowid)

    def list_control_commands(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM control_commands WHERE (? IS NULL OR status = ?)
                ORDER BY id ASC LIMIT ?
                """,
                (status, status, self._limit(limit)),
            ).fetchall()
        return [self._with_json_alias(row, "metadata_json", "metadata") for row in rows]

    def complete_control_command(self, command_id: int, result: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE control_commands SET status='completed', processed_at=?, result=?
                WHERE id=? AND status='pending'
                """,
                (utc_timestamp(), str(result)[:500], int(command_id)),
            )
            return cursor.rowcount > 0

    def create_project(
        self,
        name: str,
        description: str,
        *,
        status: str = "active",
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        now = utc_timestamp()
        with self._lock, self._connection:
            self._ensure_open()
            cursor = self._connection.execute(
                """
                INSERT INTO projects (
                    name, description, status, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(name),
                    str(description),
                    str(status),
                    now,
                    now,
                    _json_dump(dict(metadata or {})),
                ),
            )
            return int(cursor.lastrowid)

    def upsert_project(
        self,
        name: str,
        description: str,
        *,
        status: str = "active",
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> int:
        now = utc_timestamp()
        with self._lock, self._connection:
            self._ensure_open()
            self._connection.execute(
                """
                INSERT INTO projects (
                    name, description, status, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    str(name),
                    str(description),
                    str(status),
                    now,
                    now,
                    _json_dump(dict(metadata or {})),
                ),
            )
            row = self._connection.execute(
                "SELECT id FROM projects WHERE name = ?",
                (str(name),),
            ).fetchone()
            if row is None:
                raise RuntimeError("Falha ao recuperar o projeto persistido")
            return int(row["id"])

    def get_project(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT * FROM projects WHERE name = ?",
                (str(name),),
            ).fetchone()
        return None if row is None else self._project_row(row)

    def next_cycle_id(self) -> int:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT COALESCE(MAX(cycle_id), 0) AS maximum_cycle
                FROM (
                    SELECT cycle_id FROM events
                    UNION ALL SELECT cycle_id FROM internal_states
                    UNION ALL SELECT cycle_id FROM intentions
                    UNION ALL SELECT cycle_id FROM actions
                    UNION ALL SELECT cycle_number AS cycle_id FROM cycles
                )
                WHERE cycle_id IS NOT NULL
                """
            ).fetchone()
        return int(row["maximum_cycle"]) + 1

    @staticmethod
    def _with_json_alias(
        row: sqlite3.Row,
        source: str,
        alias: str,
    ) -> dict[str, Any]:
        item = dict(row)
        item[alias] = json.loads(str(item[source]))
        return item

    @classmethod
    def _event_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        return cls._with_json_alias(row, "metadata_json", "metadata")

    @classmethod
    def _memory_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        return cls._with_json_alias(row, "metadata_json", "metadata")

    @classmethod
    def _intention_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        return cls._with_json_alias(row, "plan_json", "plan")

    @classmethod
    def _action_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        item = cls._with_json_alias(row, "arguments_json", "arguments")
        item["files_changed"] = json.loads(str(item["files_changed_json"]))
        item["success"] = bool(item["success"])
        return item

    @classmethod
    def _project_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        return cls._with_json_alias(row, "metadata_json", "metadata")

    @classmethod
    def _message_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        return cls._with_json_alias(row, "metadata_json", "metadata")

    @classmethod
    def _interest_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        item = cls._with_json_alias(row, "metadata_json", "metadata")
        item["active"] = bool(item["active"])
        item["evolution"] = json.loads(str(item["evolution_json"]))
        return item

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "Database":
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


OrionDatabase = Database
