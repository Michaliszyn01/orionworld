"""Registro de auditoria legível e estruturado.

O auditor guarda somente resumos funcionais de intenção, critérios e resultado.
Ele nunca solicita nem persiste raciocínio privado do agente.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Mapping
from typing import Any, TextIO

from .database import Database


class AuditLogger:
    """Duplica eventos funcionais no terminal e no banco SQLite."""

    def __init__(
        self,
        database: Database,
        *,
        stream: TextIO | None = None,
        terminal_enabled: bool = True,
    ) -> None:
        self._database = database
        self._stream = stream if stream is not None else sys.stdout
        self._terminal_enabled = terminal_enabled
        self._print_lock = threading.Lock()

    def event(
        self,
        event_type: str,
        content: str,
        *,
        cycle_id: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        terminal_label: str | None = None,
    ) -> int:
        """Persiste um evento e, quando habilitado, imprime um resumo."""

        clean_metadata = dict(metadata or {})
        event_id = self._database.log_event(
            cycle_id=cycle_id,
            event_type=event_type,
            content=content,
            metadata=clean_metadata,
        )
        if self._terminal_enabled:
            label = terminal_label or event_type.upper()
            suffix = ""
            if clean_metadata:
                suffix = " | " + json.dumps(
                    clean_metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            with self._print_lock:
                print(f"[{label}] {content}{suffix}", file=self._stream, flush=True)
        return event_id

    def blocked(
        self,
        operation: str,
        reason: str,
        details: Mapping[str, Any] | None = None,
        *,
        cycle_id: int | None = None,
    ) -> int:
        """Registra uma tentativa impedida pela política de segurança."""

        metadata = {"operation": operation, "reason": reason}
        metadata.update(dict(details or {}))
        return self.event(
            "security_block",
            f"Operação bloqueada: {operation}",
            cycle_id=cycle_id,
            metadata=metadata,
            terminal_label="BLOQUEIO",
        )

    def display(self, label: str, content: str) -> None:
        """Mostra estado transitório sem criar ruído no histórico persistente."""

        if not self._terminal_enabled:
            return
        with self._print_lock:
            print(f"[{label}] {content}", file=self._stream, flush=True)
