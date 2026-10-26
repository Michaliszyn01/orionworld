"""Supervisor não modificável pelo agente e responsável pelo ciclo de vida."""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import Any, NoReturn

from .audit import AuditLogger
from .config import Settings, validate_settings
from .lifeloop import LifeLoop
from .security import SecurityPolicy
from .persistence_runtime import LifeMode


class SupervisorState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"


class Supervisor:
    """Controla a thread do agente e concentra as negações de capacidade.

    O estado do ciclo é mutável, mas o agente não recebe esta instância nem tem
    uma ferramenta que alcance seu código. Esse é o sentido operacional de
    Supervisor imutável nesta etapa.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        life_loop: LifeLoop,
        security_policy: SecurityPolicy,
        audit: AuditLogger,
    ) -> None:
        self.settings = validate_settings(settings)
        self.life_loop = life_loop
        self.security_policy = security_policy
        self.audit = audit

        self._lock = threading.RLock()
        self._state = SupervisorState.CREATED
        self._stop_event = threading.Event()
        self._run_event = threading.Event()
        self._run_event.set()
        self._thread: threading.Thread | None = None

    @property
    def state(self) -> SupervisorState:
        with self._lock:
            return self._state

    @property
    def maximum_actions_per_cycle(self) -> int:
        return self.settings.maximum_actions_per_cycle

    def start(self) -> None:
        with self._lock:
            if self._state is not SupervisorState.CREATED:
                raise RuntimeError("O Supervisor só pode ser iniciado uma vez")
            self._state = SupervisorState.RUNNING
            self.life_loop.set_status("running")
            self.life_loop.mode_controller.mode = LifeMode.ACTIVE
            self._thread = threading.Thread(
                target=self._run_guarded,
                name="orionworld-life-loop",
                daemon=False,
            )
            self._thread.start()
        self.audit.event(
            "supervisor_started",
            "Supervisor iniciou o agente.",
            metadata={
                "cycle_interval_seconds": self.life_loop.cycle_interval,
                "maximum_actions_per_cycle": self.maximum_actions_per_cycle,
                "simulation": self.settings.simulation,
            },
            terminal_label="SUPERVISOR",
        )

    def pause(self) -> bool:
        with self._lock:
            if self._state is not SupervisorState.RUNNING:
                return False
            self._run_event.clear()
            self._state = SupervisorState.PAUSED
            self.life_loop.set_status("paused")
            self.life_loop.mode_controller.mode = LifeMode.PAUSED
        self.audit.event(
            "supervisor_paused",
            "Agente pausado; nenhum novo ciclo será iniciado.",
            terminal_label="SUPERVISOR",
        )
        return True

    def resume(self) -> bool:
        with self._lock:
            if self._state is not SupervisorState.PAUSED:
                return False
            self._run_event.set()
            self._state = SupervisorState.RUNNING
            self.life_loop.set_status("running")
            self.life_loop.mode_controller.mode = LifeMode.ACTIVE
        self.audit.event(
            "supervisor_resumed",
            "Agente retomado.",
            terminal_label="SUPERVISOR",
        )
        return True

    def stop(self, timeout: float = 10.0) -> bool:
        """Solicita parada e aguarda a thread sem encerrar abruptamente."""

        with self._lock:
            if self._state is SupervisorState.STOPPED:
                return True
            if self._state is SupervisorState.CREATED:
                self._state = SupervisorState.STOPPED
                self.life_loop.set_status("stopped")
                self.life_loop.mode_controller.mode = LifeMode.STOPPED
                return True
            self._state = SupervisorState.STOPPING
            self._stop_event.set()
            self._run_event.set()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        stopped = thread is None or not thread.is_alive()
        with self._lock:
            if stopped:
                self._state = SupervisorState.STOPPED
                self.life_loop.set_status("stopped")
                self.life_loop.mode_controller.mode = LifeMode.STOPPED
        self.audit.event(
            "supervisor_stopped" if stopped else "supervisor_stop_timeout",
            (
                "Supervisor encerrou o agente."
                if stopped
                else "A thread do agente não encerrou dentro do tempo limite."
            ),
            metadata={"stopped": stopped},
            terminal_label="SUPERVISOR",
        )
        return stopped

    def set_cycle_interval(self, seconds: float) -> None:
        self.life_loop.set_cycle_interval(seconds)
        self.audit.event(
            "cycle_interval_changed",
            f"Intervalo entre ciclos alterado para {seconds:.3f}s.",
            metadata={"cycle_interval_seconds": float(seconds)},
            terminal_label="SUPERVISOR",
        )

    def status(self) -> dict[str, Any]:
        snapshot = self.life_loop.snapshot().to_dict()
        snapshot.update(
            {
                "supervisor": self.state.value,
                "simulation": self.settings.simulation,
                "cycle_interval_seconds": self.life_loop.cycle_interval,
                "maximum_actions_per_cycle": self.maximum_actions_per_cycle,
                "thread_alive": bool(self._thread and self._thread.is_alive()),
            }
        )
        return snapshot

    def request_network(self, target: object | None = None) -> NoReturn:
        """Ponto explícito de negação e auditoria para qualquer pedido de rede."""

        self.security_policy.check_network("network_access", target)

    def request_subprocess(self, command: object | None = None) -> NoReturn:
        """Ponto explícito de negação e auditoria para subprocessos."""

        self.security_policy.check_subprocess("subprocess", command)

    def request_system_command(self, command: object | None = None) -> NoReturn:
        """Ponto explícito de negação e auditoria para comandos do sistema."""

        self.security_policy.check_system_command("system_command", command)

    def _run_guarded(self) -> None:
        try:
            self.life_loop.run(self._stop_event, self._run_event)
        except Exception as exc:
            self.audit.event(
                "supervisor_error",
                "Exceção não tratada foi contida pelo Supervisor.",
                metadata={"error_type": type(exc).__name__, "error": str(exc)},
                terminal_label="SUPERVISOR",
            )
        finally:
            with self._lock:
                if self._state is not SupervisorState.STOPPING:
                    self._state = SupervisorState.STOPPED
                    self.life_loop.set_status("stopped")


__all__ = ["Supervisor", "SupervisorState"]
