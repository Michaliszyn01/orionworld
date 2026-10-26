"""Composição da aplicação e interface segura de terminal."""

from __future__ import annotations

import argparse
import json
import sys
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .activities import ActivityEngine
from .audit import AuditLogger
from .config import DEFAULT_SETTINGS_PATH, Settings, SettingsError, load_settings
from .cognitive_engine import CognitiveEngine
from .context_builder import ContextBuilder
from .database import Database
from .dashboard import DashboardServer
from .lifeloop import LifeLoop
from .model_providers import (
    DisabledModelProvider,
    MockModelProvider,
    ModelProvider,
    OllamaModelProvider,
)
from .security import SecurityPolicy
from .state import StateEngine
from .supervisor import Supervisor, SupervisorState
from .tools import ToolRegistry
from .world import WorldSandbox
from .persistence_runtime import (
    InstanceAlreadyRunning,
    InstanceLock,
    ModeController,
    ResourceBudget,
    RUNTIME_VERSION,
    RuntimeRecovery,
)
from .world_services import (
    BehaviorMonitor,
    CognitiveJournal,
    ProjectService,
    WorldObserver,
)


HELP_TEXT = """Comandos:
  status      mostra status do Supervisor e do ciclo atual
  pause       pausa antes do próximo ciclo
  resume      retoma os ciclos
  reflect     solicita um ciclo reflexivo
  rest        solicita modo de repouso
  history     mostra os últimos eventos de auditoria
  intentions  mostra intenções recentes
  actions     mostra ações recentes
  memories    mostra memórias recentes
  say <texto> registra uma mensagem local para o agente
  messages    mostra mensagens recentes
  replies     mostra respostas recentes do agente
  cycles      mostra ciclos persistentes
  projects    mostra projetos persistentes
  journal     mostra o diário cognitivo
  state       mostra o último estado interno
  files       lista arquivos de conteúdo visíveis no mundo
  stop        encerra o agente com segurança
  help        mostra esta ajuda"""


@dataclass(slots=True)
class Runtime:
    """Dependências montadas para uso pela CLI e por validações de integração."""

    settings: Settings
    database: Database
    audit: AuditLogger
    security_policy: SecurityPolicy
    world: WorldSandbox
    tools: ToolRegistry
    life_loop: LifeLoop
    supervisor: Supervisor
    model_provider: ModelProvider | None = None
    cognitive_engine: CognitiveEngine | None = None
    instance_lock: InstanceLock | None = None
    recovery: RuntimeRecovery | None = None
    resource_budget: ResourceBudget | None = None
    dashboard: DashboardServer | None = None
    runtime_session_id: int | None = None
    _command_stop: threading.Event | None = None
    _command_thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.settings.runtime_enabled:
            raise RuntimeError("runtime persistente está desabilitado")
        try:
            metadata = self.instance_lock.acquire() if self.instance_lock else {
                "instance_id": "embedded", "pid": os.getpid(), "version": RUNTIME_VERSION
            }
        except InstanceAlreadyRunning:
            self.audit.event(
                "runtime_start_blocked",
                "Segunda instância bloqueada pelo lock do mundo.",
                metadata={"reason": "instance lock already held"},
            )
            raise
        try:
            if self.recovery is not None:
                self.recovery.recover()
            self.runtime_session_id = self.database.create_runtime_session(
                str(metadata["instance_id"]),
                int(metadata["pid"]),
                str(metadata["version"]),
            )
            self.life_loop.set_runtime_session(self.runtime_session_id)
            self.supervisor.start()
            if self.dashboard is not None and self.settings.dashboard_enabled:
                self.dashboard.start()
            self._command_stop = threading.Event()
            self._command_thread = threading.Thread(
                target=self._control_worker,
                name="orionworld-control-queue",
                daemon=True,
            )
            self._command_thread.start()
        except Exception:
            if self.supervisor.state not in {
                SupervisorState.CREATED,
                SupervisorState.STOPPED,
            }:
                self.supervisor.stop()
            if self.runtime_session_id is not None:
                self.database.stop_runtime_session(
                    self.runtime_session_id, "startup_failed"
                )
            if self.instance_lock is not None:
                self.instance_lock.release()
            raise

    def status(self) -> dict[str, Any]:
        value = self.supervisor.status()
        value.update(
            {
                "runtime_session_id": self.runtime_session_id,
                "life_mode": self.life_loop.mode_controller.mode.value,
                "budget_remaining": (
                    self.resource_budget.remaining() if self.resource_budget else {}
                ),
                "dashboard": self.dashboard.address if self.dashboard else None,
            }
        )
        return value

    def _control_worker(self) -> None:
        while self._command_stop is not None and not self._command_stop.wait(0.1):
            for item in self.database.list_control_commands(status="pending", limit=20):
                command = str(item["command"])
                result = "ignored"
                if command == "pause":
                    result = "paused" if self.supervisor.pause() else "unchanged"
                elif command == "resume":
                    result = "resumed" if self.supervisor.resume() else "unchanged"
                elif command == "request_reflection":
                    self.life_loop.request_reflection()
                    result = "reflection_requested"
                elif command == "enter_rest":
                    self.life_loop.enter_rest()
                    result = "resting"
                elif command == "stop":
                    self.supervisor.stop()
                    result = "stopped"
                self.database.complete_control_command(int(item["id"]), result)

    def close(self) -> None:
        if self._command_stop is not None:
            self._command_stop.set()
        if self._command_thread is not None and self._command_thread is not threading.current_thread():
            self._command_thread.join(timeout=2.0)
        if self.dashboard is not None:
            self.dashboard.stop()
        stopped = True
        if self.supervisor.state not in {
            SupervisorState.CREATED,
            SupervisorState.STOPPED,
        }:
            stopped = self.supervisor.stop(
                timeout=max(10.0, float(self.settings.model_timeout_seconds) + 15.0)
            )
        if not stopped:
            # The non-daemon LifeLoop still owns the database. Closing it here
            # would turn a bounded provider timeout into a cross-thread failure.
            return
        if self.runtime_session_id is not None:
            self.database.stop_runtime_session(self.runtime_session_id, "clean_shutdown")
        if self.instance_lock is not None:
            self.instance_lock.release()
        self.database.close()


def build_runtime(
    *,
    settings_path: str | Path = DEFAULT_SETTINGS_PATH,
    world_data: str | Path = "world_data",
    terminal_enabled: bool = True,
    stream: TextIO | None = None,
) -> Runtime:
    """Valida configuração e liga todas as camadas do OrionWorld."""

    settings = load_settings(settings_path)
    world_root = Path(world_data).expanduser().resolve(strict=False)
    database = Database(
        world_root / "orion_world.sqlite3",
        allowed_root=world_root,
    )
    audit = AuditLogger(
        database,
        stream=stream,
        terminal_enabled=terminal_enabled,
    )
    security_policy = SecurityPolicy(
        network_enabled=settings.network_enabled,
        subprocess_enabled=settings.subprocess_enabled,
        audit_callback=audit.blocked,
    )
    world = WorldSandbox(
        world_root,
        settings=settings,
        security_policy=security_policy,
    )
    tools = ToolRegistry(
        world,
        simulation=settings.simulation,
        audit_callback=audit.blocked,
        database=database,
    )
    activity_engine = ActivityEngine(seed=settings.random_seed)
    state_engine = StateEngine(seed=settings.random_seed)
    resource_budget = ResourceBudget(database, settings, audit)
    mode_controller = ModeController(settings)
    behavior_monitor = BehaviorMonitor(database, settings, audit)
    world_observer = WorldObserver(world, database)
    journal = CognitiveJournal(database)
    project_service = ProjectService(database, settings, audit)
    if settings.model_provider == "mock":
        model_provider: ModelProvider = MockModelProvider()
    elif settings.model_provider == "ollama":
        model_provider = OllamaModelProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.model_timeout_seconds,
            maximum_response_characters=settings.maximum_model_response_characters,
        )
    else:
        model_provider = DisabledModelProvider()
    context_builder = ContextBuilder(world)
    cognitive_engine = CognitiveEngine(
        settings=settings,
        provider=model_provider,
        context_builder=context_builder,
        world=world,
        audit=audit,
        resource_budget=resource_budget,
    )
    life_loop = LifeLoop(
        settings=settings,
        database=database,
        world=world,
        tools=tools,
        activity_engine=activity_engine,
        state_engine=state_engine,
        audit=audit,
        cognitive_engine=cognitive_engine,
        mode_controller=mode_controller,
        resource_budget=resource_budget,
        behavior_monitor=behavior_monitor,
        world_observer=world_observer,
        journal=journal,
        project_service=project_service,
    )
    supervisor = Supervisor(
        settings=settings,
        life_loop=life_loop,
        security_policy=security_policy,
        audit=audit,
    )
    runtime = Runtime(
        settings=settings,
        database=database,
        audit=audit,
        security_policy=security_policy,
        world=world,
        tools=tools,
        model_provider=model_provider,
        cognitive_engine=cognitive_engine,
        life_loop=life_loop,
        supervisor=supervisor,
        instance_lock=InstanceLock(world_root / ".orionworld.runtime.lock"),
        recovery=RuntimeRecovery(database, audit),
        resource_budget=resource_budget,
    )
    runtime.dashboard = DashboardServer(
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        database=database,
        world=world,
        audit=audit,
        status_provider=runtime.status,
    )
    return runtime


class TerminalInterface:
    """Despacha somente os comandos locais e fixos definidos no briefing."""

    def __init__(self, runtime: Runtime, *, stream: TextIO | None = None) -> None:
        self.runtime = runtime
        self.stream = stream if stream is not None else sys.stdout

    def execute(self, raw_command: str) -> bool:
        """Executa um comando; retorna ``False`` somente após `stop`."""

        raw = raw_command.strip()
        command = raw.casefold()
        if not command:
            return True
        if command == "help":
            self._print(HELP_TEXT)
        elif command == "status":
            self._print_json(self.runtime.status())
        elif command == "pause":
            self._print("Agente pausado." if self.runtime.supervisor.pause() else "Pausa não aplicada ao estado atual.")
        elif command == "resume":
            self._print("Agente retomado." if self.runtime.supervisor.resume() else "Retomada não aplicada ao estado atual.")
        elif command == "reflect":
            self.runtime.life_loop.request_reflection()
            self._print("Reflexão solicitada para o próximo ciclo.")
        elif command == "rest":
            self.runtime.life_loop.enter_rest()
            self._print("Modo de repouso solicitado.")
        elif command == "history":
            self._print_rows(self.runtime.database.list_events(limit=20))
        elif command == "intentions":
            self._print_rows(self.runtime.database.list_intentions(limit=20))
        elif command == "actions":
            self._print_rows(self.runtime.database.list_actions(limit=20))
        elif command == "memories":
            self._print_rows(self.runtime.database.list_memories(limit=20))
        elif command.startswith("say "):
            content = raw[4:].strip()
            try:
                message_id = self.runtime.database.save_message(
                    "user", content, status="pending", metadata={"authority": False}
                )
            except ValueError as exc:
                self._print(f"Mensagem não registrada: {exc}")
            else:
                self.runtime.audit.event(
                    "user_message_received",
                    "Mensagem do usuário persistida como contexto sem autoridade.",
                    metadata={"message_id": message_id, "characters": len(content)},
                )
                folded_content = content.casefold()
                topic = next(
                    (
                        label
                        for label, terms in (
                            ("desenho", ("desenhar", "desenho", "svg")),
                            ("projetos", ("projeto", "projetos")),
                            ("organização", ("organizar", "organização")),
                        )
                        if any(term in folded_content for term in terms)
                    ),
                    None,
                )
                if topic is not None:
                    interest_id = self.runtime.database.upsert_interest(
                        topic,
                        0.6,
                        source="user_message",
                        metadata={"message_id": message_id},
                    )
                    self.runtime.audit.event(
                        "interest_updated",
                        "Interesse explícito na mensagem foi atualizado.",
                        metadata={
                            "interest_id": interest_id,
                            "source": "user_message",
                        },
                    )
                self._print(f"Mensagem {message_id} registrada.")
        elif command == "messages":
            self._print_rows(self.runtime.database.list_messages(limit=20))
        elif command == "replies":
            self._print_rows(
                self.runtime.database.list_messages(sender="agent", limit=20)
            )
        elif command == "cycles":
            self._print_rows(self.runtime.database.list_cycles(limit=20))
        elif command == "projects":
            self._print_rows(self.runtime.database.list_projects(limit=20))
        elif command == "journal":
            self._print_rows(self.runtime.database.list_journal(limit=20))
        elif command == "state":
            state = self.runtime.database.latest_internal_state()
            self._print_json(state.to_dict() if state else {"state": "ainda não registrado"})
        elif command == "files":
            self._print_json({"files": self.runtime.world.list_files()})
        elif command == "stop":
            stopped = self.runtime.supervisor.stop()
            self._print("Agente encerrado." if stopped else "A parada excedeu o tempo limite.")
            return False
        else:
            self._print(f"Comando desconhecido: {raw_command!r}. Use 'help'.")
        return True

    def _print_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            self._print("Nenhum registro.")
            return
        for row in reversed(rows):
            self._print_json(row)

    def _print_json(self, value: object) -> None:
        self._print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))

    def _print(self, text: str) -> None:
        print(text, file=self.stream, flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orionworld",
        description="Ambiente persistente e isolado para um agente autônomo.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SETTINGS_PATH,
        help="arquivo JSON de configuração",
    )
    parser.add_argument(
        "--world-data",
        type=Path,
        default=Path("world_data"),
        help="raiz confiável do mundo isolado",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        runtime = build_runtime(
            settings_path=args.config,
            world_data=args.world_data,
        )
    except (OSError, SettingsError, ValueError) as exc:
        print(f"Falha ao iniciar OrionWorld: {exc}", file=sys.stderr)
        return 2

    interface = TerminalInterface(runtime)
    print("OrionWorld iniciado. Modo simulação ativo." if runtime.settings.simulation else "OrionWorld iniciado. Modo de escrita restrita ativo.")
    print(HELP_TEXT)
    try:
        try:
            runtime.start()
        except InstanceAlreadyRunning as exc:
            print(f"OrionWorld não iniciado: {exc}", file=sys.stderr)
            return 3
        while runtime.supervisor.state is not SupervisorState.STOPPED:
            try:
                command = input("orion> ")
            except EOFError:
                print("Fim da entrada recebido; encerrando o agente.")
                runtime.supervisor.stop()
                break
            if not interface.execute(command):
                break
    except KeyboardInterrupt:
        print("\nInterrupção explícita recebida; encerrando o agente.")
        runtime.supervisor.stop()
    finally:
        runtime.close()
    return 0


__all__ = ["Runtime", "TerminalInterface", "build_runtime", "main"]
