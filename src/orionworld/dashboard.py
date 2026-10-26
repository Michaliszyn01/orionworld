"""Local-only observation and fixed-control dashboard using the standard library."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import time
from typing import Any
from urllib.parse import urlparse

from .audit import AuditLogger
from .database import Database
from .world import WorldSandbox
from .world_services import SafeSvgRenderer, SvgRejected


_PAGES = {
    "/": "Visão geral",
    "/activity": "Atividade",
    "/projects": "Projetos",
    "/memories": "Memórias",
    "/interests": "Interesses",
    "/messages": "Mensagens",
    "/world": "Mundo",
    "/journal": "Diário",
}
_READ_ENDPOINTS = {
    "/api/cycles": "cycles",
    "/api/intentions": "intentions",
    "/api/actions": "actions",
    "/api/projects": "projects",
    "/api/memories": "memories",
    "/api/interests": "interests",
    "/api/messages": "messages",
    "/api/journal": "journal",
    "/api/world": "world",
}
_CONTROL_PATHS = {
    "/api/control/pause": "pause",
    "/api/control/resume": "resume",
    "/api/control/reflection": "request_reflection",
    "/api/control/rest": "enter_rest",
    "/api/control/stop": "stop",
}


def render_dashboard_page(
    title: str,
    status: dict[str, Any],
    data: object,
    csrf_token: str,
) -> str:
    status_text = escape(json.dumps(status, ensure_ascii=False, default=str))
    data_text = escape(json.dumps(data, ensure_ascii=False, default=str))
    navigation = " ".join(
        f'<a href="{escape(path)}">{escape(label)}</a>'
        for path, label in _PAGES.items()
    )
    controls = ""
    if title == "Visão geral":
        controls = (
            "<section><h3>Controles fechados</h3>"
            + "".join(
                f'<button data-control="{path}">{label}</button>'
                for path, label in (
                    ("pause", "Pausar"), ("resume", "Retomar"),
                    ("reflection", "Solicitar reflexão"), ("rest", "Repousar"),
                    ("stop", "Encerrar"),
                )
            )
            + "</section>"
        )
    return (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<meta name='csrf-token' content='{escape(csrf_token)}'>"
        "<title>OrionWorld</title><link rel='stylesheet' href='/assets/app.css'>"
        "<script src='/assets/app.js' defer></script></head><body><header>"
        "<h1>OrionWorld</h1><nav>" + navigation + "</nav></header><main><h2>" +
        escape(title) + "</h2><h3>Status</h3><pre>" + status_text +
        "</pre><h3>Dados</h3><pre>" + data_text + "</pre>" + controls +
        "</main></body></html>"
    )


class DashboardServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        database: Database,
        world: WorldSandbox,
        audit: AuditLogger,
        status_provider: Callable[[], dict[str, Any]],
        requests_per_minute: int = 120,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("dashboard deve usar exclusivamente 127.0.0.1")
        self.host = host
        self.port = port
        self.database = database
        self.world = world
        self.audit = audit
        self.status_provider = status_provider
        self.csrf_token = secrets.token_urlsafe(32)
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._rate_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._svg = SafeSvgRenderer()

    @property
    def address(self) -> tuple[str, int] | None:
        if self._server is None:
            return None
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> tuple[str, int]:
        if self._server is not None:
            raise RuntimeError("dashboard já iniciado")
        handler = self._handler_type()
        server = ThreadingHTTPServer((self.host, self.port), handler)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="orionworld-dashboard",
            daemon=True,
        )
        self._thread.start()
        return self.address or (self.host, self.port)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def _allowed_request(self, address: str) -> bool:
        now = time.monotonic()
        with self._rate_lock:
            values = self._requests[address]
            while values and now - values[0] >= 60.0:
                values.popleft()
            if len(values) >= self.requests_per_minute:
                return False
            values.append(now)
            return True

    def _handler_type(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "OrionWorldLocal/0.3"

            def log_message(self, *_: object) -> None:
                return

            def do_GET(self) -> None:
                if not dashboard._allowed_request(self.client_address[0]):
                    self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "rate limit"})
                    return
                path = urlparse(self.path).path
                if path in _PAGES:
                    self._html(HTTPStatus.OK, self._page(_PAGES[path]))
                elif path == "/assets/app.css":
                    self._send(HTTPStatus.OK, "text/css; charset=utf-8", self._css())
                elif path == "/assets/app.js":
                    self._send(HTTPStatus.OK, "text/javascript; charset=utf-8", self._js())
                elif path == "/api/status":
                    self._json(HTTPStatus.OK, dashboard.status_provider())
                elif path in _READ_ENDPOINTS:
                    self._json(HTTPStatus.OK, self._read_data(_READ_ENDPOINTS[path]))
                elif path.startswith("/api/drawings/"):
                    self._drawing(path.removeprefix("/api/drawings/"))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:
                if not dashboard._allowed_request(self.client_address[0]):
                    self._json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "rate limit"})
                    return
                if not secrets.compare_digest(
                    self.headers.get("X-CSRF-Token", ""), dashboard.csrf_token
                ):
                    self._json(HTTPStatus.FORBIDDEN, {"error": "csrf"})
                    return
                path = urlparse(self.path).path
                try:
                    payload = self._body()
                    if path == "/api/messages":
                        content = payload.get("content")
                        if not isinstance(content, str):
                            raise ValueError("content obrigatório")
                        item_id = dashboard.database.save_message(
                            "user", content, metadata={"authority": False, "source": "dashboard"}
                        )
                    elif path in _CONTROL_PATHS:
                        item_id = dashboard.database.enqueue_control_command(
                            _CONTROL_PATHS[path], metadata={"source": "dashboard"}
                        )
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                        return
                except (ValueError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(HTTPStatus.ACCEPTED, {"id": item_id})

            def _body(self) -> dict[str, Any]:
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("content-type inválido")
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > 16_384:
                    raise ValueError("corpo excedeu o limite")
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("corpo deve ser objeto")
                return value

            def _read_data(self, kind: str) -> dict[str, Any]:
                values = {
                    "cycles": dashboard.database.list_cycles,
                    "intentions": dashboard.database.list_intentions,
                    "actions": dashboard.database.list_actions,
                    "projects": dashboard.database.list_projects,
                    "memories": dashboard.database.list_memories,
                    "interests": dashboard.database.list_interests,
                    "messages": dashboard.database.list_messages,
                    "journal": dashboard.database.list_journal,
                    "world": lambda limit=100: dashboard.database.list_world_snapshots(limit),
                }
                return {kind: values[kind](limit=100)}

            def _drawing(self, name: str) -> None:
                if "/" in name or "\\" in name or not name.casefold().endswith(".svg"):
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid drawing"})
                    return
                relative = f"drawings/{name}"
                try:
                    source = dashboard.world.read_text(relative)
                    sanitized = dashboard._svg.sanitize(source)
                except (OSError, ValueError, SvgRejected) as exc:
                    dashboard.audit.event(
                        "svg_rejected",
                        "SVG não foi renderizado por falha de sanitização.",
                        metadata={"path": relative, "reason": str(exc)[:160]},
                    )
                    self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"path": relative, "rendered": False})
                    return
                self._send(HTTPStatus.OK, "image/svg+xml; charset=utf-8", sanitized)

            def _page(self, title: str) -> str:
                kind_by_title = {
                    "Atividade": "cycles", "Projetos": "projects",
                    "Memórias": "memories", "Interesses": "interests",
                    "Mensagens": "messages", "Mundo": "world", "Diário": "journal",
                }
                kind = kind_by_title.get(title)
                data = {} if kind is None else self._read_data(kind)
                return render_dashboard_page(
                    title,
                    dashboard.status_provider(),
                    data,
                    dashboard.csrf_token,
                )

            @staticmethod
            def _css() -> str:
                return "body{font:16px system-ui;margin:2rem;background:#10151d;color:#eef}a{color:#8bd;margin-right:1rem}pre{white-space:pre-wrap}"

            @staticmethod
            def _js() -> str:
                return "document.querySelectorAll('[data-control]').forEach(b=>b.addEventListener('click',()=>fetch('/api/control/'+b.dataset.control,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':document.querySelector('meta[name=csrf-token]').content},body:'{}'})));"

            def _html(self, status: HTTPStatus, body: str) -> None:
                self._send(status, "text/html; charset=utf-8", body)

            def _json(self, status: HTTPStatus, value: object) -> None:
                self._send(status, "application/json; charset=utf-8", json.dumps(value, ensure_ascii=False, default=str))

            def _send(self, status: HTTPStatus, content_type: str, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

        return Handler


__all__ = ["DashboardServer", "render_dashboard_page"]
