"""Pluggable model providers for the optional cognitive layer."""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .cognitive_models import CognitiveRequest, CognitiveResponse


class ModelProviderError(RuntimeError):
    """A bounded, user-safe provider failure."""


class ModelProvider(Protocol):
    name: str

    def generate(self, request: CognitiveRequest) -> CognitiveResponse:
        """Return untrusted text without executing any proposed operation."""


class DisabledModelProvider:
    name = "disabled"

    def generate(self, request: CognitiveRequest) -> CognitiveResponse:
        raise ModelProviderError("cognitive model provider is disabled")


class MockModelProvider:
    """Deterministic provider for tests and offline demonstrations."""

    name = "mock"

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or ())
        self.requests: list[CognitiveRequest] = []

    def generate(self, request: CognitiveRequest) -> CognitiveResponse:
        self.requests.append(request)
        if self._responses:
            return CognitiveResponse(self._responses.pop(0), self.name)
        if request.kind == "reflection":
            payload = {
                "summary": "A ação foi observada no ambiente isolado.",
                "outcome_assessment": "resultado registrado",
                "learned_information": "O resultado pode orientar escolhas futuras.",
                "detected_problem": None,
                "confidence": 0.7,
                "suggested_interest": None,
                "should_remember": False,
            }
        else:
            payload = {
                "should_act": True,
                "activity_type": "cognitive_observation",
                "title": "Observar com atenção",
                "description": "Examinar os nomes dos arquivos disponíveis no mundo.",
                "motivation": "Buscar contexto sem alterar conteúdo.",
                "expected_value": 0.65,
                "novelty": 0.45,
                "estimated_cost": 0.1,
                "risk": "low",
                "requested_tool": "list_world_files",
                "tool_arguments": {},
                "plan": ["Listar somente arquivos permitidos do mundo."],
                "expected_result": "Uma lista limitada de caminhos relativos.",
            }
        return CognitiveResponse(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            self.name,
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise HTTPError(req.full_url, code, "redirects are disabled", headers, fp)


def _local_transport(request: Request, timeout: float, byte_limit: int) -> bytes:
    opener = build_opener(_NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        return response.read(byte_limit + 1)


class OllamaModelProvider:
    """Trusted-core adapter restricted to one loopback Ollama endpoint."""

    name = "ollama"
    _BASE_URL = "http://127.0.0.1:11434"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        maximum_response_characters: int,
        transport: Callable[[Request, float, int], bytes] = _local_transport,
    ) -> None:
        if base_url != self._BASE_URL:
            raise ValueError("Ollama is restricted to http://127.0.0.1:11434")
        self._endpoint = self._BASE_URL + "/api/generate"
        self._model = model
        self._timeout = float(timeout_seconds)
        self._character_limit = maximum_response_characters
        self._transport = transport

    def generate(self, request: CognitiveRequest) -> CognitiveResponse:
        body = json.dumps(
            {
                "model": self._model,
                "prompt": request.prompt,
                "stream": False,
                "format": "json",
                "think": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            raw = self._transport(
                http_request,
                self._timeout,
                self._character_limit * 4,
            )
            if len(raw) > self._character_limit * 4:
                raise ModelProviderError("model response exceeded the size limit")
            decoded = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, URLError) as exc:
            raise ModelProviderError("local Ollama request failed") from exc
        if not isinstance(decoded, dict) or not isinstance(decoded.get("response"), str):
            raise ModelProviderError("local Ollama returned an invalid envelope")
        text = decoded["response"]
        if len(text) > self._character_limit:
            raise ModelProviderError("model response exceeded the size limit")
        return CognitiveResponse(text=text, provider=self.name)


__all__ = [
    "DisabledModelProvider",
    "MockModelProvider",
    "ModelProvider",
    "ModelProviderError",
    "OllamaModelProvider",
]
