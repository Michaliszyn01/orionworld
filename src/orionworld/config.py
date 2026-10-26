"""Loading and validation for OrionWorld's immutable settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Final


SAFE_WRITABLE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".txt", ".md", ".json", ".svg"}
)
DEFAULT_ALLOWED_EXTENSIONS: Final[tuple[str, ...]] = (
    ".txt",
    ".md",
    ".json",
    ".svg",
)
ALLOWED_MODEL_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"disabled", "mock", "ollama"}
)
LOCAL_OLLAMA_BASE_URL: Final[str] = "http://127.0.0.1:11434"
_CHECKOUT_SETTINGS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "config" / "settings.json"
)
_PACKAGED_SETTINGS_PATH: Final[Path] = Path(__file__).with_name(
    "default_settings.json"
)
DEFAULT_SETTINGS_PATH: Final[Path] = (
    _CHECKOUT_SETTINGS_PATH
    if _CHECKOUT_SETTINGS_PATH.is_file()
    else _PACKAGED_SETTINGS_PATH
)


class SettingsError(ValueError):
    """Raised when a settings file is missing, malformed, or unsafe."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings.

    The tuple used for extensions keeps the frozen dataclass deeply immutable.
    Network and subprocess access are intentionally impossible to enable in this
    first version, including through direct construction of this class.
    """

    simulation: bool = True
    cycle_interval_seconds: float = 10.0
    random_seed: int = 42
    maximum_actions_per_cycle: int = 1
    maximum_file_size_bytes: int = 100_000
    maximum_total_world_size_bytes: int = 10_000_000
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    network_enabled: bool = False
    subprocess_enabled: bool = False
    cognitive_engine_enabled: bool = False
    model_provider: str = "disabled"
    ollama_base_url: str = LOCAL_OLLAMA_BASE_URL
    ollama_model: str = "gemma3:4b"
    model_timeout_seconds: float = 60.0
    maximum_model_response_characters: int = 30_000
    runtime_enabled: bool = True
    dashboard_enabled: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    maximum_model_calls_per_hour: int = 20
    maximum_model_calls_per_day: int = 100
    maximum_actions_per_hour: int = 30
    maximum_writes_per_day: int = 50
    maximum_new_files_per_day: int = 20
    minimum_rest_seconds: float = 300.0
    maximum_active_cycles_before_rest: int = 10
    maximum_consecutive_failures: int = 3
    maximum_consecutive_similar_activities: int = 3
    maximum_active_projects: int = 5
    world_snapshot_interval_cycles: int = 1

    def __post_init__(self) -> None:
        _require_bool("simulation", self.simulation)
        _require_number(
            "cycle_interval_seconds", self.cycle_interval_seconds, minimum=0.001
        )
        _require_int("random_seed", self.random_seed)
        _require_int(
            "maximum_actions_per_cycle",
            self.maximum_actions_per_cycle,
            minimum=1,
        )
        _require_int(
            "maximum_file_size_bytes", self.maximum_file_size_bytes, minimum=1
        )
        _require_int(
            "maximum_total_world_size_bytes",
            self.maximum_total_world_size_bytes,
            minimum=1,
        )
        if self.maximum_total_world_size_bytes < self.maximum_file_size_bytes:
            raise SettingsError(
                "maximum_total_world_size_bytes must be greater than or equal "
                "to maximum_file_size_bytes"
            )

        raw_extensions = self.allowed_extensions
        if isinstance(raw_extensions, (str, bytes)):
            raise SettingsError("allowed_extensions must be a sequence of strings")
        try:
            extensions = tuple(raw_extensions)
        except TypeError as exc:
            raise SettingsError(
                "allowed_extensions must be a sequence of strings"
            ) from exc
        if not extensions:
            raise SettingsError("allowed_extensions cannot be empty")
        if any(not isinstance(extension, str) for extension in extensions):
            raise SettingsError("allowed_extensions must contain only strings")
        if len(set(extensions)) != len(extensions):
            raise SettingsError("allowed_extensions cannot contain duplicates")
        unsafe = set(extensions).difference(SAFE_WRITABLE_EXTENSIONS)
        if unsafe:
            unsafe_text = ", ".join(sorted(repr(item) for item in unsafe))
            raise SettingsError(f"unsafe or unsupported extensions: {unsafe_text}")
        object.__setattr__(self, "allowed_extensions", extensions)

        _require_bool("network_enabled", self.network_enabled)
        _require_bool("subprocess_enabled", self.subprocess_enabled)
        if self.network_enabled:
            raise SettingsError("network_enabled must remain false")
        if self.subprocess_enabled:
            raise SettingsError("subprocess_enabled must remain false")

        _require_bool("cognitive_engine_enabled", self.cognitive_engine_enabled)
        if self.model_provider not in ALLOWED_MODEL_PROVIDERS:
            raise SettingsError("model_provider must be disabled, mock, or ollama")
        if self.ollama_base_url != LOCAL_OLLAMA_BASE_URL:
            raise SettingsError(
                "ollama_base_url must be exactly http://127.0.0.1:11434"
            )
        if not isinstance(self.ollama_model, str) or not self.ollama_model.strip():
            raise SettingsError("ollama_model must be a non-empty string")
        if len(self.ollama_model) > 120:
            raise SettingsError("ollama_model must contain at most 120 characters")
        _require_number(
            "model_timeout_seconds", self.model_timeout_seconds, minimum=0.1
        )
        if float(self.model_timeout_seconds) > 300.0:
            raise SettingsError("model_timeout_seconds must be at most 300")
        _require_int(
            "maximum_model_response_characters",
            self.maximum_model_response_characters,
            minimum=1,
        )
        if self.maximum_model_response_characters > 100_000:
            raise SettingsError(
                "maximum_model_response_characters must be at most 100000"
            )
        if not self.cognitive_engine_enabled and self.model_provider != "disabled":
            raise SettingsError(
                "model_provider must be disabled when cognitive_engine_enabled is false"
            )
        _require_bool("runtime_enabled", self.runtime_enabled)
        _require_bool("dashboard_enabled", self.dashboard_enabled)
        if self.dashboard_host != "127.0.0.1":
            raise SettingsError("dashboard_host must be exactly 127.0.0.1")
        _require_int("dashboard_port", self.dashboard_port, minimum=1)
        if self.dashboard_port > 65_535:
            raise SettingsError("dashboard_port must be at most 65535")
        for name in (
            "maximum_model_calls_per_hour",
            "maximum_model_calls_per_day",
            "maximum_actions_per_hour",
            "maximum_writes_per_day",
            "maximum_new_files_per_day",
            "maximum_active_cycles_before_rest",
            "maximum_consecutive_failures",
            "maximum_consecutive_similar_activities",
            "maximum_active_projects",
            "world_snapshot_interval_cycles",
        ):
            _require_int(name, getattr(self, name), minimum=1)
        if self.maximum_model_calls_per_day < self.maximum_model_calls_per_hour:
            raise SettingsError("daily model limit cannot be lower than hourly limit")
        _require_number("minimum_rest_seconds", self.minimum_rest_seconds, minimum=0.001)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Settings:
        """Build settings from a complete mapping, rejecting unknown keys."""

        expected = {field.name for field in fields(cls)}
        supplied = set(values)
        missing = expected.difference(supplied)
        unknown = supplied.difference(expected)
        if missing:
            raise SettingsError(
                "missing settings: " + ", ".join(sorted(missing))
            )
        if unknown:
            raise SettingsError(
                "unknown settings: "
                + ", ".join(sorted((str(item) for item in unknown), key=str.casefold))
            )
        return cls(**dict(values))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Settings:
        """Compatibility spelling for callers that already hold decoded JSON."""

        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of these settings."""

        result = asdict(self)
        result["allowed_extensions"] = list(self.allowed_extensions)
        return result


def load_settings(path: str | Path | None = None) -> Settings:
    """Read and validate a settings JSON file.

    Loading is deliberately strict so a misspelled or omitted security option
    cannot silently change the runtime policy.
    """

    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    try:
        raw = settings_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SettingsError(f"unable to read settings file: {settings_path}") from exc
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SettingsError(f"invalid JSON in settings file: {settings_path}") from exc
    if not isinstance(decoded, dict):
        raise SettingsError("settings JSON must contain an object")
    return Settings.from_mapping(decoded)


def validate_settings(settings: Settings) -> Settings:
    """Provide an explicit validation hook for dependency-injected settings."""

    if not isinstance(settings, Settings):
        raise SettingsError("settings must be a Settings instance")
    # Construction performs all validation; recreating also protects callers
    # using unusual deserialization mechanisms that bypassed __init__.
    return Settings.from_mapping(settings.to_dict())


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise SettingsError(f"{name} must be a boolean")


def _require_int(name: str, value: object, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SettingsError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise SettingsError(f"{name} must be at least {minimum}")


def _require_number(name: str, value: object, minimum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SettingsError(f"{name} must be a number")
    if not math.isfinite(float(value)) or float(value) < minimum:
        raise SettingsError(f"{name} must be a finite number of at least {minimum}")
