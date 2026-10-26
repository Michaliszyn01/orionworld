"""Fail-closed security policy and auditable security violations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import NoReturn, TypeAlias


AuditCallback: TypeAlias = Callable[[str, str, dict[str, object]], None]


@dataclass(frozen=True, slots=True)
class SecurityEvent:
    """Public, functional description of a blocked operation."""

    operation: str
    reason: str
    category: str
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "operation": self.operation,
            "reason": self.reason,
            "category": self.category,
            "details": dict(self.details),
        }


class SecurityViolation(PermissionError):
    """Base exception for an operation rejected by the security boundary."""

    category = "security"

    def __init__(
        self,
        operation: str,
        reason: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.operation = operation
        self.reason = reason
        self.details = dict(details or {})
        super().__init__(f"{operation} blocked: {reason}")

    @property
    def event(self) -> SecurityEvent:
        return SecurityEvent(
            operation=self.operation,
            reason=self.reason,
            category=self.category,
            details=self.details,
        )


class PathSecurityViolation(SecurityViolation):
    category = "path"


class WorldLimitViolation(SecurityViolation):
    category = "world_limit"


class NetworkSecurityViolation(SecurityViolation):
    category = "network"


class SubprocessSecurityViolation(SecurityViolation):
    category = "subprocess"


class SystemCommandSecurityViolation(SecurityViolation):
    category = "system_command"


@dataclass(frozen=True, slots=True)
class SecurityPolicy:
    """Central deny policy for capabilities excluded from the MVP.

    ``audit_callback`` receives ``(operation, reason, details)``. Callback
    failures are contained so they can never turn a denial into an allowance or
    replace the original security exception.
    """

    network_enabled: bool = False
    subprocess_enabled: bool = False
    audit_callback: AuditCallback | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.network_enabled) is not bool:
            raise ValueError("network_enabled must be a boolean")
        if type(self.subprocess_enabled) is not bool:
            raise ValueError("subprocess_enabled must be a boolean")
        if self.network_enabled:
            raise ValueError("network access cannot be enabled in this version")
        if self.subprocess_enabled:
            raise ValueError("subprocess access cannot be enabled in this version")

    def deny_path(
        self,
        operation: str,
        path: object,
        reason: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> NoReturn:
        payload = dict(details or {})
        payload["path"] = str(path)
        self._deny(PathSecurityViolation, operation, reason, payload)

    def block_path(
        self,
        operation: str,
        path: object,
        reason: str,
    ) -> NoReturn:
        """Alias with terminology convenient for Supervisor callers."""

        self.deny_path(operation, path, reason)

    def deny_world_limit(
        self,
        operation: str,
        path: object,
        reason: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> NoReturn:
        payload = dict(details or {})
        payload["path"] = str(path)
        self._deny(WorldLimitViolation, operation, reason, payload)

    def require_network_denied(
        self, operation: str = "network_access", target: object | None = None
    ) -> NoReturn:
        details = {} if target is None else {"target": str(target)}
        self._deny(
            NetworkSecurityViolation,
            operation,
            "network access is disabled",
            details,
        )

    def check_network(
        self, operation: str = "network_access", target: object | None = None
    ) -> NoReturn:
        self.require_network_denied(operation, target)

    def require_subprocess_denied(
        self, operation: str = "subprocess", command: object | None = None
    ) -> NoReturn:
        details = {} if command is None else {"command": str(command)}
        self._deny(
            SubprocessSecurityViolation,
            operation,
            "subprocess execution is disabled",
            details,
        )

    def check_subprocess(
        self, operation: str = "subprocess", command: object | None = None
    ) -> NoReturn:
        self.require_subprocess_denied(operation, command)

    def require_system_command_denied(
        self, operation: str = "system_command", command: object | None = None
    ) -> NoReturn:
        details = {} if command is None else {"command": str(command)}
        self._deny(
            SystemCommandSecurityViolation,
            operation,
            "system commands are disabled",
            details,
        )

    def check_system_command(
        self, operation: str = "system_command", command: object | None = None
    ) -> NoReturn:
        self.require_system_command_denied(operation, command)

    def _deny(
        self,
        violation_type: type[SecurityViolation],
        operation: str,
        reason: str,
        details: Mapping[str, object],
    ) -> NoReturn:
        payload = dict(details)
        payload["category"] = violation_type.category
        payload["blocked"] = True
        if self.audit_callback is not None:
            try:
                self.audit_callback(operation, reason, payload)
            except Exception:
                # Security must remain fail-closed even if observability fails.
                pass
        raise violation_type(operation, reason, payload)


# Concise aliases for callers that do not need the longer class names.
PathViolation = PathSecurityViolation
NetworkViolation = NetworkSecurityViolation
SubprocessViolation = SubprocessSecurityViolation
SystemCommandViolation = SystemCommandSecurityViolation

