"""Explicit registry of caller-provided backend declarations, API version 1.

No backends are registered by default. Registration takes immutable data rather
than an implementation callback: no entry points, plugin imports, driver probes,
device initialization, network requests or fallback paths are invoked. Selection
matches declared capabilities and does not verify availability or admit training.
Runtime/provider integration requires a separate, future protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from threading import RLock

from .capabilities import (
    CapabilityError, CapabilityMatch, DeviceCapabilities, DeviceIdentity, OperatorRequirement,
    _identifier, _records, _requirements, _text, _version, match_capabilities,
)

BACKEND_API_VERSION = 1
MAX_BACKENDS = 128
MAX_DEVICES_PER_BACKEND = 256


class RegistryError(ValueError):
    """A registry refusal with a stable ``code``."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    backend_id: str
    name: str
    api_version: int = BACKEND_API_VERSION
    devices: tuple[DeviceCapabilities, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.backend_id, "backend-id")
        _text(self.name, "backend-name")
        _version(self.api_version, "backend-api-version")
        devices = _records(self.devices, DeviceCapabilities, MAX_DEVICES_PER_BACKEND, "devices")
        if any(device.identity.backend_id != self.backend_id for device in devices):
            raise CapabilityError("device-backend-mismatch")
        if len({device.identity.device_id for device in devices}) != len(devices):
            raise CapabilityError("duplicate-device-id")
        object.__setattr__(self, "devices", devices)


@dataclass(frozen=True, slots=True)
class DeviceRejection:
    """Requirements not supported by one device's declarations."""

    identity: DeviceIdentity
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.identity) is not DeviceIdentity:
            raise CapabilityError("invalid-device-identity")
        reasons = CapabilityMatch(self.reasons).reasons
        if not reasons:
            raise CapabilityError("empty-device-rejection")
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """All declared matches in registration order, with no preferred vendor.

``matches`` are declarations, not opened devices. ``reasons`` describe selection
scope refusals; ``rejected`` retains per-device unsupported reasons. A nonempty
match list says nothing about evidence, hardware availability or training.
Identities cannot repeat within or across outcomes. Global scope refusals cannot
coexist with matches; unsupported individual devices belong in ``rejected``.
    """

    matches: tuple[DeviceCapabilities, ...] = ()
    rejected: tuple[DeviceRejection, ...] = ()
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        maximum = MAX_BACKENDS * MAX_DEVICES_PER_BACKEND
        object.__setattr__(self, "matches", _records(self.matches, DeviceCapabilities, maximum, "matches"))
        object.__setattr__(self, "rejected", _records(self.rejected, DeviceRejection, maximum, "rejections"))
        object.__setattr__(self, "reasons", CapabilityMatch(self.reasons).reasons)
        matches = {(value.identity.backend_id, value.identity.device_id) for value in self.matches}
        rejections = {(value.identity.backend_id, value.identity.device_id) for value in self.rejected}
        if len(matches) != len(self.matches):
            raise CapabilityError("duplicate-matched-device")
        if len(rejections) != len(self.rejected):
            raise CapabilityError("duplicate-rejected-device")
        if matches & rejections:
            raise CapabilityError("device-both-matched-and-rejected")
        if matches and self.reasons:
            raise CapabilityError("matches-with-global-refusal")


class BackendRegistry:
    """An initially empty, explicit registry; no global or installed providers.

Registration is atomic and refuses duplicate IDs. Selection uses an immutable
snapshot, so concurrent explicit registrations cannot partially enter a result.
The lock protects this Python registry only; it makes no native runtime claim.
    """

    def __init__(self) -> None:
        self._backends: dict[str, BackendDescriptor] = {}
        self._lock = RLock()

    def register(self, backend: BackendDescriptor) -> None:
        if type(backend) is not BackendDescriptor:
            raise RegistryError("invalid-backend-descriptor")
        if backend.api_version != BACKEND_API_VERSION:
            raise RegistryError("unsupported-backend-api-version")
        with self._lock:
            if backend.backend_id in self._backends:
                raise RegistryError("duplicate-backend-id")
            if len(self._backends) >= MAX_BACKENDS:
                raise RegistryError("backend-registry-full")
            self._backends[backend.backend_id] = backend

    def registered(self) -> tuple[BackendDescriptor, ...]:
        """Return only explicitly registered declarations, without discovery."""
        with self._lock:
            return tuple(self._backends.values())

    def select(self, requirements: Iterable[OperatorRequirement], *,
               backend_id: str | None = None,
               device_id: str | None = None) -> SelectionResult:
        """Match declarations on each device; never combine devices or fall back.

A device filter requires a backend ID because device IDs are backend-scoped.
An explicit backend that is missing or unsupported cannot select another one.
        """
        requirements = _requirements(requirements)
        if backend_id is not None:
            _identifier(backend_id, "backend-id")
        if device_id is not None:
            _text(device_id, "device-id")
            if backend_id is None:
                raise RegistryError("device-filter-requires-backend")
        backends = self.registered()
        if backend_id is not None:
            backends = tuple(value for value in backends if value.backend_id == backend_id)
            if not backends:
                return SelectionResult(reasons=("backend-not-registered",))
        elif not backends:
            return SelectionResult(reasons=("no-backends-registered",))
        devices = tuple(device for backend in backends for device in backend.devices)
        if device_id is not None:
            devices = tuple(device for device in devices if device.identity.device_id == device_id)
            if not devices:
                return SelectionResult(reasons=("device-not-declared",))
        elif not devices:
            return SelectionResult(reasons=("no-devices-declared",))
        matches = []
        rejected = []
        for device in devices:
            result = match_capabilities(device, requirements)
            if result.supported:
                matches.append(device)
            else:
                rejected.append(DeviceRejection(device.identity, result.reasons))
        return SelectionResult(tuple(matches), tuple(rejected),
                               () if matches else ("no-declared-device-matches",))
